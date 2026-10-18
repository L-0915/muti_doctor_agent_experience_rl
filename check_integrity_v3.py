"""Integrity checks for corpus v3 and the training files rebuilt from it.

Every check is evidence for one claim the data has to support:

  schema       the parquet still speaks the veRL field set trainers read
  counts       the files match the reports written beside them
  leakage      a case never appears in two splits, in either view
  first_ask    the same opening question never straddles a split
  hidden       nothing a patient cannot see reaches the prompt
  conclusion   the reward reference is a real, usable conclusion
  budget       turn_budget is inside the range the filters allow
  quality      blocked cases are absent from the SFT set
  sft          no system message, every assistant turn is an action object
  corpus       the cleaned corpus is structurally clean and duplicate-free

Exit code 0 means every check passed; anything else means the build is not
shippable.

Usage:
  python scripts/check_integrity_v3.py
  python scripts/check_integrity_v3.py --corpus data/processed_v3 ^
      --parquet data/sft_rl_v3 --sft data/sft_rl_v3
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402
import corpus_v3_rules as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/processed_v3"
DEFAULT_PARQUET = ROOT / "data/sft_rl_v3"
DEFAULT_SFT = ROOT / "data/sft_rl_v3"

SPLITS = ("train", "dev", "test")
ACTIONS = {"WEN", "TALK", "DIG"}
HIDDEN_KEYS = ("sample_id", "case_id", "quality", "turn_budget", "full_dialogue",
               "reference_conclusion", "presenting_problem", "doctor_turns_ok")
PLACEHOLDER = "<IMAGE_UNAVAILABLE>"


class Report:
    def __init__(self) -> None:
        self.rows: list = []

    def check(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((group, name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {group:<10} {name}" + (f"  {detail}" if detail else ""), flush=True)
        return bool(ok)

    @property
    def failed(self) -> list:
        return [row for row in self.rows if not row[2]]

    def summary(self) -> str:
        passed = sum(1 for row in self.rows if row[2])
        return f"{passed}/{len(self.rows)} 项通过"


def read_parquet_batches(path: Path, columns=None, batch_size: int = 2000):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns)


def parquet_schema(path: Path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema_arrow


EXPECTED_TOP = {"prompt", "ground_truth", "extra_info", "data_source", "index"}
EXPECTED_GROUND = {"reference_conclusion", "diagnosis", "presenting_problem",
                   "department", "department_path"}
EXPECTED_EXTRA = {"sample_id", "case_id", "patient_state", "profile", "quality",
                  "turn_budget", "conclusion_turn_index", "post_conclusion_turns",
                  "post_conclusion_has_new_info"}
EXPECTED_PROFILE = {"dialogue_turns", "doctor_turns", "wen", "dig", "talk"}
EXPECTED_QUALITY = {"rules", "image_dependent", "doctor_turns_ok", "pp_is_department",
                    "conclusion_available"}


def check_schema(report: Report, path: Path) -> None:
    schema = parquet_schema(path)
    names = set(schema.names)
    report.check("schema", "parquet 顶层字段", names == EXPECTED_TOP,
                 f"{sorted(names)}")
    for field in schema:
        if field.name == "ground_truth":
            report.check("schema", "ground_truth 字段", set(field.type.names) == EXPECTED_GROUND,
                         f"{sorted(field.type.names)}")
        if field.name == "extra_info":
            report.check("schema", "extra_info 字段", set(field.type.names) == EXPECTED_EXTRA,
                         f"{sorted(field.type.names)}")
            for inner in field.type:
                if inner.name == "profile":
                    report.check("schema", "extra_info.profile 字段",
                                 set(inner.type.names) == EXPECTED_PROFILE)
                if inner.name == "quality":
                    report.check("schema", "extra_info.quality 字段",
                                 set(inner.type.names) == EXPECTED_QUALITY)
                if inner.name == "patient_state":
                    report.check("schema", "extra_info.patient_state 字段",
                                 set(inner.type.names) == {"full_dialogue", "symptoms"})


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--max-doctor-turns", type=int, default=B.DEFAULT_MAX_DOCTOR_TURNS)
    parser.add_argument("--sample", type=Path, default=ROOT / "data/sft_rl_v3/sft10k",
                        help="stratified SFT sample directory to verify")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="skip the full corpus rescan (it is the slow part)")
    args = parser.parse_args()

    report = Report()
    max_turns = args.max_doctor_turns

    # ---------------------------------------------------------------- corpus
    corpus_cases = 0
    corpus_quality = collections.Counter()
    corpus_fingerprints: dict = {}
    corpus_duplicates = 0
    if not args.skip_corpus:
        for path, line_no, row in R.iter_rows(args.corpus):
            corpus_cases += 1
            turns = R.turns_of(row)
            problems = []
            if not row.get("case_id"):
                problems.append("no_case_id")
            marker = row.get("quality") or {}
            if not marker:
                problems.append("no_quality")
            else:
                corpus_quality["passed" if marker.get("passed") else "blocked"] += 1
                unknown = set(marker.get("rules") or []) - set(R.RULE_TITLES)
                if unknown:
                    problems.append(f"unknown_rules={sorted(unknown)}")
            if any(not str(t.get("text") or "").strip() for t in turns):
                problems.append("empty_turn")
            if any("\ufffd" in str(t.get("text") or "") for t in turns):
                problems.append("replacement_char")
            roles = [t.get("role") for t in turns]
            if "doctor" not in roles or "patient" not in roles:
                problems.append("missing_role")
            digest = R.fingerprint(turns)
            if digest in corpus_fingerprints:
                corpus_duplicates += 1
            corpus_fingerprints.setdefault(digest, row.get("case_id"))
            if problems:
                report.check("corpus", f"行 {path.name}:{line_no}", False, ",".join(problems))
                if len(report.failed) > 5:
                    break
        report.check("corpus", "对话结构干净（无空轮/无 U+FFFD/角色齐全）",
                     not any(row[0] == "corpus" and not row[2] for row in report.rows),
                     f"{corpus_cases:,} 条")
        report.check("corpus", "轮次指纹无重复", corpus_duplicates == 0,
                     f"{corpus_cases:,} 条，重复 {corpus_duplicates}")
        report.check("corpus", "每条带 quality 字段", corpus_cases > 0 and
                     (corpus_quality["passed"] + corpus_quality["blocked"]) == corpus_cases,
                     f"passed {corpus_quality['passed']:,} / "
                     f"blocked {corpus_quality['blocked']:,}")

    # ------------------------------------------------------------- alignment
    census = load_json(ROOT / "data/audit_v3/census.json") or {}
    cleaner_report = load_json(args.corpus / "_report.json") or {}
    if census and cleaner_report:
        for name, before, after in (
            ("清洗前对话数", census["totals"]["cases"], cleaner_report["before"]["cases"]),
            ("删除对话数", census["drop_decision"]["cases_dropped"],
             cleaner_report["dropped"]["cases"]),
            ("存活对话数", census["projection"]["cases_kept"],
             cleaner_report["written"]["cases"]),
            ("存活轮数", census["projection"]["turns_kept"],
             cleaner_report["after"]["turns"]),
        ):
            report.check("alignment", f"普查与清洗一致：{name}", before == after,
                         f"普查 {before:,} / 清洗 {after:,}")
        for code in R.DROP_RULES:
            if census["drop_decision"]["primary_rule"].get(code, 0) !=                     cleaner_report["dropped"]["by_rule"].get(code, 0):
                report.check("alignment", f"主删除规则一致 {code}", False,
                             f"普查 {census['drop_decision']['primary_rule'].get(code, 0)} / "
                             f"清洗 {cleaner_report['dropped']['by_rule'].get(code, 0)}")
        report.check("alignment", "主删除规则分布一致",
                     all(census["drop_decision"]["primary_rule"].get(code, 0)
                         == cleaner_report["dropped"]["by_rule"].get(code, 0)
                         for code in R.DROP_RULES))
        # R11 sends a patient turn written in a doctor's voice to the manual
        # queue instead of rewriting it, so the cleaner books it with the
        # manual/flag rows while the census counts it under repairs.
        mismatch = [code for code in R.S2_RULES if code != "R11"
                    if census["repairs"][code]["hits"]
                    != cleaner_report["repaired"]["by_rule_turns"].get(code, 0)]
        report.check("alignment", "修复轮数按规则一致", not mismatch, f"{mismatch}")
        report.check("alignment", "人工队列（R11）按规则一致",
                     census["repairs"]["R11"]["hits"]
                     == cleaner_report["manual_review"]["rows"],
                     f"普查 {census['repairs']['R11']['hits']} / "
                     f"清洗人工队列 {cleaner_report['manual_review']['rows']}")
        mismatch = [code for code in R.FLAG_RULES
                    if census["flags"][code]["hits"]
                    != cleaner_report["flagged"]["by_rule_cases"].get(code, 0)]
        report.check("alignment", "标记对话数按规则一致", not mismatch, f"{mismatch}")
        if corpus_cases:
            report.check("counts", "语料对话数与清洗报告一致",
                         corpus_cases == cleaner_report["written"]["cases"],
                         f"{corpus_cases:,} / 报告 {cleaner_report['written']['cases']:,}")
    else:
        report.check("alignment", "普查与清洗报告可读", False,
                     "缺少 census.json 或清洗 _report.json")

    # --------------------------------------------------------------- parquet
    check_schema(report, args.parquet / "train.parquet")
    parquet_report = load_json(args.parquet / "_report.json") or {}
    split_cases: dict = {}
    split_samples: dict = {}
    first_ask: dict = {}
    # The cleaner's group key is the 40-character window at offsets 0/40/80 of
    # the opening; a group that spans splits is the near-duplicate leak the
    # exact-text check cannot see.
    first_ask_window: dict = {}
    first_ask_exact: dict = {}
    all_rows = 0
    budget_bad = 0
    prompt_shape_bad = 0
    leak_reference = 0
    leak_hidden = 0
    leak_doctor = 0
    leak_patient = 0
    conclusion_bad = 0
    no_conclusion_rows = 0
    profile_bad = 0
    quality_bad = 0
    blocked_in_parquet = set()

    for split in SPLITS:
        path = args.parquet / f"{split}.parquet"
        if not path.exists():
            report.check("counts", f"{split}.parquet 存在", False)
            continue
        cases = set()
        samples = set()
        rows = 0
        seen_dupes = 0
        batch_source = read_parquet_batches(path)
        for batch in batch_source:
            for row in batch.to_pylist():
                rows += 1
                all_rows += 1
                extra = row["extra_info"]
                truth = row["ground_truth"]
                profile = extra["profile"]
                marker = extra["quality"]
                dialogue = extra["patient_state"]["full_dialogue"]
                case_id = extra["case_id"]
                cases.add(case_id)
                if extra["sample_id"] in samples:
                    seen_dupes += 1
                samples.add(extra["sample_id"])

                # prompt holds the opening patient turn and nothing else. The
                # counters stay apart because only the conclusion in the prompt
                # measures an answer leak; a later turn quoted inside the opening
                # is a segmentation artefact worth its own number.
                prompt = row["prompt"]
                visible = prompt[0]["content"] if prompt else ""
                if len(prompt) != 1 or prompt[0]["role"] != "user":
                    prompt_shape_bad += 1
                elif not dialogue or dialogue[0]["role"] != "patient" \
                        or dialogue[0]["content"] != visible:
                    prompt_shape_bad += 1
                if truth["reference_conclusion"] and truth["reference_conclusion"] in visible:
                    leak_reference += 1
                if any(k in visible for k in HIDDEN_KEYS):
                    leak_hidden += 1
                for index, turn in enumerate(dialogue):
                    if index == 0:
                        continue
                    content = turn["content"]
                    if turn["role"] == "doctor" and len(content) >= 8 and content in visible:
                        leak_doctor += 1
                    if turn["role"] == "patient" and len(content) >= 12 and content in visible:
                        leak_patient += 1

                # The reward reference must be the turn the engine's own
                # conclusion definition would pick, and a case without one must
                # say so in its markers (F14 / conclusion_available=false).
                reference = (truth["reference_conclusion"] or "").strip()
                doctor_texts = [t["content"] for t in dialogue if t["role"] == "doctor"]
                if bool(marker["conclusion_available"]) != bool(reference):
                    conclusion_bad += 1
                elif reference:
                    if not B.is_substantive_doctor(reference) or reference not in doctor_texts:
                        conclusion_bad += 1
                else:
                    no_conclusion_rows += 1

                # profile arithmetic and the rollout budget
                doctor_turns = sum(1 for t in dialogue if t["role"] == "doctor")
                if profile["dialogue_turns"] != len(dialogue) \
                        or profile["doctor_turns"] != doctor_turns \
                        or profile["wen"] + profile["dig"] + profile["talk"] != doctor_turns:
                    profile_bad += 1
                budget = extra["turn_budget"]
                if not isinstance(budget, int) \
                        or budget < B.TURN_BUDGET_FLOOR \
                        or budget > max_turns \
                        or budget < min(doctor_turns, max_turns) \
                        or doctor_turns > max_turns:
                    budget_bad += 1

                # quality markers are coherent with the case they describe
                rules = set(marker["rules"])
                if rules - set(R.RULE_TITLES) \
                        or bool(marker["image_dependent"]) != ("F03" in rules) \
                        or marker["doctor_turns_ok"] != (2 <= doctor_turns <= max_turns):
                    quality_bad += 1
                if rules & R.BLOCKING_FLAGS:
                    blocked_in_parquet.add(case_id)
                # F14 marks exactly the cases the engine finds no conclusion for.
                if ("F14" in rules) != (not marker["conclusion_available"]):
                    quality_bad += 1

                key = "".join((prompt[0]["content"] or "").split()) if prompt else ""
                first_ask.setdefault(key, set()).add(split)
                # The cleaner groups cases by exactly these keys, so the check
                # reads them from the same function instead of re-deriving a
                # near-duplicate measure of its own.
                for token in R.first_ask_keys([{"role": "patient", "text": visible}]):
                    if len(token) == R.FIRST_ASK_WINDOW:
                        first_ask_window.setdefault(token, set()).add(split)
                    else:
                        first_ask_exact.setdefault(token, set()).add(split)

        split_cases[split] = cases
        split_samples[split] = samples
        expected = (parquet_report.get("episodes") or {}).get(split)
        report.check("counts", f"{split}.parquet 行数与报告一致",
                     expected is None or expected == rows,
                     f"{rows:,} / 报告 {expected}")
        report.check("leakage", f"{split}.parquet sample_id 唯一", seen_dupes == 0,
                     f"重复 {seen_dupes}")
        print(f"--- {split}: {rows:,} 行", flush=True)

    total = sum(len(v) for v in split_cases.values())
    report.check("counts", "parquet 总行数与报告一致",
                 not parquet_report or parquet_report.get("episodes_total") == total,
                 f"{total:,} / 报告 {parquet_report.get('episodes_total')}")

    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        shared = split_cases.get(left, set()) & split_cases.get(right, set())
        report.check("leakage", f"case_id 跨 split 无交集 {left}-{right}",
                     not shared, f"{len(shared)} 个")
        shared_samples = split_samples.get(left, set()) & split_samples.get(right, set())
        report.check("leakage", f"sample_id 跨 split 无交集 {left}-{right}",
                     not shared_samples, f"{len(shared_samples)} 个")

    substantive = {key: splits for key, splits in first_ask.items() if len(key) >= 12}
    cross = {key: splits for key, splits in substantive.items() if len(splits) > 1}
    repeated = {key: splits for key, splits in substantive.items()}
    multiplicity = collections.Counter()
    for key, splits in repeated.items():
        multiplicity[len(splits)] += 1
    report.check("first_ask", "实质首问不跨 split", not cross,
                 f"{len(cross)} 条跨 split；实质首问去重后 {len(substantive):,} 条"
                 f"（重复 {sum(v for k, v in multiplicity.items() if k > 1):,}）")
    print(f"      首问重复分布（按出现过的 split 数）：{dict(multiplicity)}", flush=True)
    cross_windows = {key: splits for key, splits in first_ask_window.items()
                     if len(splits) > 1}
    report.check("first_ask", "首问 40 字窗口不跨 split", not cross_windows,
                 f"{len(cross_windows)} 条窗口跨 split；"
                 f"40 字窗口去重后 {len(first_ask_window):,} 条")
    cross_exact = {key: splits for key, splits in first_ask_exact.items()
                   if len(splits) > 1}
    report.check("first_ask", "首问规范化全文不跨 split（<40 字首问）", not cross_exact,
                 f"{len(cross_exact)} 条；精确键 {len(first_ask_exact):,} 条")

    report.check("hidden", "prompt 形态与隐藏字段", prompt_shape_bad == 0 and leak_hidden == 0,
                 f"形态 {prompt_shape_bad} 行 / 隐藏字段 {leak_hidden} 行")
    report.check("hidden", "结局原文不出现在 prompt", leak_reference == 0,
                 f"{leak_reference} 行")
    report.check("hidden", "prompt 未复用其它轮原文（医生≥8字/患者≥12字）",
                 leak_doctor == 0 and leak_patient == 0,
                 f"医生轮 {leak_doctor} 行 / 患者轮 {leak_patient} 行")
    report.check("conclusion", "reference_conclusion 完整可用", conclusion_bad == 0,
                 f"{conclusion_bad} 行；无结论轮（F14 保留）{no_conclusion_rows:,} 行")
    report.check("budget", "turn_budget 与 profile 一致", budget_bad == 0, f"{budget_bad} 行")
    report.check("budget", "医生轮数不超上限", budget_bad == 0,
                 f"上限 {max_turns}")
    report.check("schema", "profile 计数自洽", profile_bad == 0, f"{profile_bad} 行")
    report.check("quality", "quality 标记自洽", quality_bad == 0, f"{quality_bad} 行")

    # ------------------------------------------------------------------- sft
    sft_report = load_json(args.sft / "_sft_report.json") or {}
    sft_cases: dict = {}
    sft_total = 0
    sft_bad = 0
    blocked_in_sft = 0
    split_mismatch = 0
    for split in SPLITS:
        path = args.sft / f"{split}.jsonl"
        if not path.exists():
            report.check("counts", f"{split}.jsonl 存在", False)
            continue
        cases = set()
        rows = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows += 1
                sft_total += 1
                case_id = record.get("case_id")
                cases.add(case_id)
                marker = record.get("quality") or {}
                if record.get("split") != split or marker.get("passed") is not True \
                        or "system" in {m["role"] for m in record["messages"]} \
                        or {m["role"] for m in record["messages"]} - {"user", "assistant"}:
                    sft_bad += 1
                if case_id in blocked_in_parquet:
                    blocked_in_sft += 1
                for message in record["messages"]:
                    if message["role"] != "assistant":
                        continue
                    try:
                        payload = json.loads(message["content"])
                    except (TypeError, json.JSONDecodeError):
                        sft_bad += 1
                        continue
                    if not isinstance(payload, dict) or payload.get("action") not in ACTIONS \
                            or not str(payload.get("message") or "").strip():
                        sft_bad += 1
        sft_cases[split] = cases
        expected = (sft_report.get("records") or {}).get(split)
        report.check("counts", f"{split}.jsonl 行数与报告一致",
                     expected is None or expected == rows, f"{rows:,} / 报告 {expected}")
        print(f"--- {split}.jsonl: {rows:,} 条", flush=True)

    for left, right in pairs:
        shared = sft_cases.get(left, set()) & sft_cases.get(right, set())
        report.check("leakage", f"SFT case_id 跨 split 无交集 {left}-{right}",
                     not shared, f"{len(shared)} 个")

    for split, cases in sft_cases.items():
        unknown = cases - split_cases.get(split, set())
        if unknown:
            split_mismatch += len(unknown)
    report.check("leakage", "SFT case_id 与 parquet 同 split 对齐", split_mismatch == 0,
                 f"{split_mismatch} 个不对齐（SFT {sft_total:,} 条 / parquet {total:,} 行）")
    report.check("sft", "SFT 记录结构合法（无 system / assistant 均为动作 JSON）",
                 sft_bad == 0, f"{sft_bad} 条")
    report.check("quality", "quality.passed=false 不进 SFT", blocked_in_sft == 0,
                 f"{blocked_in_sft} 条")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 60)
    print(f"校验结果：{report.summary()}")
    if report.failed:
        print("未通过项：")
        for group, name, _ok, detail in report.failed:
            print(f"  - [{group}] {name} {detail}")
        return 1
    print("全部通过。")
    return 0


def check_sample(report: Report, args, rl_sets: dict) -> None:
    """The cold-start sample: sizes, fields, split consistency and action mix."""
    sample_dir = args.sample
    if not sample_dir.exists():
        report.check("sample", "分层抽样目录存在", False, str(sample_dir))
        return
    sample_report = load_json(sample_dir / "_sample_report.json") or {}
    quotas = sample_report.get("quotas") or {}
    action_counts = collections.Counter()
    doctor_turns_all: list = []
    case_splits: dict = {}
    missing_fields = collections.Counter()
    total = 0
    for split in SPLITS:
        path = sample_dir / f"{split}.jsonl"
        if not path.exists():
            report.check("sample", f"{split}.jsonl 存在", False)
            continue
        rows = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows += 1
                total += 1
                for field in ("case_id", "source", "split", "n_doctor_turns",
                              "quality", "messages", "sample_id"):
                    if field not in record:
                        missing_fields[field] += 1
                if record.get("split") != split:
                    missing_fields["split_mismatch"] += 1
                case_id = record.get("case_id")
                if case_id in case_splits and case_splits[case_id] != split:
                    missing_fields["case_in_two_splits"] += 1
                case_splits[case_id] = split
                if case_id not in rl_sets.get(split, set()):
                    missing_fields["outside_rl"] += 1
                doctor_turns = int(record.get("n_doctor_turns") or 0)
                doctor_turns_all.append(doctor_turns)
                if doctor_turns > args.max_doctor_turns:
                    missing_fields["doctor_turns_over_limit"] += 1
                for message in record["messages"]:
                    if message["role"] != "assistant":
                        continue
                    try:
                        action = json.loads(message["content"]).get("action")
                    except (TypeError, json.JSONDecodeError):
                        missing_fields["bad_action_json"] += 1
                        continue
                    action_counts[action] += 1
        expected = quotas.get(split)
        report.check("sample", f"{split} 抽样条数 = 配额",
                     expected is None or expected == rows,
                     f"{rows:,} / 配额 {expected}")

    supervised = sum(action_counts.values())
    mix = {key: action_counts.get(key, 0) / max(supervised, 1)
           for key in ("WEN", "DIG", "TALK")}
    buckets = {label: sum(1 for value in doctor_turns_all if low <= value <= high)
               for label, low, high in (("1-3", 1, 3), ("4-7", 4, 7),
                                        ("8-15", 8, 15), ("16-20", 16, 20))}
    report.check("sample", "抽样字段齐全",
                 not missing_fields, f"{dict(missing_fields)}")
    report.check("sample", "抽样 case 集合是 RL 子集且 split 一致",
                 not (missing_fields["outside_rl"] + missing_fields["case_in_two_splits"]
                      + missing_fields["split_mismatch"]),
                 f"总 {total:,} 条")
    report.check("sample", "动作配比达标 WEN>=33% DIG>=25% TALK<=42%",
                 mix["WEN"] >= 0.33 and mix["DIG"] >= 0.25 and mix["TALK"] <= 0.42,
                 f"WEN {mix['WEN']:.1%} / DIG {mix['DIG']:.1%} / TALK {mix['TALK']:.1%}")
    report.check("sample", "医生轮数桶齐全且不超上限",
                 all(count > 0 for count in buckets.values())
                 and not missing_fields["doctor_turns_over_limit"],
                 f"{buckets}")
    print(f"      抽样：{total:,} 条，监督轮 {supervised:,}，"
          f"平均医生轮 {sum(doctor_turns_all) / max(len(doctor_turns_all), 1):.2f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
