"""Apply the v3 cleaning rules to the corpus and record every change.

Two passes over the input: the first indexes turn-sequence fingerprints (D07
needs the whole corpus to judge one case), the second writes the cleaned corpus
one source file at a time, so peak memory stays at one file's rows.

Nothing is dropped or rewritten silently. Every case that leaves the corpus and
every turn that changes produces one ledger row naming the rule, the turn, and
the SHA-1 of the text before and after (and, for a dropped case, of the raw
input line, plus its file and line number).

Output:
  <output>/<Source>.jsonl      cleaned corpus, one file per source
  <output>/_report.json        counts, before/after table, rule tallies
  <output>/_ledger.jsonl       one row per drop / repair / flag
  <output>/_manual_review.jsonl  R11 queue: turns a human must adjudicate

Usage:
  python scripts/clean_corpus_v3.py
  python scripts/clean_corpus_v3.py --input data/processed_2 --output data/processed_v3
  python scripts/clean_corpus_v3.py --limit 5000        # smoke test
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402  episode semantics, split
import corpus_v3_rules as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed_2"
DEFAULT_OUTPUT = ROOT / "data/processed_v3"
CENSUS_PATH = ROOT / "data/audit_v3/census.json"

DOCTOR_BUCKETS = [
    ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8),
    ("9-12", 9, 12), ("13-20", 13, 20), ("21+", 21, 10 ** 6),
]


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def histogram(values, buckets) -> dict:
    out: collections.Counter = collections.Counter()
    for value in values:
        for label, low, high in buckets:
            if low <= value <= high:
                out[label] += 1
                break
    return {label: out.get(label, 0) for label, _, _ in buckets}


def median(values) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def profile(turns) -> dict:
    roles = [str(t.get("role") or "") for t in turns]
    return {"turns": len(turns), "doctor_turns": roles.count("doctor"),
            "patient_turns": roles.count("patient")}


class Collector:
    """Counters for one side of the comparison (before / after)."""

    def __init__(self) -> None:
        self.cases = 0
        self.turns = 0
        self.doctor_turns = 0
        self.patient_turns = 0
        self.turn_counts: list = []
        self.doctor_counts: list = []
        self.by_source: dict = collections.defaultdict(
            lambda: {"cases": 0, "turns": 0, "doctor_turns": 0})

    def add(self, source: str, turns) -> None:
        shape = profile(turns)
        self.cases += 1
        self.turns += shape["turns"]
        self.doctor_turns += shape["doctor_turns"]
        self.patient_turns += shape["patient_turns"]
        self.turn_counts.append(shape["turns"])
        self.doctor_counts.append(shape["doctor_turns"])
        entry = self.by_source[source]
        entry["cases"] += 1
        entry["turns"] += shape["turns"]
        entry["doctor_turns"] += shape["doctor_turns"]

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "turns": self.turns,
            "doctor_turns": self.doctor_turns,
            "patient_turns": self.patient_turns,
            "mean_turns": round(self.turns / max(self.cases, 1), 3),
            "median_turns": median(self.turn_counts),
            "mean_doctor_turns": round(self.doctor_turns / max(self.cases, 1), 3),
            "median_doctor_turns": median(self.doctor_counts),
            "doctor_turns_histogram": histogram(self.doctor_counts, DOCTOR_BUCKETS),
            "turns_histogram": histogram(self.turn_counts, [
                ("1-2", 0, 2), ("3-4", 3, 4), ("5-6", 5, 6), ("7-10", 7, 10),
                ("11-15", 11, 15), ("16-20", 16, 20), ("21-30", 21, 30),
                ("31+", 31, 10 ** 6)]),
            "by_source": {src: dict(entry) | {
                "share": round(entry["cases"] / max(self.cases, 1), 4),
                "mean_turns": round(entry["turns"] / max(entry["cases"], 1), 3),
            } for src, entry in sorted(self.by_source.items())},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N cases (smoke tests only)")
    parser.add_argument("--seed", type=int, default=20261016,
                        help="recorded only; the rule engine has no randomness")
    parser.add_argument("--progress-every", type=int, default=20000)
    args = parser.parse_args()

    started = time.time()
    files = R.input_files(args.input)
    if not files:
        raise SystemExit(f"no corpus jsonl under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: fingerprints and first-question groups
    #
    # Two cases that open with the same question must land in the same split,
    # or the evaluation set is the training set with different punctuation. A
    # per-case hash cannot see that, so the corpus is grouped first: cases
    # sharing an opening (exact or as a 40-character window) are unioned, and
    # every member of a group takes the group's split.
    print("[1/2] 指纹与首问分组…", flush=True)
    first_seen: dict = {}
    ask_index: dict = {}
    opening_of: dict = {}
    groups_uf = R.UnionFind()
    indexed = 0
    for path, line_no, row in R.iter_rows(args.input, limit=args.limit):
        indexed += 1
        key = R.case_key(row)
        groups_uf.add(key)
        opening = R.first_ask_text(R.turns_of(row))
        if opening:
            opening_of.setdefault(opening, key)
        for token in R.first_ask_keys(R.turns_of(row)):
            other = ask_index.setdefault(token, key)
            if other != key:
                groups_uf.union(other, key)
        digest = R.fingerprint(R.turns_of(row))
        first_seen.setdefault(digest, key)
        other = ask_index.get("seq::" + digest)
        if other is None:
            ask_index["seq::" + digest] = key
        elif other != key:
            groups_uf.union(other, key)
    print(f"      {indexed:,} 条，唯一指纹 {len(first_seen):,}", flush=True)

    prefix_merges = 0
    for short, longer in R.prefix_merges(opening_of):
        groups_uf.union(opening_of[short], opening_of[longer])
        prefix_merges += 1
    print(f"      首问前缀合并 {prefix_merges:,} 对", flush=True)

    groups = groups_uf.components()
    key_split: dict = {}
    key_group: dict = {}
    for members in groups.values():
        canonical = min(members)
        split = B.split_of(canonical)
        group_id = B.opaque("g", canonical)
        for member in members:
            key_split[member] = split
            key_group[member] = group_id
    sizes = sorted((len(members) for members in groups.values()), reverse=True)
    print(f"      首问分组 {len(groups):,} 组，最大 {sizes[0]:,} 条，"
          f"平均 {sum(sizes) / max(len(sizes), 1):.2f} 条", flush=True)

    # ---- pass 2: clean, write, record
    print("[2/2] 清洗与记账…", flush=True)
    before = Collector()
    after = Collector()
    drop_cases = collections.Counter()
    drop_by_source: dict = collections.defaultdict(collections.Counter)
    primary = collections.Counter()
    repair_turns = collections.Counter()
    repair_by_source: dict = collections.defaultdict(collections.Counter)
    repair_case_hits = collections.Counter()
    flag_cases = collections.Counter()
    flag_by_source: dict = collections.defaultdict(collections.Counter)
    ledger_counts = collections.Counter()
    repair_touched_cases = 0
    manual_rows = 0
    quality_passed = 0
    quality_blocked = 0
    written = 0
    split_counts: dict = collections.Counter()
    spliced = 0

    ledger_path = args.output / "_ledger.jsonl"
    manual_path = args.output / "_manual_review.jsonl"
    with ledger_path.open("w", encoding="utf-8", newline="\n") as ledger, \
            manual_path.open("w", encoding="utf-8", newline="\n") as manual_stream:
        for path in files:
            out_path = args.output / path.name
            with path.open(encoding="utf-8") as stream, \
                    out_path.open("w", encoding="utf-8", newline="\n") as out:
                for line_no, line in enumerate(stream, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if args.limit and before.cases >= args.limit:
                        break
                    row = json.loads(line)
                    source = str(row.get("source") or path.stem)
                    split = key_split.get(R.case_key(row)) or R.split_of(row)
                    case_id = R.case_id_of(row)
                    turns = R.turns_of(row)
                    before.add(source, turns)

                    result = R.analyse(row, dup_of=first_seen)
                    drops = result["drops"] or result["post_drops"]
                    base = {"case_id": case_id, "source": source, "split": split,
                            "file": path.name, "line_no": line_no,
                            "id": str(row.get("id"))}

                    if drops:
                        code = next(code for code in R.DROP_RULES if code in drops)
                        detail = drops[code]
                        if result["post_drops"]:
                            detail = "post_repair|" + detail
                        others = sorted(set(drops) - {code})
                        if others:
                            detail += "|also=" + ",".join(others)
                        ledger.write(json.dumps({
                            **base,
                            "action": "drop",
                            "rule_code": code,
                            "turn_index": -1,
                            "before_sha1": hashlib.sha1(
                                line.encode("utf-8")).hexdigest(),
                            "after_sha1": "",
                            "detail": detail,
                        }, ensure_ascii=False) + "\n")
                        ledger_counts["drop"] += 1
                        drop_cases[code] += 1
                        drop_by_source[code][source] += 1
                        primary[code] += 1
                        continue

                    clean = dict(row)
                    clean["turns"] = result["turns"]
                    clean["case_id"] = case_id
                    clean["quality"] = result["quality"]
                    # Pinned so both training views read the same split.
                    clean["v3_split"] = split
                    clean["group_id"] = key_group.get(R.case_key(row))
                    out.write(json.dumps(clean, ensure_ascii=False,
                                         separators=(",", ":")) + "\n")
                    written += 1
                    split_counts[split] += 1
                    spliced += 1 if result["spliced"] else 0
                    after.add(source, result["turns"])
                    if result["quality"]["passed"]:
                        quality_passed += 1
                    else:
                        quality_blocked += 1

                    touched = set()
                    for op in result["ops"]:
                        code = op["rule_code"]
                        ledger.write(json.dumps({
                            **base,
                            "action": op["action"],
                            "rule_code": code,
                            "turn_index": op["turn_index"],
                            "before_sha1": op["before_sha1"],
                            "after_sha1": op["after_sha1"],
                            "detail": op["detail"],
                        }, ensure_ascii=False) + "\n")
                        ledger_counts[op["action"]] += 1
                        if op["action"] == "repair":
                            repair_turns[code] += 1
                            repair_by_source[code][source] += 1
                        touched.add(code)
                    if touched:
                        repair_touched_cases += 1
                    for code in touched:
                        repair_case_hits[code] += 1
                    for item in result["manual"]:
                        manual_stream.write(json.dumps({
                            **base, "action": "flag", "rule_code": "R11",
                            "turn_index": item["turn_index"],
                            "before_sha1": R.sha1(item["text"]),
                            "after_sha1": R.sha1(item["text"]),
                            "detail": "doctor_voice_in_patient_turn",
                            "text": item["text"]}, ensure_ascii=False) + "\n")
                        manual_rows += 1
                    for code in result["flags"]:
                        flag_cases[code] += 1
                        flag_by_source[code][source] += 1

                    if args.progress_every and before.cases % args.progress_every == 0:
                        print(f"      {before.cases:,} 条已处理", flush=True)

    elapsed = time.time() - started
    source_names = sorted(before.by_source)

    census_alignment = None
    if CENSUS_PATH.exists():
        census = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
        census_alignment = {
            "census_path": rel(CENSUS_PATH),
            "census_cases": census["totals"]["cases"],
            "census_cases_dropped": census["drop_decision"]["cases_dropped"],
            "census_cases_kept": census["projection"]["cases_kept"],
            "cleaner_cases_dropped": sum(drop_cases.values()),
            "cleaner_cases_kept": written,
            "matches": census["projection"]["cases_kept"] == written
                       and census["drop_decision"]["cases_dropped"] == sum(drop_cases.values())
                       and census["totals"]["cases"] == before.cases,
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": "scripts/clean_corpus_v3.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "rules_engine": {"path": "scripts/corpus_v3_rules.py",
                         "sha256": hashlib.sha256(
                             (Path(__file__).resolve().parent / "corpus_v3_rules.py")
                             .read_bytes()).hexdigest()},
        "input": rel(args.input),
        "output": rel(args.output),
        "limit": args.limit,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "rule_titles": R.RULE_TITLES,
        "blocking_flags": sorted(R.BLOCKING_FLAGS),
        "before": before.as_dict(),
        "after": after.as_dict(),
        "dropped": {
            "cases": sum(drop_cases.values()),
            "cases_share": round(sum(drop_cases.values()) / max(before.cases, 1), 4),
            "turns": before.turns - after.turns,
            "by_rule": dict(drop_cases),
            "by_rule_source": {code: dict(drop_by_source[code])
                               for code in sorted(drop_cases)},
        },
        "repaired": {
            "turns": sum(repair_turns.values()),
            "cases": repair_touched_cases,
            "by_rule_turns": dict(repair_turns),
            "by_rule_cases": dict(repair_case_hits),
            "by_rule_source": {code: dict(repair_by_source[code])
                               for code in sorted(repair_turns)},
        },
        "flagged": {
            "by_rule_cases": dict(flag_cases),
            "by_rule_source": {code: dict(flag_by_source[code])
                               for code in sorted(flag_cases)},
        },
        "quality": {"passed": quality_passed, "blocked": quality_blocked},
        "spliced_opening_repaired": spliced,
        "manual_review": {"rows": manual_rows, "path": rel(manual_path)},
        "ledger": {"rows": sum(ledger_counts.values()), "by_action": dict(ledger_counts),
                   "path": rel(ledger_path)},
        "groups": {
            "count": len(groups),
            "largest": sizes[0] if sizes else 0,
            "mean_size": round(sum(sizes) / max(len(sizes), 1), 3),
            "multi_member": sum(1 for size in sizes if size > 1),
            "cases": sum(sizes),
            "prefix_merges": prefix_merges,
            "definition": "首问规范化后整句（前 120 字）、起始 0/40/80 处的 40 字窗口、"
                          "轮次序列指纹：任一相同的对话归为一组，组内取最小组键做哈希切分",
        },
        "split_sizes": dict(split_counts),
        "written": {"cases": written, "path": rel(args.output),
                    "files": {p.name: p.stat().st_size for p in sorted(args.output.glob("*.jsonl"))
                              if not p.name.startswith("_")}},
        "census_alignment": census_alignment,
        "sources": {src: {
            "before_cases": before.by_source[src]["cases"],
            "after_cases": after.by_source.get(src, {}).get("cases", 0),
            "dropped": before.by_source[src]["cases"] - after.by_source.get(src, {}).get("cases", 0),
        } for src in source_names},
    }
    (args.output / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n清洗完成：{before.cases:,} → {written:,} 条"
          f"（删除 {sum(drop_cases.values()):,}，修复轮 {sum(repair_turns.values()):,}，"
          f"质量门通过 {quality_passed:,} / 阻断 {quality_blocked:,}）")
    print(f"轮数 {before.turns:,} → {after.turns:,}；医生轮 {before.doctor_turns:,} → "
          f"{after.doctor_turns:,}")
    print(f"报告 {args.output / '_report.json'}")
    print(f"台账 {ledger_path}")
    if census_alignment:
        print(f"普查对齐：{census_alignment}")
    print(f"耗时 {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
