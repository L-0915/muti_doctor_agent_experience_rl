"""Build the RL training set: one row per consultation, one episode per row.

The rollout is generated online — the policy consults, the patient simulator
answers, until the policy concludes or the turn budget ends — so a row stores
only what that loop needs: the initial state, the simulator's knowledge base,
and the reward reference. The columns follow the interface veRL-style trainers
already read (ATPO's released RL set is organized the same way), so PPO, GRPO
and tree-search variants consume the file without adaptation:

  prompt        [user(chief complaint)] — the system block is injected at
                training time from scripts/sys_prompt.py, so wording changes
                cost nothing; protocol is configuration, not data
  ground_truth  reference_conclusion + honest label fields          reward input
  extra_info    patient_state (simulator knowledge) + profile       interaction env
                + quality (v3 markers) + turn_budget (rollout cap)
  data_source   fixed tag; no corpus name ever appears              routing
  index         row number within the split

Innovation lives in the content, not the shape: the environment is a real
consultation rather than a synthesized exam item, and the primary reward
reference is the conclusion the real physician actually reached. One
`patient_state.full_dialogue` serves three uses — simulator knowledge base,
reward reference source, and the future experience-retrieval corpus.

The same field also warns about what it is not: `presenting_problem` is the
platform topic tag (what the patient asked about), which once masqueraded as
the diagnosis and scored a consultation against "尖锐湿疣" while the doctor's
actual conclusion was "不考虑尖锐湿疣". `diagnosis` is a true outcome label and
exists only where a corpus supplies one (IMCS-21).

Output: data/sft_rl/{train,dev,test}.parquet, data/sft_rl/_report.json,
        data/local/provenance_map.parquet (local audit, never published)

Usage:
    python scripts/build_training_data.py                  # every episode
    python scripts/build_training_data.py --sample 20000   # random episodes
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clean_cases as V2  # noqa: E402  v2 promotion/contact patterns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed_2"
DEFAULT_OUTPUT = ROOT / "data/sft_rl"
PROVENANCE_PATH = ROOT / "data/local/provenance_map.parquet"

SPLIT_SALT = "heapo-v2"
ID_SALT = "heapo-v2-id"
DATA_SOURCE = "heapo-episode-v1"

# An episode whose reward reference is a two-character acknowledgement teaches
# the policy nothing about concluding; it only teaches the reward to be noise.
MIN_CONCLUSION_CHARS = 8

# A consultation this long is an anxiety loop, not information gathering: the
# clinical conclusion lands early and the remaining turns are reassurance
# ping-pong (one 100-turn episode in the corpus holds 43 reassurance turns
# against 6 questions). It also cannot be rolled out under any sane turn
# budget. The corpus median is 8 doctor turns; 20 keeps 97.5%.
DEFAULT_MAX_DOCTOR_TURNS = 20

# Annotations that mark a doctor turn as closing rather than gathering.
CLOSING_LABELS = {
    "prescribe",
    "Recommend",
    "Diagnosis",
    "Inform-Drug_Recommendation",
    "Inform-Medical_Advice",
    "Inform-Precautions",
}

# A turn is only a question if it really asks one: 77% of the turns the first
# build labelled as questions were advice or acknowledgement.
QUESTION_TURN = re.compile(
    r"[？?]"                                          # 1. 问号
    r"|(?:请问|想问|咨询一下|问一下|麻烦问|请教|能否问)"  # 2. 问句套话
    r"|(?:吗|呢)(?=[，,。.!！~～；;：:\s]|$)"           # 3. 句尾语气词
    r"|(?:是否|能否)"                                  # 4. 正反问
    r"|(?:能不能|可不可以|好不好|行不行|有没有|是不是|对不对|"
    r"要不要|会不会|严不严重|疼不疼|痛不痛|烧不烧|大不大|"
    r"多不多|少不少)"                                  # 5. A-不-A
    r"|(?:什么|怎么|为什么|为何|如何|"
    r"哪[个些里儿边]?|多少|多[长多久大高重]|"
    r"几[天年月岁次个])"                               # 6. 疑问词
)

# The rollout gets the reference consultation's own doctor turns plus two, so a
# policy that asks one more question than the physician did is not cut off, and
# the same ceiling that rejected the longest consultations caps the episode.
TURN_BUDGET_SLACK = 2
TURN_BUDGET_FLOOR = 3

PLEASANTRY = re.compile(
    r"欢迎随时|随时咨询|动动.*手指|对本次咨询满意|满意.*评价|"
    r"祝您|祝早日|祝宝宝|感谢您|谢谢您|不客气|再见|好评"
)
# A conclusion that reads as a greeting, a booking receipt, a promotion or a
# platform notice is not a conclusion; the audit found 931 promotion templates
# and 13 booking forms serving as reward references.
TRAILING_PLEASANTRY = re.compile(
    r"(?:祝您|祝早日|祝宝宝|祝宝宝健康|感谢您的信任|感谢您的咨询|谢谢您的信任|"
    r"不客气|再见|拜拜|有问题随时|随时(?:咨询|联系|追问)|欢迎随时|"
    r"如果(?:还有)?(?:疑问|问题)[^。！？!?]*)[^。！？!?]*[。！？!?]?s*$")

GREETING_ONLY = re.compile(
    r"^(?:你好|您好|在吗|在不在|医生你好|医生您好|您好医生|我是.{0,8}医生|"
    r"祝您|祝早日|感谢您|谢谢您|不客气|再见|拜拜|晚安)[\s，。！？!?~～.…]*$")

REGISTRATION_FORM = re.compile(
    r"预约成功|预约失败|挂号成功|挂号失败|订单号|订单金额|支付成功|支付方式|"
    r"退款|退费|就诊人|就诊时间|就诊科室|取消预约|取号|叫号|签到|导诊|"
    r"请选择|请填写|请输入您的|您的预约|预约码|候诊")

PLATFORM_PLACEHOLDER = re.compile(
    r"语音因?隐私|因隐私问题无法显示|无法显示图片|该内容仅自己可见|消息已撤回|"
    r"撤回了一条消息|对方正在输入|该用户已注销|系统消息|自动回复|"
    r"\[(?:语音|图片|视频|表情|位置)\]|<\s*IMAGE_UNAVAILABLE\s*>")

# A question is a question at its end. Mid-sentence interrogatives ("看有没有
# 贫血", "什么原因导致的") are how physicians write conclusions, so treating them
# as questions would throw away real conclusions.
INTERROGATIVE_END = re.compile(r"[？?]\s*$|(?:吗|呢)\s*[。.!！~～…]?\s*$")

DEPARTMENT_TAIL = re.compile(r"^[\u4e00-\u9fffA-Za-z]{1,12}(?:科|室|中心|门诊|病区|病室)$")


def is_department_name(value) -> bool:
    """True when a label that should name the patient's complaint names a
    department instead. The audit found 31.8% of presenting_problem values
    carrying a department, which trains the model to read "消化内科" as a
    symptom."""
    text = str(value or "").strip()
    if not text:
        return False
    if ">" in text:
        return True
    return bool(DEPARTMENT_TAIL.match(text))


def is_substantive_doctor(text: str, min_chars: int = MIN_CONCLUSION_CHARS) -> bool:
    """A doctor turn that can serve as the consultation's conclusion."""
    raw = str(text or "")
    cleaned = PLATFORM_PLACEHOLDER.sub(" ", raw)
    cleaned = TRAILING_PLEASANTRY.sub("", cleaned).strip()
    body = re.sub(r"\s+", "", cleaned)
    if len(body) < min_chars:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", body):
        return False
    if GREETING_ONLY.match(cleaned):
        return False
    if REGISTRATION_FORM.search(cleaned):
        return False
    if INTERROGATIVE_END.search(cleaned):
        return False
    if V2.PROMOTION_CUE.search(cleaned):
        return False
    if any(pattern.search(cleaned) for pattern, _ in V2.DIRECT_IDENTIFIERS):
        return False
    return True


CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
DIGIT_RUN = re.compile(r"\d+")
CLOSING_TOKENS = re.compile(r"谢|好|明白|知道|辛苦|麻烦|可以|已经|一定|嗯|哦|行|收到|了解")


def _tokens(text: str) -> set:
    tokens = set(DIGIT_RUN.findall(text))
    for run in CJK_RUN.findall(text):
        for i in range(len(run) - 1):
            tokens.add(run[i:i + 2])
    return tokens


def post_conclusion(turns: list[dict], index: int) -> dict:
    """What the consultation kept saying after the conclusion.

    The audit found 21.1% of episodes concluding before the dialogue ended,
    with a median of two turns still to come; the count and whether the patient
    still supplied new information let a training run tell an early conclusion
    from a wrong one.
    """
    later = turns[index + 1:]
    patient_later = [str(t.get("text") or "") for t in later if t.get("role") == "patient"]
    early = " ".join(str(t.get("text") or "") for t in turns[:index + 1])
    fresh = _tokens(" ".join(patient_later)) - _tokens(early)
    informative = {token for token in fresh if not CLOSING_TOKENS.search(token)}
    return {
        "post_conclusion_turns": len(later),
        "post_conclusion_has_new_info": bool(informative),
        "post_conclusion_new_tokens": len(informative),
    }


def opaque(prefix: str, text: str, length: int = 14) -> str:
    digest = hashlib.sha256(f"{ID_SALT}:{text}".encode()).hexdigest()[:length]
    return f"{prefix}_{digest}"


def split_of(case) -> str:
    """Hold a whole consultation inside one split, in both this file and the
    SFT export, which imports this function.

    Accepts a corpus row or a plain case key. A row the v3 cleaner pinned to a
    group-level split keeps that split, so every case that opens with the same
    question stays together; anything else falls back to the per-case hash that
    v2 used, which is also what assigns a group its split.
    """
    if isinstance(case, dict):
        pinned = case.get("v3_split")
        if pinned in ("train", "dev", "test"):
            return pinned
        case_id = f"{case.get('source')}:{case.get('id')}"
    else:
        case_id = str(case)
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{case_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def turn_budget_of(doctor_turns: int, max_doctor_turns: int = DEFAULT_MAX_DOCTOR_TURNS) -> int:
    return max(TURN_BUDGET_FLOOR, min(doctor_turns + TURN_BUDGET_SLACK, max_doctor_turns))


def quality_marker(row: dict, doctor_turns: int,
                   max_doctor_turns: int = DEFAULT_MAX_DOCTOR_TURNS,
                   pp_is_department: bool = False) -> dict:
    """The v3 markers travel with the episode so a training run can stratify on
    them. A corpus without markers reads as unmarked rather than as clean."""
    marker = row.get("quality") or {}
    return {
        "rules": sorted(str(code) for code in (marker.get("rules") or [])),
        "image_dependent": bool(marker.get("image_dependent")),
        "doctor_turns_ok": 2 <= doctor_turns <= max_doctor_turns,
        "pp_is_department": pp_is_department,
        "conclusion_available": bool(marker.get("conclusion_available", True)),
    }


def as_list(value) -> list:
    if value in (None, "", [], {}):
        return []
    return value if isinstance(value, list) else [value]


def conclusion_of(turns: list[dict], text_key: str = "text") -> int | None:
    """The last doctor turn that can serve as this consultation's conclusion.

    Scans backwards and returns the first substantive doctor turn, so the
    reward reference is what the physician finally said rather than whatever
    happened to be last (a "祝您早日康复", a booking receipt, a promotion).
    None means the consultation has no conclusion at all — the cleaner drops
    those as D10.
    """
    for i in range(len(turns) - 1, -1, -1):
        turn = turns[i]
        if turn.get("role") != "doctor":
            continue
        if is_substantive_doctor(str(turn.get(text_key, ""))):
            return i
    return None


def action_of(turn: dict, is_conclusion: bool) -> str:
    """WEN (问, ask), DIG (诊, conclude), or TALK.

    The turn the consultation concludes on is DIG; a turn that really asks is
    WEN; the rest is TALK — reassurance, acknowledgement, trailing
    pleasantries — kept in histories, never a supervised decision.
    """
    if is_conclusion:
        return "DIG"
    if QUESTION_TURN.search(turn["text"]):
        return "WEN"
    labels: set[str] = set()
    for key in ("type", "dialogue_act"):
        labels.update(str(v) for v in as_list(turn.get(key)))
    for item in as_list(turn.get("actions")):
        if isinstance(item, dict) and item.get("intent"):
            labels.add(str(item["intent"]))
    if labels & CLOSING_LABELS:
        return "DIG"
    
    return "TALK"


def dedupe(findings: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in findings:
        key = (item["name"], item["value"])
        if item["name"] and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def case_findings(row: dict) -> list[dict]:
    """Clinical findings the corpus records for this patient, if any."""
    source = row["source"]
    labels = row.get("labels") or {}
    out: list[dict] = []

    if source == "CHIP-MDCFNPC":
        for turn in row["turns"]:
            for entity in turn.get("ner") or []:
                name = entity.get("name") or entity.get("mention")
                if name and name != "undefined":
                    out.append({"name": str(name), "value": str(entity.get("attr") or "")})
    elif source == "IMCS-21":
        for group, items in (labels.get("implicit_info") or {}).items():
            if isinstance(items, dict):
                for name, value in items.items():
                    out.append({"name": str(name), "value": f"{group}:{value}"})
        for group, items in (labels.get("explicit_info") or {}).items():
            for name in as_list(items):
                out.append({"name": str(name), "value": f"{group}:显性"})
    elif source == "MedDG":
        for turn in row["turns"]:
            for name in as_list(turn.get("Symptom")):
                out.append({"name": str(name), "value": "症状"})
            for name in as_list(turn.get("Disease")):
                out.append({"name": str(name), "value": "疾病"})
    return dedupe(out)


def build_episode(row: dict,
                  max_doctor_turns: int = DEFAULT_MAX_DOCTOR_TURNS) -> tuple[dict | None, str | None]:
    """One consultation as one episode row, or the reason it was rejected."""
    turns = row["turns"]
    origin = f"{row['source']}:{row['id']}"
    case_id = opaque("c", origin)
    labels = row.get("labels") or {}
    doctor_indexes = [i for i, t in enumerate(turns) if t["role"] == "doctor"]

    conclusion_index = conclusion_of(turns)
    if conclusion_index is None:
        # No reward reference exists, but the consultation still shows what the
        # patient reports and what was asked, so it stays in the corpus as a
        # consultation state; quality.conclusion_available keeps it out of the
        # conclusion reward downstream.
        reference_conclusion = None
        tail = {"post_conclusion_turns": 0, "post_conclusion_has_new_info": False}
    else:
        reference_conclusion = turns[conclusion_index]["text"].strip()
        tail = post_conclusion(turns, conclusion_index)
    # presenting_problem is the platform topic tag, not a chief complaint: when
    # it names a department the model would learn to read "消化内科" as a
    # symptom, so it is nulled and the shape is recorded instead.
    topic = labels.get("topic")
    department_value = labels.get("disease_grad")
    department_path = (
        str(department_value) if ">" in str(department_value or "") else None)
    pp_is_department = is_department_name(topic)
    if pp_is_department:
        topic = None

    actions = {i: action_of(turns[i], i == conclusion_index) for i in doctor_indexes}
    profile = {
        "dialogue_turns": len(turns),
        "doctor_turns": len(doctor_indexes),
        "wen": sum(1 for a in actions.values() if a == "WEN"),
        "dig": sum(1 for a in actions.values() if a == "DIG"),
        "talk": sum(1 for a in actions.values() if a == "TALK"),
    }

    episode = {
        "prompt": [
            {"role": "user", "content": turns[0]["text"]},
        ],
        "ground_truth": {
            "reference_conclusion": reference_conclusion,
            "diagnosis": labels.get("diagnosis"),
            "presenting_problem": topic,
            "department": department_value,
            "department_path": department_path,
        },
        "extra_info": {
            "sample_id": opaque("e", origin),
            "case_id": case_id,
            "patient_state": {
                "full_dialogue": [
                    {"role": t["role"], "content": t["text"]} for t in turns],
                "symptoms": case_findings(row),
            },
            "profile": profile,
            "quality": quality_marker(
                row, profile["doctor_turns"], max_doctor_turns, pp_is_department),
            "turn_budget": turn_budget_of(profile["doctor_turns"], max_doctor_turns),
            "conclusion_turn_index": conclusion_index,
            "post_conclusion_turns": tail["post_conclusion_turns"],
            "post_conclusion_has_new_info": tail["post_conclusion_has_new_info"],
        },
        "data_source": DATA_SOURCE,
        "index": 0,
    }
    return episode, None


def arrow_schema():
    import pyarrow as pa

    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    finding = pa.struct([("name", pa.string()), ("value", pa.string())])
    return pa.schema([
        ("prompt", pa.list_(message)),
        ("ground_truth", pa.struct([
            ("reference_conclusion", pa.string()),
            ("diagnosis", pa.string()),
            ("presenting_problem", pa.string()),
            ("department", pa.string()),
            ("department_path", pa.string()),
        ])),
        ("extra_info", pa.struct([
            ("sample_id", pa.string()),
            ("case_id", pa.string()),
            ("patient_state", pa.struct([
                ("full_dialogue", pa.list_(message)),
                ("symptoms", pa.list_(finding)),
            ])),
            ("profile", pa.struct([
                ("dialogue_turns", pa.int32()),
                ("doctor_turns", pa.int32()),
                ("wen", pa.int32()),
                ("dig", pa.int32()),
                ("talk", pa.int32()),
            ])),
            ("quality", pa.struct([
                ("rules", pa.list_(pa.string())),
                ("image_dependent", pa.bool_()),
                ("doctor_turns_ok", pa.bool_()),
                ("pp_is_department", pa.bool_()),
                ("conclusion_available", pa.bool_()),
            ])),
            ("turn_budget", pa.int32()),
            ("conclusion_turn_index", pa.int32()),
            ("post_conclusion_turns", pa.int32()),
            ("post_conclusion_has_new_info", pa.bool_()),
        ])),
        ("data_source", pa.string()),
        ("index", pa.int64()),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="total episodes to keep, drawn at random from each split in "
             "proportion to its size; 0 keeps everything")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH,
                        help="local audit map (corpus name ↔ case_id); kept out "
                             "of the published dataset")
    parser.add_argument("--max-doctor-turns", type=int, default=DEFAULT_MAX_DOCTOR_TURNS,
                        help="reject consultations longer than this; the "
                             "corpus median is 8 and 20 keeps 97.5%%")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    rejects: collections.Counter = collections.Counter()
    provenance: list[dict] = []
    dialogues = 0
    with_symptoms = 0

    # A leading underscore marks bookkeeping written next to the corpus
    # (_report.json, _ledger.jsonl), never a source.
    for path in sorted(p for p in args.input.glob("*.jsonl")
                       if not p.name.startswith("_")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                dialogues += 1
                episode, reason = build_episode(row, args.max_doctor_turns)
                if episode is None:
                    rejects[reason] += 1
                    continue
                profile = episode["extra_info"]["profile"]
                if profile["doctor_turns"] > args.max_doctor_turns:
                    rejects["医生轮过多"] += 1
                    continue
                origin = f"{row['source']}:{row['id']}"
                split = split_of(row)
                splits[split].append(episode)
                provenance.append({
                    "case_id": episode["extra_info"]["case_id"],
                    "sample_id": episode["extra_info"]["sample_id"],
                    "split": split,
                    "corpus": row["source"],
                    "original_id": str(row["id"]),
                })
                if episode["extra_info"]["patient_state"]["symptoms"]:
                    with_symptoms += 1

    total = sum(len(v) for v in splits.values())
    print(f"{dialogues:,} 条问诊 → {total:,} 个 episode"
          f"（剔除 {sum(rejects.values()):,}：{dict(rejects)}）")
    print(f"带临床发现标注 {with_symptoms:,} ({with_symptoms / max(total, 1):.1%})\n")

    sampled_from = total
    if args.sample > 0 and args.sample < total:
        randomizer = random.Random(args.seed)
        remaining = args.sample
        names = list(splits)
        for position, name in enumerate(names):
            if position == len(names) - 1:
                quota = remaining
            else:
                quota = round(args.sample * len(splits[name]) / total)
                remaining -= quota
            quota = max(0, min(quota, len(splits[name])))
            splits[name] = sorted(
                randomizer.sample(splits[name], quota),
                key=lambda e: e["extra_info"]["sample_id"])
        total = sum(len(v) for v in splits.values())
        print(f"随机取样 {total:,} / {sampled_from:,} 个 episode  (seed {args.seed})\n")

    schema = arrow_schema()
    for name, episodes in splits.items():
        for position, episode in enumerate(episodes):
            episode["index"] = position
        table = pa.Table.from_pylist(episodes, schema=schema)
        out = args.output / f"{name}.parquet"
        pq.write_table(table, out, compression="zstd")
        doctor_turns = [e["extra_info"]["profile"]["doctor_turns"] for e in episodes]
        doctor_turns.sort()
        median = doctor_turns[len(doctor_turns) // 2] if doctor_turns else 0
        print(f"写出 {name:<6}{len(episodes):>8,} 行  {out.stat().st_size / 1e6:>7.1f} MB"
              f"   医生轮中位 {median}")

    wen = sum(e["extra_info"]["profile"]["wen"] for v in splits.values() for e in v)
    dig = sum(e["extra_info"]["profile"]["dig"] for v in splits.values() for e in v)
    talk = sum(e["extra_info"]["profile"]["talk"] for v in splits.values() for e in v)
    print(f"\n轮次轮廓合计  WEN {wen:,} / DIG {dig:,} / TALK {talk:,}")

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(provenance), args.provenance)
    print(f"来源映射（本地审计，不随数据集发布）: {args.provenance}")

    episodes = [e for bucket in splits.values() for e in bucket]
    report = {
        "data_source": DATA_SOURCE,
        "conclusion_not_last_turn": sum(
            1 for e in episodes if e["extra_info"]["post_conclusion_turns"] > 0),
        "post_conclusion_new_info": sum(
            1 for e in episodes if e["extra_info"]["post_conclusion_has_new_info"]),
        "no_conclusion_episodes": sum(
            1 for e in episodes
            if not e["extra_info"]["quality"]["conclusion_available"]),
        "presenting_problem_nulled_as_department": sum(
            1 for e in episodes if e["extra_info"]["quality"]["pp_is_department"]),
        "department_path_episodes": sum(
            1 for e in episodes if e["ground_truth"]["department_path"]),
        "quality_rules": dict(collections.Counter(
            code for e in episodes for code in e["extra_info"]["quality"]["rules"])),
        "unit": "episode (one consultation per row)",
        "dialogues": dialogues,
        "episodes": {k: len(v) for k, v in splits.items()},
        "episodes_total": total,
        "sample_requested": args.sample,
        "sample_seed": args.seed if args.sample > 0 else None,
        "max_doctor_turns": args.max_doctor_turns,
        "turn_budget": {"slack": TURN_BUDGET_SLACK, "floor": TURN_BUDGET_FLOOR},
        "rejected": dict(rejects),
        "with_symptoms": with_symptoms,
        "turn_totals": {"wen": wen, "dig": dig, "talk": talk},
    }
    (args.output / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告: {args.output / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
