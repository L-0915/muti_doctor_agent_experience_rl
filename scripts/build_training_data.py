"""Build the RL decision points, with the SFT target carried on the same rows.

Every point at which a doctor speaks is one decision point: the policy sees the
history up to that point and decides whether to ask again or to conclude. One
row serves both training stages.

  - RL reads `observation` and `environment`; it samples its own actions and is
    scored by rollout against `environment.ground_truth`.
  - SFT reads `observation.messages` and `sft_target`, which is the action the
    real doctor took at that state.

Keeping both on one row keeps the state distribution identical across the two
stages, which matters because the cold-start checkpoint shapes RL sampling.

Visibility rule: `observation` is all the policy may read. Everything under
`environment` is for the patient simulator and the evaluator, and must never
enter the prompt.

Anonymity: no field in the output identifies which corpus a sample came from.
Identifiers are opaque digests, and the corpus-specific metadata is dropped
rather than carried along. The digest-to-origin mapping is written to
data/local/ as a local audit record and is not part of the dataset.

Field coverage: only some corpora annotate the patient's clinical findings, and
none annotate past history or risk factors, so fields that cannot be filled are
left empty rather than guessed. `full_dialogue` is always present; it is what
the patient simulator role-plays from, following Doctor-R1.

Output: data/sft_rl/{train,dev,test}.parquet, data/sft_rl/_report.json,
        data/local/provenance_map.parquet

Usage:
    python scripts/build_training_data.py                # every decision point
    python scripts/build_training_data.py --sample 90000 # random draw
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed"
DEFAULT_OUTPUT = ROOT / "data/sft_rl"
PROVENANCE_PATH = ROOT / "data/local/provenance_map.parquet"

SPLIT_SALT = "heapo-v2"
ID_SALT = "heapo-v2-id"

# The doctor must have heard something before a decision point is meaningful.
MIN_PRIOR_PATIENT_TURNS = 1

# Annotations that mark a doctor turn as closing rather than gathering.
CLOSING_LABELS = {
    "prescribe",
    "Recommend",
    "Diagnosis",
    "Inform-Drug_Recommendation",
    "Inform-Medical_Advice",
    "Inform-Precautions",
}

# Turns keep the domain's own vocabulary: a consultation has a patient and a
# doctor. The translation to the chat template's user/assistant happens at the
# boundary, in export_sft.py, and nowhere else.


def opaque(prefix: str, text: str, length: int = 14) -> str:
    """A stable, opaque identifier. Same input always gives the same output."""
    digest = hashlib.sha256(f"{ID_SALT}:{text}".encode()).hexdigest()[:length]
    return f"{prefix}_{digest}"


def split_of(case_id: str) -> str:
    """Hold a whole consultation, and every decision point from it, in one split."""
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{case_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def as_list(value) -> list:
    if value in (None, "", [], {}):
        return []
    return value if isinstance(value, list) else [value]


def action_of(turn: dict, is_last_doctor_turn: bool) -> str:
    """ASK to keep gathering, FINAL to close the consultation.

    The turn the consultation ends on is closing by definition. Elsewhere the
    source annotation decides, because a doctor may prescribe and then keep
    talking, and that turn is still a conclusion.
    """
    if is_last_doctor_turn:
        return "FINAL"
    labels: set[str] = set()
    for key in ("type", "dialogue_act"):
        labels.update(str(v) for v in as_list(turn.get(key)))
    for item in as_list(turn.get("actions")):
        if isinstance(item, dict) and item.get("intent"):
            labels.add(str(item["intent"]))
    return "FINAL" if labels & CLOSING_LABELS else "ASK"


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


def turn_findings(row: dict) -> dict[int, list[dict]]:
    """Findings recorded at each turn, for corpora that annotate per turn."""
    per_turn: dict[int, list[dict]] = {}
    if row["source"] != "CHIP-MDCFNPC":
        return per_turn
    for index, turn in enumerate(row["turns"]):
        found = []
        for entity in turn.get("ner") or []:
            name = entity.get("name") or entity.get("mention")
            if name and name != "undefined":
                found.append({"name": str(name), "value": str(entity.get("attr") or "")})
        if found:
            per_turn[index] = dedupe(found)
    return per_turn


def build_samples(row: dict) -> list[dict]:
    turns = row["turns"]
    origin = f"{row['source']}:{row['id']}"
    case_id = opaque("c", origin)
    labels = row.get("labels") or {}
    last_doctor_index = max(i for i, t in enumerate(turns) if t["role"] == "doctor")

    full_dialogue = [{"role": t["role"], "content": t["text"]} for t in turns]
    findings = case_findings(row)
    per_turn = turn_findings(row)
    # The evaluator's reference. The patient state deliberately excludes the
    # diagnosis: a simulator holding it could reveal the answer instead of
    # answering from reported findings, which is the leak ATPO guards against.
    ground_truth = {
        "diagnosis": labels.get("diagnosis") or labels.get("topic"),
        "department": labels.get("disease_grad"),
    }

    samples: list[dict] = []
    for index, turn in enumerate(turns):
        if turn["role"] != "doctor":
            continue
        history = turns[:index]
        prior_patients = sum(1 for t in history if t["role"] == "patient")
        if prior_patients < MIN_PRIOR_PATIENT_TURNS:
            continue

        if per_turn:
            disclosed = {f["name"] for i in range(index) for f in per_turn.get(i, [])}
            hidden = [f for f in findings if f["name"] not in disclosed]
        else:
            hidden = []

        samples.append({
            "sample_id": opaque("s", f"{origin}#{index}"),
            "case_id": case_id,
            "start_type": "initial" if prior_patients == 1 else "continuation",
            "observation": {
                "messages": [{"role": t["role"], "content": t["text"]} for t in history],
                "turn_id": index,
                "conversation_id": case_id,
            },
            "environment": {
                "patient_state": {
                    "full_dialogue": full_dialogue,
                    "symptoms": findings,
                    "history": [],
                    "risk_factors": [],
                    "hidden_facts": hidden,
                },
                "ground_truth": ground_truth,
            },
            "sft_target": {
                "action": action_of(turn, index == last_doctor_index),
                "message": turn["text"],
            },
            "metadata": {
                "dialogue_turns": len(turns),
                "prior_patient_turns": prior_patients,
                "remaining_doctor_turns": sum(
                    1 for t in turns[index + 1:] if t["role"] == "doctor"),
            },
        })
    return samples


def arrow_schema():
    import pyarrow as pa

    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    finding = pa.struct([("name", pa.string()), ("value", pa.string())])
    return pa.schema([
        ("sample_id", pa.string()),
        ("case_id", pa.string()),
        ("start_type", pa.string()),
        ("observation", pa.struct([
            ("messages", pa.list_(message)),
            ("turn_id", pa.int32()),
            ("conversation_id", pa.string()),
        ])),
        ("environment", pa.struct([
            ("patient_state", pa.struct([
                ("full_dialogue", pa.list_(message)),
                ("symptoms", pa.list_(finding)),
                ("history", pa.list_(finding)),
                ("risk_factors", pa.list_(finding)),
                ("hidden_facts", pa.list_(finding)),
            ])),
            ("ground_truth", pa.struct([
                ("diagnosis", pa.string()),
                ("department", pa.string()),
            ])),
        ])),
        ("sft_target", pa.struct([("action", pa.string()), ("message", pa.string())])),
        ("metadata", pa.struct([
            ("dialogue_turns", pa.int32()),
            ("prior_patient_turns", pa.int32()),
            ("remaining_doctor_turns", pa.int32()),
        ])),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="total decision points to keep, drawn at random from each split in "
             "proportion to its size; 0 keeps everything")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    by_action: collections.Counter = collections.Counter()
    by_start: collections.Counter = collections.Counter()
    provenance: list[dict] = []
    with_symptoms = 0
    with_reference = 0
    dialogues = 0
    total = 0

    for path in sorted(args.input.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                dialogues += 1
                samples = build_samples(row)
                if not samples:
                    continue
                split = split_of(f"{row['source']}:{row['id']}")
                for sample in samples:
                    splits[split].append(sample)
                    by_action[sample["sft_target"]["action"]] += 1
                    by_start[sample["start_type"]] += 1
                    if sample["environment"]["patient_state"]["symptoms"]:
                        with_symptoms += 1
                    truth = sample["environment"]["ground_truth"]
                    if truth["diagnosis"] or truth["department"]:
                        with_reference += 1
                    total += 1
                provenance.append({
                    "case_id": samples[0]["case_id"],
                    "split": split,
                    "corpus": row["source"],
                    "original_id": str(row["id"]),
                    "samples": len(samples),
                })

    print(f"{'划分':<8}{'决策点':>12}{'占比':>9}")
    print("-" * 30)
    for name, rows in splits.items():
        print(f"{name:<8}{len(rows):>12,}{len(rows) / total:>9.1%}")
    print("-" * 30)
    print(f"{'合计':<8}{total:>12,}")
    print()

    # Draw the sample per split so the 80/10/10 proportions survive. Sampling
    # is over decision points rather than dialogues, which is what the policy
    # actually sees: the states it must act in.
    sampled_from = total
    if args.sample > 0 and args.sample < total:
        randomizer = random.Random(args.seed)
        remaining = args.sample
        names = list(splits)
        for index, name in enumerate(names):
            rows = splits[name]
            if index == len(names) - 1:
                quota = remaining
            else:
                quota = round(args.sample * len(rows) / total)
                remaining -= quota
            quota = max(0, min(quota, len(rows)))
            splits[name] = sorted(randomizer.sample(rows, quota),
                                  key=lambda r: r["sample_id"])
        total = sum(len(v) for v in splits.values())
        print(f"随机取样 {total:,} / {sampled_from:,} 个决策点  (seed {args.seed})")

    schema = arrow_schema()
    for name, rows in splits.items():
        table = pa.Table.from_pylist(rows, schema=schema)
        out = args.output / f"{name}.parquet"
        pq.write_table(table, out, compression="zstd")
        print(f"写出 {name:<6}{len(rows):>10,} 行  {out.stat().st_size / 1e6:>8.1f} MB")

    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(provenance), PROVENANCE_PATH)

    print()
    print(f"来源对话 {dialogues:,}，平均每个对话 {total / dialogues:.1f} 个决策点")
    print(f"动作分布   {dict(by_action)}")
    print(f"起点类型   {dict(by_start)}")
    print()
    print(f"有症状标注   {with_symptoms:,} / {total:,}  ({with_symptoms / total:.1%})")
    print(f"有参考诊断   {with_reference:,} / {total:,}  ({with_reference / total:.1%})")
    print(f"来源映射（不随数据集发布）: {PROVENANCE_PATH}  {len(provenance):,} 条")

    report = {
        "salt": SPLIT_SALT,
        "min_prior_patient_turns": MIN_PRIOR_PATIENT_TURNS,
        "sample_requested": args.sample,
        "sample_seed": args.seed if args.sample > 0 else None,
        "available_total": sampled_from,
        "dialogues": dialogues,
        "samples": {k: len(v) for k, v in splits.items()},
        "samples_total": total,
        "by_action": dict(by_action),
        "by_start_type": dict(by_start),
        "with_symptoms": with_symptoms,
        "with_reference": with_reference,
        "note": "输出中不含任何来源标识；来源映射见 data/local/provenance_map.parquet",
    }
    (args.output / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告: {args.output / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
