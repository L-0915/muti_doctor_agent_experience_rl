"""Export the cold-start SFT set out of the RL decision points.

Both stages read the same rows, so the cold-start model meets exactly the state
distribution that RL will sample from. The only difference is what is kept: SFT
needs the history and the action the real doctor took, and nothing else.

Doctor-visibility rule: only `observation.messages` enters the record. The full
dialogue and the reference diagnosis stay behind, because they are for the
patient simulator and the evaluator.

The decision points name a patient and a doctor. ms-swift and LLaMA-Factory
chat templates expect `user` and `assistant`, which is what the export writes by
default; `--roles patient-doctor` keeps the consultation's own vocabulary for a
custom template.

Output: data/sft_rl/{train,dev,test}.jsonl, plus _sft_report.json

Usage:
    python scripts/export_sft.py
    python scripts/export_sft.py --roles patient-doctor
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/sft_rl"
DEFAULT_OUTPUT = ROOT / "data/sft_rl"


def select(rows: list[dict], limit: int) -> list[dict]:
    """Spread the picks over the consultation: how it opens, how it closes, and
    a few points in between. Even spacing keeps every stage of the dialogue in
    the cold-start set instead of only its beginning.

    A limit of zero or less means no limit: every decision point is exported.
    Neighbouring decision points share almost all of their history, so the
    default caps each consultation rather than repeating it several times.
    """
    ordered = sorted(rows, key=lambda r: r["observation"]["turn_id"])
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    step = (len(ordered) - 1) / (limit - 1)
    picked = [ordered[round(i * step)] for i in range(limit)]
    # The closing turn is the one that teaches how to end a consultation, so it
    # is always kept even when it falls outside the even spacing.
    closing = ordered[-1]
    if closing not in picked:
        picked[-1] = closing
    return picked


# The decision points name a patient and a doctor. What the record must call
# them depends on the trainer consuming it, so both names are decided together
# and applied to the whole record: a record with `patient` history and an
# `assistant` target matches no template at all.
ROLE_MAP = {
    "user-assistant": {"patient": "user", "doctor": "assistant"},
    "patient-doctor": {"patient": "patient", "doctor": "doctor"},
}


def to_record(row: dict, role_map: dict[str, str]) -> dict:
    """Emit one record. Only `observation` is visible to the model;
    `environment` is deliberately left out of the record."""
    history = row["observation"]["messages"]
    if not history:
        raise ValueError(f"Empty history in {row['sample_id']}")
    roles = {m["role"] for m in history}
    if not roles <= set(role_map):
        raise ValueError(f"Unexpected role in {row['sample_id']}: {sorted(roles)}")
    if history[-1]["role"] != "patient":
        raise ValueError(f"History must end with a patient turn: {row['sample_id']}")
    target = json.dumps(
        {"action": row["sft_target"]["action"], "message": row["sft_target"]["message"]},
        ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            *({"role": role_map[m["role"]], "content": m["content"]} for m in history),
            {"role": role_map["doctor"], "content": target},
        ],
        "sample_id": row["sample_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-dialogue", type=int, default=4)
    parser.add_argument(
        "--roles", choices=["user-assistant", "patient-doctor"],
        default="user-assistant",
        help="how to name the two parties in the record; ms-swift and "
             "LLaMA-Factory templates expect user-assistant, which is the default")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    role_map = ROLE_MAP[args.roles]

    print(f"{'划分':<8}{'决策点':>12}{'SFT 样本':>12}{'占比':>8}")
    print("-" * 42)
    totals = collections.Counter()
    actions = collections.Counter()

    for split in ("train", "dev", "test"):
        table = pq.read_table(args.input / f"{split}.parquet")
        by_case: dict[str, list[dict]] = collections.defaultdict(list)
        for row in table.to_pylist():
            by_case[row["case_id"]].append(row)

        kept: list[dict] = []
        for rows in by_case.values():
            kept.extend(select(rows, args.max_per_dialogue))
        kept.sort(key=lambda r: r["sample_id"])

        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in kept:
                stream.write(json.dumps(to_record(row, role_map), ensure_ascii=False) + "\n")
                actions[row["sft_target"]["action"]] += 1

        totals[split] = len(kept)
        print(f"{split:<8}{table.num_rows:>12,}{len(kept):>12,}{len(kept) / table.num_rows:>8.1%}")

    print("-" * 42)
    print(f"{'合计':<8}{'':>12}{sum(totals.values()):>12,}")
    print()
    print(f"动作分布 {dict(actions)}")
    for split in totals:
        path = args.output / f"{split}.jsonl"
        print(f"写出 {split:<6}{totals[split]:>8,} 行  {path.stat().st_size / 1e6:>7.1f} MB")

    report = {
        "max_per_dialogue": args.max_per_dialogue,
        "roles": args.roles,
        "source": "data/training (RL decision points)",
        "samples": dict(totals),
        "samples_total": sum(totals.values()),
        "by_action": dict(actions),
    }
    # Distinct name: build_training_data.py writes its own report into the same
    # directory, and the two must not overwrite each other.
    (args.output / "_sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
