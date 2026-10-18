"""Draw the cold-start SFT sample: 10k train, 1.25k dev, 1.25k test.

The full v3 export is the mother set; this draws a stratified sample from it.
Random sampling is not enough here: one source is 51% of the corpus, half of a
physician's turns are TALK, and a cold-start model trained on that learns to
keep asking and never to conclude. So the draw is stratified on three axes and
the action mix is a target, not an outcome:

  source        KaMed capped, IMCS-21 (the only source with diagnosis labels)
                floored, so no single corpus silently dominates
  doctor turns  1-3 / 4-7 / 8-15 / 16-20 all represented, > 20 excluded
  action mix    each record is scored by how close its (wen, dig, talk) share is
                to the target, and the aggregate must reach WEN >= 33%,
                DIG >= 25%, TALK <= 42%

The quota weights start from a clinical-realism prior and are relaxed toward
short consultations until the action constraints hold; the loop is
deterministic, so the same command reproduces the same files.

Output:
  <output>/{train,dev,test}.jsonl        dialogue records
  <output>/_sample_report.json
  <turn-output>/{train,dev,test}.jsonl   one record per WEN/DIG decision point

Usage:
  python scripts/sample_sft_stratified.py --seed 42
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402
import export_sft as E  # noqa: E402  rl_case_sets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/sft_rl_v3"
DEFAULT_TURN_INPUT = ROOT / "data/sft_rl_v3/sft_turn_full"
DEFAULT_OUTPUT = ROOT / "data/sft_rl_v3/sft10k"
DEFAULT_TURN_OUTPUT = ROOT / "data/sft_rl_v3/sft10k_turn"
DEFAULT_RL_DIR = ROOT / "data/sft_rl_v3"

SPLITS = ("train", "dev", "test")
DOCTOR_BUCKETS = (("1-3", 1, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16-20", 16, 20))
# What the draw aims at per record; the constraints below are the floors.
IDEAL = {"wen": 0.37, "dig": 0.28, "talk": 0.35}
CONSTRAINTS = {"wen": 0.33, "dig": 0.25, "talk_max": 0.42}
KAIMED_CAP = 0.35
IMCS_FLOOR = 800
MAX_DOCTOR_TURNS = 20

# The user's earlier 10k sample, measured by the independent audit. Kept here so
# the report can compare like with like without re-reading that file.
BASELINE = {
    "note": "用户旧采样 scripts/sample.py 实测（外部审计提供），仅作对照",
    "records": 10000,
    "fields": ["messages", "sample_id"],
    "supervised_assistant_turns": 77061,
    "assistant_turns_per_record": {"p50": 7, "p90": 15, "max": 50, "min": 1},
    "actions": {"TALK": 37476, "WEN": 25248, "DIG": 14337},
    "action_share": {"TALK": 0.486, "WEN": 0.328, "DIG": 0.186},
}


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def tiebreak(record: dict, seed: int) -> str:
    """Stable per-record ordering key: no RNG state, so a rerun on the same
    input reproduces the same draw."""
    return hashlib.sha256(f"{seed}:{record.get('sample_id')}".encode()).hexdigest()


def action_counts(record: dict) -> tuple:
    wen = dig = talk = 0
    for message in record["messages"]:
        if message["role"] != "assistant":
            continue
        try:
            action = json.loads(message["content"]).get("action")
        except (TypeError, json.JSONDecodeError):
            continue
        if action == "WEN":
            wen += 1
        elif action == "DIG":
            dig += 1
        elif action == "TALK":
            talk += 1
    return wen, dig, talk


def bucket_of(doctor_turns: int):
    for label, low, high in DOCTOR_BUCKETS:
        if low <= doctor_turns <= high:
            return label
    return None


def shares(counts: dict) -> dict:
    total = counts["wen"] + counts["dig"] + counts["talk"]
    if not total:
        return {"wen": 0.0, "dig": 0.0, "talk": 0.0}
    return {key: counts[key] / total for key in ("wen", "dig", "talk")}


def distance(record: dict) -> float:
    """How far this record's own action mix is from the target mix."""
    mix = shares(record["_counts"])
    return sum(abs(mix[key] - IDEAL[key]) for key in ("wen", "dig", "talk"))


def load_split(input_dir: Path, split: str) -> list:
    path = input_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare(records: list, seed: int) -> list:
    """Attach the per-record attributes the selection needs."""
    prepared = []
    for record in records:
        doctor_turns = int(record.get("n_doctor_turns") or 0)
        bucket = bucket_of(doctor_turns)
        if bucket is None:
            continue                      # > 20 doctor turns: token cost, excluded
        counts = action_counts(record)
        record["_counts"] = {"wen": counts[0], "dig": counts[1], "talk": counts[2]}
        record["_doctor_turns"] = doctor_turns
        record["_bucket"] = bucket
        record["_source"] = str(record.get("source") or "unknown")
        record["_tiebreak"] = tiebreak(record, seed)
        prepared.append(record)
    return prepared


def draw(records: list, n: int, weights: dict, kam_allowance: int, imcs_floor: int) -> list:
    """Take the quota of each doctor-turn bucket, best action mix first."""
    by_bucket: dict = collections.defaultdict(list)
    for record in records:
        by_bucket[record["_bucket"]].append(record)
    for bucket in by_bucket:
        by_bucket[bucket].sort(key=lambda r: (distance(r), r["_tiebreak"]))

    quotas = {}
    for label, _, _ in DOCTOR_BUCKETS:
        quotas[label] = max(1, round(weights[label] * n))

    picked: list = []
    kam = 0
    for label, _, _ in DOCTOR_BUCKETS:
        taken = 0
        for record in by_bucket.get(label, []):
            if taken >= quotas[label]:
                break
            if record["_source"] == "KaMed":
                if kam >= kam_allowance:
                    continue
                kam += 1
            picked.append(record)
            taken += 1

    # IMCS-21 is the only source with diagnosis labels; buy the floor back by
    # swapping out the worst-fitting records of other sources.
    if imcs_floor:
        have = sum(1 for r in picked if r["_source"] == "IMCS-21")
        if have < imcs_floor:
            pool = sorted((r for r in records
                           if r["_source"] == "IMCS-21" and r not in picked),
                          key=lambda r: (distance(r), r["_tiebreak"]))
            donors = sorted((r for r in picked if r["_source"] != "IMCS-21"),
                            key=lambda r: (-distance(r), r["_tiebreak"]))
            for candidate, donor in zip(pool, donors):
                if have >= imcs_floor:
                    break
                picked[picked.index(donor)] = candidate
                have += 1
    return picked


def measure(picked: list) -> dict:
    counts = collections.Counter()
    for record in picked:
        for key, value in record["_counts"].items():
            counts[key] += value
    by_source = collections.Counter(record["_source"] for record in picked)
    return {
        "records": len(picked),
        "action_counts": dict(counts),
        "action_share": shares(counts),
        "by_source": dict(by_source),
        "source_share": {source: count / max(len(picked), 1)
                         for source, count in by_source.items()},
        "doctor_turns_histogram": dict(collections.Counter(
            str(record["_doctor_turns"]) for record in picked)),
        "bucket_counts": dict(collections.Counter(record["_bucket"] for record in picked)),
        "mean_doctor_turns": round(
            sum(record["_doctor_turns"] for record in picked) / max(len(picked), 1), 3),
    }


def meets(stats: dict, n: int, imcs_floor: int, split: str) -> bool:
    mix = stats["action_share"]
    return (mix["wen"] >= CONSTRAINTS["wen"]
            and mix["dig"] >= CONSTRAINTS["dig"]
            and mix["talk"] <= CONSTRAINTS["talk_max"]
            and stats["source_share"].get("KaMed", 0.0) <= KAIMED_CAP
            and sum(1 for label, _, _ in DOCTOR_BUCKETS if stats["bucket_counts"].get(label))
            == len(DOCTOR_BUCKETS)
            and stats["by_source"].get("IMCS-21", 0) >= min(imcs_floor, n))


def relax(weights: dict) -> dict:
    """Move weight from the longest consultations to the shortest: the action
    constraints are what the long tail makes impossible."""
    labels = [label for label, _, _ in DOCTOR_BUCKETS]
    weights = dict(weights)
    step = min(0.02, weights[labels[-1]])
    if step <= 0.0005:
        return weights
    for label in labels[-2:]:
        if weights[label] > 0.005:
            weights[label] -= step
    weights[labels[0]] += step * (len(labels) - 2)
    return weights


def select(records: list, n: int, seed: int, split: str, imcs_floor: int) -> tuple:
    weights = {"1-3": 0.50, "4-7": 0.32, "8-15": 0.14, "16-20": 0.04}
    kam_allowance = int(KAIMED_CAP * n)
    best = None
    for attempt in range(60):
        picked = draw(records, n, weights, kam_allowance, imcs_floor)
        stats = measure(picked)
        if best is None or (meets(stats, n, imcs_floor, split)
                            and not meets(best[1], n, imcs_floor, split)):
            best = (picked, stats, dict(weights))
        if meets(stats, n, imcs_floor, split):
            return picked, stats, weights, attempt
        weights = relax(weights)
    return best[0], best[1], best[2], 60


def write_split(output: Path, split: str, records: list) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{split}.jsonl"
    rows = sorted(records, key=lambda r: r["sample_id"])
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in rows:
            clean = {key: value for key, value in record.items() if not key.startswith("_")}
            stream.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return path


def required_fields(records: list) -> dict:
    needed = ("case_id", "source", "split", "n_doctor_turns", "quality", "messages", "sample_id")
    missing = {field: 0 for field in needed}
    for record in records:
        for field in needed:
            if field not in record:
                missing[field] += 1
    return {"fields": list(needed), "missing": missing,
            "complete": all(count == 0 for count in missing.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="directory of the full dialogue-mode SFT export")
    parser.add_argument("--turn-input", type=Path, default=DEFAULT_TURN_INPUT,
                        help="directory of the full turn-mode SFT export")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turn-output", type=Path, default=DEFAULT_TURN_OUTPUT)
    parser.add_argument("--rl-dir", type=Path, default=DEFAULT_RL_DIR,
                        help="RL parquet directory, for the subset check")
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--dev", type=int, default=1250)
    parser.add_argument("--test", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imcs-floor", type=int, default=IMCS_FLOOR,
                        help="IMCS-21 records the train draw must contain")
    parser.add_argument("--skip-turn", action="store_true")
    args = parser.parse_args()

    quotas = {"train": args.train, "dev": args.dev, "test": args.test}
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": "scripts/sample_sft_stratified.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "seed": args.seed,
        "quotas": quotas,
        "input": rel(args.input),
        "turn_input": None if args.skip_turn else rel(args.turn_input),
        "output": rel(args.output),
        "turn_output": None if args.skip_turn else rel(args.turn_output),
        "targets": {"ideal": IDEAL, "constraints": CONSTRAINTS,
                    "kaimed_cap": KAIMED_CAP, "imcs_floor": args.imcs_floor,
                    "max_doctor_turns": MAX_DOCTOR_TURNS},
        "splits": {},
    }

    all_records: dict = {}
    for split in SPLITS:
        records = prepare(load_split(args.input, split), args.seed)
        n = quotas[split]
        floor = round(args.imcs_floor * n / max(args.train, 1)) if split != "train" else args.imcs_floor
        picked, stats, weights, attempts = select(records, n, args.seed, split, floor)
        path = write_split(args.output, split, picked)
        all_records[split] = picked
        stats["file"] = rel(path)
        stats["available"] = len(records)
        stats["weights"] = weights
        stats["weight_attempts"] = attempts
        stats["imcs_floor"] = floor
        stats["meets_constraints"] = meets(stats, n, floor, split)
        stats["fields"] = required_fields(picked)
        report["splits"][split] = stats
        print(f"{split}: 抽 {len(picked):,} / 候选 {len(records):,}"
              f"  WEN {stats['action_share']['wen']:.1%}"
              f"  DIG {stats['action_share']['dig']:.1%}"
              f"  TALK {stats['action_share']['talk']:.1%}"
              f"  KaMed {stats['source_share'].get('KaMed', 0):.1%}"
              f"  IMCS {stats['by_source'].get('IMCS-21', 0):,}"
              f"  约束 {'通过' if stats['meets_constraints'] else '未达'}")

    totals_counts = collections.Counter()
    for picked in all_records.values():
        for record in picked:
            for key, value in record["_counts"].items():
                totals_counts[key] += value
    report["totals"] = {
        "records": sum(len(picked) for picked in all_records.values()),
        "action_counts": dict(totals_counts),
        "action_share": shares(totals_counts),
        "by_source": dict(collections.Counter(
            record["_source"] for picked in all_records.values() for record in picked)),
    }

    # The sample must be a subset of the RL episodes, in the same split.
    rl_sets = E.rl_case_sets(args.rl_dir) if args.rl_dir else {}
    if rl_sets:
        rl_all = set().union(*rl_sets.values()) if rl_sets else set()
        rows = []
        mismatched = 0
        for split, picked in all_records.items():
            for record in picked:
                case_id = record["case_id"]
                if case_id not in rl_sets.get(split, set()):
                    mismatched += 1
                rows.append(case_id)
        sampled = set(rows)
        report["case_set_relation"] = {
            "rl_dir": rel(args.rl_dir),
            "sampled_cases": len(sampled),
            "rl_cases": len(rl_all),
            "subset_of_rl": not (sampled - rl_all),
            "outside_rl": len(sampled - rl_all),
            "split_conflicts": mismatched,
            "rl_minus_sampled": len(rl_all - sampled),
        }
        print(f"case 集合：{len(sampled):,} 条，"
              f"不在 RL 中 {len(sampled - rl_all):,}，split 冲突 {mismatched:,}")

    if not args.skip_turn and args.turn_input.exists():
        turn_report = {}
        turn_records = {}
        for split in SPLITS:
            records = prepare(load_split(args.turn_input, split), args.seed)
            n = quotas[split]
            floor = round(args.imcs_floor * n / max(args.train, 1)) if split != "train" else args.imcs_floor
            picked, stats, weights, attempts = select(records, n, args.seed, split, floor)
            path = write_split(args.turn_output, split, picked)
            turn_records[split] = picked
            stats["file"] = rel(path)
            stats["available"] = len(records)
            stats["weights"] = weights
            stats["weight_attempts"] = attempts
            stats["imcs_floor"] = floor
            stats["meets_constraints"] = meets(stats, n, floor, split)
            stats["fields"] = required_fields(picked)
            turn_report[split] = stats
            print(f"turn {split}: 抽 {len(picked):,} / 候选 {len(records):,}"
                  f"  DIG {stats['action_share']['dig']:.1%}"
                  f"  KaMed {stats['source_share'].get('KaMed', 0):.1%}")
        turn_counts = collections.Counter()
        for picked in turn_records.values():
            for record in picked:
                for key, value in record["_counts"].items():
                    turn_counts[key] += value
        report["turn_mode"] = {
            "splits": turn_report,
            "totals": {"records": sum(len(p) for p in turn_records.values()),
                       "action_counts": dict(turn_counts),
                       "action_share": shares(turn_counts)},
        }

    report["baseline_reference"] = BASELINE
    report["reproduce"] = ("python scripts/sample_sft_stratified.py "
                           f"--seed {args.seed} --train {args.train} "
                           f"--dev {args.dev} --test {args.test}")
    (args.output / "_sample_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告 {args.output / '_sample_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
