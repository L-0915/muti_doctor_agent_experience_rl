"""Select a source-balanced short-dialogue synthesis pilot without API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from synthesize_short_cases import seed_skip_reason, sha256


def canonical(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def rank(seed_id: str) -> str:
    return hashlib.sha256(("heapo-short-synthesis-v2:" + seed_id).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/heapo_zh_v1/short_case_seeds_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/audit/short_synthesis_pilot_v2_ids.txt"))
    parser.add_argument("--per-source", type=int, default=2,
                        help="Eligible unique openings per source; 0 selects all")
    args = parser.parse_args()
    if not args.input.is_file() or args.per_source < 0:
        parser.error("Input missing or per-source quota negative")
    buckets = defaultdict(list)
    counts = Counter()
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            seed = json.loads(line)
            counts["all_seeds"] += 1
            reason = seed_skip_reason(seed)
            if reason:
                counts[f"skipped_{reason}"] += 1
                continue
            source = seed["source"]
            buckets[source].append((rank(seed["seed_case_id"]), seed))
            counts[f"eligible_{source}"] += 1
    selected = []
    seen_openings = set()
    for source in sorted(buckets):
        source_count = 0
        for _, seed in sorted(buckets[source], key=lambda item: item[0]):
            opening = canonical(seed["initial_patient_utterance"])
            if opening in seen_openings:
                counts["duplicate_opening_skipped"] += 1
                continue
            selected.append(seed)
            seen_openings.add(opening)
            source_count += 1
            if args.per_source and source_count >= args.per_source:
                break
        counts[f"selected_{source}"] = source_count
    selected.sort(key=lambda item: item["seed_case_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(item["seed_case_id"] for item in selected) +
                           ("\n" if selected else ""), encoding="utf-8")
    preview_path = args.output.with_suffix(".preview.csv")
    with preview_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed_case_id", "source",
                                                  "original_completed_exchanges",
                                                  "initial_patient_utterance"])
        writer.writeheader()
        for seed in selected:
            writer.writerow({name: seed[name] for name in writer.fieldnames})
    manifest = {"input": str(args.input), "input_sha256": sha256(args.input),
                "output": str(args.output), "output_sha256": sha256(args.output),
                "preview": str(preview_path), "per_source": args.per_source,
                "counts": dict(sorted(counts.items())), "api_calls": 0,
                "selected_seed_count": len(selected)}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": len(selected),
                      "api_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
