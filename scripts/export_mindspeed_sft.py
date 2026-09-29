"""Export real HEAPO SFT targets for instruction fine-tuning.

Alpaca format has exactly one supervised assistant response per sample. The
prior conversation is rendered as user-side context, so earlier logged doctor
responses do not accidentally become additional training targets.

OpenAI format preserves the original doctor messages. With ms-swift, use
``--loss_scale last_round`` to train on the final assistant response only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/train.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/mindspeed_sft"
INSTRUCTION_ZH = "根据已公开的患者主诉和问诊记录，输出医生下一步JSON动作。"


def split_for_case(case_id: str, dev_fraction: float) -> str:
    rank = int.from_bytes(hashlib.sha256(
        ("heapo-sft-v1:" + case_id).encode("utf-8")).digest()[:8], "big")
    return "dev" if rank / (1 << 64) < dev_fraction else "train"


def sft_record(row: dict, style: str) -> dict:
    messages = row["messages"]
    target = row["sft_target"]
    if (not isinstance(messages, list) or len(messages) < 2 or
            messages[0]["role"] != "system" or messages[-1]["role"] != "user" or
            any(item["role"] not in {"system", "user", "assistant"}
                or not isinstance(item["content"], str) or not item["content"].strip()
                for item in messages) or
            target["action"] not in {"ASK", "FINAL"} or
            not isinstance(target["message"], str) or not target["message"].strip()):
        raise ValueError("Malformed SFT example: " + row["sample_id"])
    answer = json.dumps({"action": target["action"],
                         "message": target["message"]}, ensure_ascii=False,
                        separators=(",", ":"))
    if style == "openai":
        return {"messages": [*messages, {"role": "assistant", "content": answer}]}
    if style == "alpaca":
        history = "\n\n".join(
            ("患者：" if item["role"] == "user" else "医生：") + item["content"]
            for item in messages[1:])
        return {"system": messages[0]["content"],
                "instruction": INSTRUCTION_ZH,
                "input": history,
                "output": answer}
    raise ValueError("Unsupported output style")


def export(input_path: Path, output_dir: Path, style: str,
           dev_fraction: float = 0.1) -> dict:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if input_path.resolve() == output_dir.resolve():
        raise ValueError("Input and output directory must differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.{style}.jsonl"
             for split in ("train", "dev")}
    counts = Counter()
    cases_by_split = {"train": set(), "dev": set()}
    with (paths["train"].open("w", encoding="utf-8", newline="\n") as train_file,
          paths["dev"].open("w", encoding="utf-8", newline="\n") as dev_file):
        writers = {"train": train_file, "dev": dev_file}
        columns = ["sample_id", "case_id", "messages", "sft_target", "metadata"]
        for batch in pq.ParquetFile(input_path).iter_batches(
                batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                if row["metadata"]["synthetic"]:
                    counts["excluded_synthetic"] += 1
                    continue
                if row["sft_target"] is None:
                    continue
                if row["metadata"]["language"] != "zh":
                    raise ValueError("This export expects Chinese training cases")
                split = split_for_case(row["case_id"], dev_fraction)
                record = sft_record(row, style)
                writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                cases_by_split[split].add(row["case_id"])
                counts[f"{split}_samples"] += 1
                counts[f"{split}_{row['sft_target']['action'].lower()}"] += 1
    if not counts["train_samples"] or not counts["dev_samples"]:
        raise ValueError("Both training and validation sets need labeled examples")
    if cases_by_split["train"] & cases_by_split["dev"]:
        raise ValueError("Case leakage between SFT train and dev")
    report = {"input": str(input_path), "style": style,
              "case_grouped_split": True, "dev_fraction": dev_fraction,
              "counts": dict(counts),
              "unique_cases": {key: len(value) for key, value in cases_by_split.items()},
              "files": {key: str(value) for key, value in paths.items()},
              "requires_last_response_only_loss": style == "openai"}
    (output_dir / f"{style}.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--style", choices=("alpaca", "openai"), default="alpaca")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.output_dir, args.style,
                            args.dev_fraction), ensure_ascii=False))


if __name__ == "__main__":
    main()
