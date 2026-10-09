"""Parse every raw consultation source into one JSONL of its own.

This is the first step of the pipeline: read the original archives exactly as
released and write one line per dialogue, preserving each source's own
annotation fields. No filtering, deduplication or cleaning happens here.

Output: data/sft_rl/<Source>.jsonl

Usage: python scripts/parse_sources.py
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/sft_rl"

# Display name for each configured source, matching the name used by its authors.
DISPLAY_NAME = {
    "imcs21": "IMCS-21",
    "remedi": "ReMeDi",
    "meddg": "MedDG",
    "chip_mdcfnpc": "CHIP-MDCFNPC",
    "meddialog": "MedDialog",
    "kamed": "KaMed",
    "meddg_session": "MedDG-Session",
}

PATIENT_TOKENS = {"patient", "user", "患", "患者", "病人", "病患", "咨询者"}
DOCTOR_TOKENS = {"doctor", "assistant", "医生", "医师", "大夫"}


def get_nested(record: object, key: str | None):
    if not key or key == "$":
        return record
    value = record
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def role_of(value: object) -> str | None:
    token = str(value or "").strip().lower()
    if token in PATIENT_TOKENS:
        return "patient"
    if token in DOCTOR_TOKENS:
        return "doctor"
    return None


def iter_raw_records(source: dict):
    """Yield (record_id, record) from one configured source, unmodified."""
    path = ROOT / source["path"]
    member = source.get("member")

    if source.get("record_mode") == "zip_session_json":
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.endswith(".session.json"):
                    continue
                with archive.open(entry) as binary:
                    record = json.load(io.TextIOWrapper(binary, encoding="utf-8-sig"))
                yield Path(entry.filename).name.split(".", 1)[0], record
        return

    if member:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as binary:
                handle = io.TextIOWrapper(binary, encoding="utf-8-sig")
                if Path(member).suffix.lower() == ".jsonl":
                    for index, line in enumerate(handle):
                        if line.strip():
                            record = json.loads(line)
                            yield str(get_nested(record, source["columns"].get("id")) or index), record
                else:
                    data = json.load(handle)
                    yield from enumerate_container(data)
        return

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    yield from enumerate_container(data)


def enumerate_container(data: object):
    if isinstance(data, dict):
        # IMCS-21 stores dialogues under top-level case identifiers.
        if all(isinstance(v, (dict, list)) for v in data.values()):
            yield from data.items()
            return
        for key in ("data", "dialogues", "cases"):
            if isinstance(data.get(key), list):
                for index, item in enumerate(data[key]):
                    yield str(index), item
                return
        raise ValueError("Unrecognised top-level structure")
    if isinstance(data, list):
        for index, item in enumerate(data):
            yield str(index), item
        return
    raise ValueError("Unrecognised record container")


def parse_turns(source: dict, record: object) -> list[dict] | None:
    columns = source["columns"]
    raw_turns = get_nested(record, columns["turns"])
    if not isinstance(raw_turns, list):
        return None
    annotation_fields = columns.get("turn_annotation_fields") or []
    annotation_key = columns.get("turn_annotations")
    annotation_keys = annotation_key if isinstance(annotation_key, list) else (
        [annotation_key] if annotation_key else [])

    turns: list[dict] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            return None
        role = role_of(get_nested(item, columns["role"]))
        text = get_nested(item, columns["text"])
        if role is None or not isinstance(text, str):
            return None
        turn = {"role": role, "text": text}
        for field in annotation_fields:
            value = get_nested(item, field)
            if value not in (None, "", [], {}):
                turn[field] = value
        for field in annotation_keys:
            value = get_nested(item, field)
            if value not in (None, "", [], {}):
                turn[field if field != "$" else "annotation"] = value
        turns.append(turn)
    return turns or None


def parse_labels(source: dict, record: object) -> dict:
    columns = source["columns"]
    labels: dict[str, object] = {}
    for key in ("diagnosis", "self_report"):
        field = columns.get(key)
        if not field:
            continue
        value = get_nested(record, field)
        if value not in (None, "", [], {}):
            labels[key] = value
    for name, field in (columns.get("source_labels") or {}).items():
        value = get_nested(record, field)
        if value not in (None, "", [], {}):
            labels[name] = value
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "data/source_config.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    streams: dict[str, object] = {}
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    try:
        for source in config["sources"]:
            name = DISPLAY_NAME.get(source["name"], source["name"])
            if name not in streams:
                path = args.output / f"{name}.jsonl"
                streams[name] = path.open("w", encoding="utf-8", newline="\n")
                counts[name] = 0
                skipped[name] = 0
            stream = streams[name]
            for record_id, record in iter_raw_records(source):
                turns = parse_turns(source, record)
                if not turns:
                    skipped[name] += 1
                    continue
                row = {
                    "id": str(record_id),
                    "source": name,
                    "split": source["split"],
                    "turns": turns,
                }
                labels = parse_labels(source, record)
                if labels:
                    row["labels"] = labels
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[name] += 1
    finally:
        for stream in streams.values():
            stream.close()

    print(f"{'文件':<22}{'对话数':>10}{'跳过':>9}")
    print("-" * 43)
    for name in counts:
        print(f"{name + '.jsonl':<22}{counts[name]:>10,}{skipped[name]:>9,}")
    print("-" * 43)
    print(f"{'合计':<22}{sum(counts.values()):>10,}{sum(skipped.values()):>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
