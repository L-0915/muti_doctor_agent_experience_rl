"""Extract the per-episode key-fact list from the ENGLISH RL parquet.

One record per episode:

    {"case_id": ..., "sample_id": ..., "split": ...,
     "facts": [{"category": "symptom", "name": "tinnitus",
                "value": "bilateral, continuous, worse in quiet",
                "source_turn": 2,
                "evidence": "both ears keep ringing when it is quiet"}],
     "dropped": [...]}

Categories (fixed, exactly seven):

    symptom          症状      — what the patient reports feeling
    duration         病程      — how long it has been going on
    medical_history  既往史    — prior conditions, surgeries, family history
    allergy_history  过敏史    — drug / food / other allergies
    medication       用药      — drugs the patient is currently taking, and
                                drug advice tied to the conclusion
    cause            病因      — the cause named by the doctor's conclusion
    disease_name     病名      — the disease named by the doctor's conclusion

Patient-side categories must cite the patient turn that discloses the fact
(source_turn + evidence quote); cause / disease_name / medication advice come
from the reference conclusion. Every fact is validated against the dialogue:
a fact whose evidence cannot be located at its cited turn is dropped and
logged, never silently kept.

The LLM call is left as an empty stub on purpose (see call_llm); fill it with
whatever client the platform provides, keep it returning the raw model string.

Results stream to a cache keyed by case_id + dialogue fingerprint, so reruns
skip finished episodes and re-extract ones whose dialogue changed.

Usage:
    python scripts/extract_key_facts.py --input data/sft_rl_v4/rl_en
    python scripts/extract_key_facts.py --input data/sft_rl_v4/rl_en --limit 1000
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/sft_rl_v4/rl_en"
DEFAULT_OUTPUT = ROOT / "data/sft_rl_v4/facts"
CACHE_NAME = "_facts_cache.jsonl"

PATIENT_SIDE = {"symptom", "duration", "medical_history", "allergy_history",
                "medication"}
CONCLUSION_SIDE = {"cause", "disease_name"}
CATEGORIES = PATIENT_SIDE | CONCLUSION_SIDE
MAX_FACTS = 10

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", str(text or "").strip().lower())


def fingerprint(dialogue: list[dict]) -> str:
    payload = "|".join(f"{t['role']}:{t['content']}" for t in dialogue)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_episodes(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    episodes = []
    for row in rows:
        extra = row["extra_info"]
        episodes.append({
            "case_id": extra["case_id"],
            "sample_id": extra["sample_id"],
            "dialogue": extra["patient_state"]["full_dialogue"],
            "conclusion": row["ground_truth"]["reference_conclusion"] or "",
            "department": row["ground_truth"]["department_path"]
            or row["ground_truth"]["department"] or "",
            "conclusion_turn": extra.get("conclusion_turn_index"),
        })
    return episodes


def build_prompt(episode: dict) -> str:
    """The full extraction prompt; English in, JSON out."""
    turns = [{"role": t["role"], "text": t["content"]}
             for t in episode["dialogue"]]
    request = {
        "turns": turns,
        "reference_conclusion": episode["conclusion"],
        "department": episode["department"],
    }
    return json.dumps(request, ensure_ascii=False)


SYSTEM_PROMPT = """Extract the key medical facts of this consultation.

Input JSON: {"turns": [{"role": "patient"|"doctor", "text": ...}],
"reference_conclusion": <the doctor's final substantive conclusion>,
"department": <department path>}

Output JSON only:
{"facts": [{"category": <one of the seven below>, "name": <short fact name>,
"value": <the fact stated as attribute: value, in English>,
"source_turn": <0-based turn index that discloses it, or null>,
"evidence": <short verbatim quote from that turn or from the conclusion>}]}

Categories:
- symptom: what the patient reports (e.g. tinnitus, abdominal pain). Split
  distinct complaints into separate facts.
- duration: how long the complaint has lasted (e.g. 2 weeks).
- medical_history: prior conditions, surgeries, family history.
- allergy_history: drug/food/other allergies, including "no known allergy"
  only if the patient explicitly says so.
- medication: drugs the patient currently takes; drug advice that the
  reference conclusion gives also goes here with evidence quoted from the
  conclusion.
- cause: the cause the conclusion attributes the complaint to (from the
  reference conclusion, source_turn null).
- disease_name: the disease the conclusion names, or the department's
  condition label (from reference_conclusion / department, source_turn null).

Rules:
1. A patient-side fact (symptom, duration, medical_history, allergy_history,
   medication in current use) MUST cite the patient turn that discloses it
   and quote its exact words as evidence. Do not invent, do not use doctor
   turns as the source.
2. cause / disease_name / conclusion medication advice cite the reference
   conclusion (source_turn null, evidence from the conclusion).
3. Only include a category if the dialogue or conclusion actually contains
   it. At most 10 facts total, the ones the conclusion relies on first.
4. No chit-chat, no purely platform content. English only."""


def call_llm(prompt: str) -> str:
    """FIXME(user): plug the platform LLM here.

    Input is the request JSON built by build_prompt; return the raw model
    reply string. Suggested settings: temperature 0, JSON response mode.
    Raise on failure so the caller retries; never return a half reply.
    """
    raise NotImplementedError("fill in the platform LLM call")


def parse_reply(raw: str) -> dict | None:
    text = FENCE.sub("", str(raw or "")).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return None
    return parsed


def validate(parsed: dict, episode: dict) -> tuple[list[dict], list[dict]]:
    """Keep only facts whose evidence can be located; return (kept, dropped)."""
    dialogue = episode["dialogue"]
    conclusion = norm(episode["conclusion"])
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set = set()
    for fact in parsed.get("facts", []):
        category = fact.get("category")
        name = str(fact.get("name") or "").strip()
        value = str(fact.get("value") or "").strip()
        evidence = norm(fact.get("evidence"))
        source = fact.get("source_turn")
        if category not in CATEGORIES:
            dropped.append({"reason": "bad_category", "fact": fact})
            continue
        if not name or not value:
            dropped.append({"reason": "empty_field", "fact": fact})
            continue
        key = (category, name.lower())
        if key in seen:
            dropped.append({"reason": "duplicate", "fact": fact})
            continue
        if category in PATIENT_SIDE:
            if not isinstance(source, int) or not 0 <= source < len(dialogue):
                dropped.append({"reason": "bad_source_turn", "fact": fact})
                continue
            if dialogue[source]["role"] != "patient":
                dropped.append({"reason": "not_patient_turn", "fact": fact})
                continue
            if evidence and evidence not in norm(dialogue[source]["content"]):
                dropped.append({"reason": "evidence_mismatch", "fact": fact})
                continue
        else:
            if evidence and conclusion and evidence not in conclusion:
                dropped.append({"reason": "evidence_mismatch", "fact": fact})
                continue
        kept.append({"category": category, "name": name, "value": value,
                     "source_turn": source if isinstance(source, int) else None,
                     "evidence": str(fact.get("evidence") or "").strip()})
        seen.add(key)
    return kept[:MAX_FACTS], dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="directory of the ENGLISH rl parquet files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0,
                        help="extract at most this many episodes per split")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / CACHE_NAME

    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            entry = json.loads(line)
            done[entry["case_id"]] = entry
        print(f"缓存命中 {len(done):,} 条")

    stats = {"episodes": 0, "failed": 0}
    by_category: collections.Counter = collections.Counter()
    drop_reasons: collections.Counter = collections.Counter()
    no_facts = 0

    for split in args.splits:
        path = args.input / f"{split}.parquet"
        if not path.exists():
            print(f"跳过 {split}：{path} 不存在")
            continue
        episodes = load_episodes(path)
        if args.limit:
            episodes = episodes[: args.limit]
        out_path = args.output / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as out, \
                cache_path.open("a", encoding="utf-8", newline="\n") as cache:
            for episode in episodes:
                stats["episodes"] += 1
                fp = fingerprint(episode["dialogue"])
                cached = done.get(episode["case_id"])
                if cached and cached.get("fp") == fp:
                    record = cached["record"]
                else:
                    record = None
                    for attempt in range(args.retries):
                        try:
                            raw = call_llm(build_prompt(episode))
                            parsed = parse_reply(raw)
                            if parsed is None:
                                raise ValueError("reply is not the expected JSON")
                            facts, dropped = validate(parsed, episode)
                            record = {"facts": facts, "dropped": dropped}
                            break
                        except NotImplementedError:
                            raise SystemExit(
                                "call_llm 还是空的：填入平台调用后重跑")
                        except Exception as exc:  # noqa: BLE001
                            if attempt == args.retries - 1:
                                print(f"[FAIL] {episode['case_id']}: {exc}")
                            else:
                                time.sleep(2 ** attempt)
                    if record is None:
                        stats["failed"] += 1
                        continue
                    entry = {"case_id": episode["case_id"], "fp": fp,
                             "record": record}
                    cache.write(json.dumps(entry, ensure_ascii=False) + "\n")
                row = {"case_id": episode["case_id"],
                       "sample_id": episode["sample_id"],
                       "split": split,
                       "facts": record["facts"],
                       "dropped": record["dropped"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                for fact in record["facts"]:
                    by_category[fact["category"]] += 1
                for miss in record["dropped"]:
                    drop_reasons[miss["reason"]] += 1
                if not record["facts"]:
                    no_facts += 1
        print(f"{split}: {len(episodes):,} episode → {out_path}")

    report = {"episodes": stats["episodes"], "failed": stats["failed"],
              "no_facts": no_facts, "by_category": dict(by_category),
              "drop_reasons": dict(drop_reasons)}
    (args.output / "_facts_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
