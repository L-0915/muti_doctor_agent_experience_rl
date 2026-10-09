"""Quality-check and translate the RL decision points into English.

One API call per consultation, not per decision point. The rows describe the
same consultation several times over, once per point at which a doctor spoke,
and the visible history and the target are both slices of `full_dialogue`. So
the consultation is translated once and the slices are read back out of the
result: the current 90,000-row export covers 58,808 distinct consultations, so
it costs 58,808 calls rather than 90,000.

Each call returns a verdict on whether the consultation is usable and the
translation of every turn. Verdicts are per consultation; the decisions drawn
from it inherit them.

Results stream to a cache file as they arrive, so an interrupted run resumes
without paying for the calls it already made. Each cache line carries a hash of
the dialogue it translated: the case_id alone does not identify content, and a
re-cleaned corpus reuses the same ids with different text. Entries whose hash
no longer matches are retranslated instead of reused.

Consultations longer than CHUNK_TURNS are translated in chunks, because the
whole-consultation reply of a hundred-turn dialogue does not fit in the output
limit and arrives as truncated JSON. Each chunk is validated on its own; a
consultation is usable only if every chunk is.

Output: data/sft_rl/{train,dev,test}_en.parquet, data/sft_rl/_translate_report.json

Usage:
    python scripts/translate_rl_data.py
    python scripts/translate_rl_data.py --concurrency 16 --limit 200
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/sft_rl"
DEFAULT_OUTPUT = ROOT / "data/sft_rl"
CACHE_PATH = ROOT / "data/sft_rl/_translate_cache.jsonl"
ENV_PATH = ROOT / "data/.env"

# A whole-consultation reply up to this many turns fits comfortably in the
# 8,192-token output limit; beyond it the reply truncates. The four failures in
# the first live run were 74-, 95-, 98- and 100-turn consultations.
CHUNK_TURNS = 40

SYSTEM_PROMPT = """Translate this Chinese medical consultation into English.

Input: {"turns": [{"role": "patient"|"doctor", "text": "中文"}], "reference": {...}}

Output JSON only, same structure with all text in English, plus:
- "usable": false only if the text is unreadable, a sentence stops mid-way, or it
  is not a medical consultation; a conversation ending on the patient's turn is
  complete, and informal wording is not a defect
- "issue": none | truncated | incoherent | garbled | not_consultation

Keep the same turns, the same order and the same roles. Translate faithfully."""

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
ISSUES = {"none", "truncated", "incoherent", "garbled", "not_consultation"}


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_reply(text: str) -> dict | None:
    text = FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        return None
    return payload


def build_request(turns: list[dict], reference: dict) -> str:
    payload = {
        "turns": [{"role": t["role"], "text": t["content"]} for t in turns],
        "reference": {k: v for k, v in reference.items() if v},
    }
    return json.dumps(payload, ensure_ascii=False)


def dialogue_fingerprint(dialogue: list[dict]) -> str:
    payload = "|".join(f"{t['role']}:{t['content']}" for t in dialogue)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def call_chunk(client, semaphore, turns: list[dict], reference: dict,
                     model: str, retries: int) -> dict:
    """Translate and validate one chunk. Raises on any mismatch so the caller
    can retry the chunk; a failure is never written as if it succeeded."""
    request = build_request(turns, reference)
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("output truncated at the token limit")
            parsed = parse_reply(choice.message.content or "")
            if parsed is None:
                raise ValueError("reply is not the expected JSON")
            returned = parsed["turns"]
            if len(returned) != len(turns):
                raise ValueError(
                    f"turn count changed: {len(turns)} -> {len(returned)}")
            for produced, original in zip(returned, turns):
                if produced.get("role") != original["role"]:
                    raise ValueError("role order changed")
            return parsed
        except Exception:  # noqa: BLE001 - retried, then raised by the caller
            if attempt == retries - 1:
                raise
            await asyncio.sleep(min(2 ** attempt + random.random(), 30))
    raise ValueError("unreachable")


async def check_case(client, semaphore, case: dict, model: str,
                     retries: int) -> tuple[str, dict]:
    """Return (case_id, result). A failure is recorded rather than raised, so
    one bad consultation does not stop the run."""
    try:
        dialogue = case["dialogue"]
        chunks = [dialogue[i:i + CHUNK_TURNS]
                  for i in range(0, len(dialogue), CHUNK_TURNS)]
        # The reference block travels with the first chunk only; later chunks
        # get an empty one so the request keeps the shape the prompt describes.
        results = []
        for index, chunk in enumerate(chunks):
            reference = case["reference"] if index == 0 else {}
            results.append(await call_chunk(
                client, semaphore, chunk, reference, model, retries))

        turns_en: list[dict] = []
        blocking = "none"
        for parsed in results:
            turns_en.extend(
                {"role": t["role"], "text": str(t.get("text") or "")}
                for t in parsed["turns"])
            issue = str(parsed.get("issue") or "none")
            if (blocking == "none" and issue in ISSUES and issue != "none"
                    and not bool(parsed.get("usable", True))):
                blocking = issue

        return case["case_id"], {
            "ok": True,
            "usable": blocking == "none",
            "issue": blocking,
            "turns": turns_en,
            "reference": results[0].get("reference") or {},
        }
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return case["case_id"], {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}


def collect_cases(tables: dict) -> dict[str, dict]:
    """One entry per consultation: its full dialogue and reference fields."""
    cases: dict[str, dict] = {}
    for rows in tables.values():
        for row in rows:
            case_id = row["case_id"]
            if case_id in cases:
                continue
            cases[case_id] = {
                "case_id": case_id,
                "dialogue": row["environment"]["patient_state"]["full_dialogue"],
                "reference": row["environment"]["ground_truth"],
                "symptoms": row["environment"]["patient_state"]["symptoms"],
            }
    return cases


def apply_translation(row: dict, result: dict, symptoms_en: list[dict]) -> dict:
    """Rebuild one decision point in English.

    The visible history and the target are slices of the translated dialogue:
    the history is its first `turn_id` turns, and the target is the turn at
    `turn_id` itself. Reading them back out keeps the three consistent by
    construction, with no separate translation to drift.
    """
    translated = result["turns"]
    patient_state = dict(row["environment"]["patient_state"])
    ground_truth = dict(row["environment"]["ground_truth"])
    history = translated[: row["observation"]["turn_id"]]

    patient_state["full_dialogue"] = [
        {"role": t["role"], "content": t["text"]} for t in translated]
    patient_state["symptoms"] = symptoms_en
    reference = result.get("reference") or {}
    if reference.get("diagnosis"):
        ground_truth["diagnosis"] = reference["diagnosis"]
    if reference.get("department"):
        ground_truth["department"] = reference["department"]

    out = dict(row)
    out["observation"] = {
        "messages": [{"role": t["role"], "content": t["text"]} for t in history],
        "turn_id": row["observation"]["turn_id"],
        "conversation_id": row["observation"]["conversation_id"],
    }
    out["environment"] = {"patient_state": patient_state, "ground_truth": ground_truth}
    out["sft_target"] = {
        "action": row["sft_target"]["action"],
        "message": translated[row["observation"]["turn_id"]]["text"],
    }
    out["qc"] = {"usable": result["usable"], "issue": result["issue"]}
    return out


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
        ("qc", pa.struct([("usable", pa.bool_()), ("issue", pa.string())])),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many consultations; 0 for all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("needs pyarrow and openai: pip install -r requirements-llm.txt pyarrow") from exc

    env = load_env(ENV_PATH)
    for key in ("API_KEY", "BASE_URL", "BASE_MODEL"):
        if not env.get(key):
            raise SystemExit(f"missing {key} in {ENV_PATH}")

    tables = {s: pq.read_table(args.input / f"{s}.parquet").to_pylist()
              for s in args.splits}
    cases = collect_cases(tables)
    rows_total = sum(len(v) for v in tables.values())
    print(f"{rows_total:,} 条决策点，来自 {len(cases):,} 个不同对话"
          f"  (每条对话调用一次，省下 {rows_total / len(cases):.1f}x)" )

    done: dict[str, dict] = {}
    stale = 0
    if CACHE_PATH.exists():
        for line in CACHE_PATH.open(encoding="utf-8"):
            entry = json.loads(line)
            case = cases.get(entry["case_id"])
            # An entry is reusable only if it translated exactly the text the
            # current corpus holds. Ids outlive content: a re-cleaned corpus
            # keeps the same case_id with different turns, and entries from
            # before a rebuild (which carry no hash) cannot be checked at all.
            if case is None or entry.get("fp") != dialogue_fingerprint(case["dialogue"]):
                stale += 1
                continue
            done[entry["case_id"]] = entry["result"]
        print(f"缓存命中 {len(done):,} 个对话，过期 {stale:,} 条（重译）")

    pending = [c for cid, c in cases.items() if cid not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"待处理 {len(pending):,} 个对话\n")
    if not pending:
        print("没有新对话需要处理")
    else:
        client = AsyncOpenAI(api_key=env["API_KEY"], base_url=env["BASE_URL"])
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run() -> None:
            finished = 0
            with CACHE_PATH.open("a", encoding="utf-8", newline="\n") as stream:
                tasks = [asyncio.create_task(
                    check_case(client, semaphore, case, env["BASE_MODEL"], args.retries))
                    for case in pending]
                for task in asyncio.as_completed(tasks):
                    case_id, result = await task
                    done[case_id] = result
                    stream.write(json.dumps(
                        {"case_id": case_id,
                         "fp": dialogue_fingerprint(cases[case_id]["dialogue"]),
                         "result": result},
                        ensure_ascii=False) + "\n")
                    stream.flush()
                    finished += 1
                    if finished % 200 == 0 or finished == len(pending):
                        ok = sum(1 for c in done.values() if c.get("ok"))
                        print(f"  {finished:,}/{len(pending):,}  成功 {ok:,}")

        asyncio.run(run())

    ok_cases = {c: r for c, r in done.items() if r.get("ok")}
    usable = {c: r for c, r in ok_cases.items() if r.get("usable")}
    print(f"\n质检结果：可用 {len(usable):,} / 成功翻译 {len(ok_cases):,} / 共 {len(cases):,}")
    issues = collections.Counter(r["issue"] for r in ok_cases.values())
    for issue, count in issues.most_common():
        print(f"   {issue:<18}{count:>8,}")

    schema = arrow_schema()
    written: dict[str, int] = {}
    for split, rows in tables.items():
        rebuilt = []
        for row in rows:
            result = done.get(row["case_id"])
            if not result or not result.get("ok"):
                continue
            symptoms_en = result.get("symptoms") or [
                {"name": f["name"], "value": f["value"]}
                for f in row["environment"]["patient_state"]["symptoms"]]
            rebuilt.append(apply_translation(row, result, symptoms_en))
        out = args.output / f"{split}_en.parquet"
        pq.write_table(pa.Table.from_pylist(rebuilt, schema=schema), out,
                       compression="zstd")
        written[split] = len(rebuilt)
        print(f"写出 {out.name:<22}{len(rebuilt):>8,} 行  {out.stat().st_size / 1e6:>7.1f} MB")

    report = {
        "rows_in": rows_total,
        "cases": len(cases),
        "cases_translated": len(ok_cases),
        "cases_usable": len(usable),
        "rows_out": written,
        "issues": dict(issues),
        "model": env["BASE_MODEL"],
        "concurrency": args.concurrency,
    }
    (args.output / "_translate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告: {args.output / '_translate_report.json'}")
    print(f"缓存: {CACHE_PATH}  (重跑会自动跳过已完成的对话)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
