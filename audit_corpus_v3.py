"""Full-corpus census for corpus v3: what is wrong, how often, and where.

Read-only. Every S1/S2/S3 rule from the task book is evaluated over every case
of the input directory with the same engine the cleaner uses
(scripts/corpus_v3_rules.py), so the numbers here are the numbers the cleaner
will act on, not an estimate. Two passes: the first indexes turn-sequence
fingerprints (D07 needs the whole corpus before it can judge one case), the
second classifies.

Examples are drawn by reservoir sampling under a fixed seed, so a rerun on the
same corpus reproduces the same report.

Output:
  data/audit_v3/census.json
  docs/audit/corpus_census_before.md

Usage:
  python scripts/audit_corpus_v3.py
  python scripts/audit_corpus_v3.py --input data/processed_2 --limit 5000
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402  episode semantics, split
import corpus_v3_rules as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed_2"
DEFAULT_OUTPUT = ROOT / "data/audit_v3"
DEFAULT_DOC = ROOT / "docs/audit/corpus_census_before.md"

SEED = 20261016
EXAMPLES_PER_RULE = 3
EXCERPT = 80
WHITESPACE = re.compile(r"\s+")
REPEAT_TWO = re.compile(r"([，。、；：！？,.!?;:])\1{1}")

TURN_BUCKETS = [
    ("1-2", 0, 2), ("3-4", 3, 4), ("5-6", 5, 6), ("7-10", 7, 10),
    ("11-15", 11, 15), ("16-20", 16, 20), ("21-30", 21, 30), ("31+", 31, 10 ** 6),
]
DOCTOR_BUCKETS = [
    ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8),
    ("9-12", 9, 12), ("13-20", 13, 20), ("21+", 21, 10 ** 6),
]
CJK_BUCKETS = [("<0.10", 0.0, 0.0999), ("0.10-0.39", 0.10, 0.3999),
               ("0.40-0.69", 0.40, 0.6999), ("0.70-0.94", 0.70, 0.9499),
               ("0.95-1.00", 0.95, 1.0001)]
LEAD_BUCKETS = [("0-1", 0, 1), ("2-5", 2, 5), ("6-15", 6, 15),
                ("16-40", 16, 40), ("41+", 41, 10 ** 6)]


def rel(path) -> str:
    """Repo-relative when inside the project, so the report travels."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def excerpt(text: str, size: int = EXCERPT) -> str:
    """One line, at most size characters, so examples never flood a table."""
    flat = WHITESPACE.sub(" ", str(text or "")).strip()
    return flat if len(flat) <= size else flat[: size - 1] + "…"


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


def example_for_drop(code: str, texts: list, roles: list) -> str:
    """The most relevant text for a drop rule: the offending turn when the
    rule names one, otherwise the opening turn."""
    if code == "D03":
        for text in texts:
            if any(pattern.search(text) for _, pattern in R.PRIVACY_PATTERNS):
                return text
    if code == "D05":
        for text in texts:
            if R.CONTROL_CHARS.search(text):
                return text
    if code == "D06":
        for index, role in enumerate(roles):
            if role == "patient":
                return texts[index]
    if code == "D01":
        return " / ".join(texts[:2])
    return texts[0] if texts else ""


class Reservoir:
    """Deterministic sample of at most n hits, uniform over the whole corpus."""

    def __init__(self, n: int = EXAMPLES_PER_RULE, seed: int = SEED):
        self.n = n
        self.random = random.Random(seed)
        self.items: list = []
        self.seen = 0

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.n:
            self.items.append(item)
            return
        pick = self.random.randrange(self.seen)
        if pick < self.n:
            self.items[pick] = item

    def as_list(self) -> list:
        return list(self.items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for census.json")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC,
                        help="markdown census report path")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N cases (smoke tests only; D07 then "
                             "reflects the prefix only)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--progress-every", type=int, default=20000,
                        help="print progress every N cases; 0 disables")
    args = parser.parse_args()

    started = time.time()
    files = R.input_files(args.input)
    if not files:
        raise SystemExit(f"no corpus jsonl under {args.input}")
    file_bytes = sum(p.stat().st_size for p in files)

    # ---- pass 1: fingerprints, so D07 can see the whole corpus
    print("[1/2] 指纹与首问分组索引（D07 与切分都需要全量）…", flush=True)
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
        if args.progress_every and indexed % (args.progress_every * 5) == 0:
            print(f"      {indexed:,} 条已索引，{time.time() - started:.0f}s", flush=True)
    prefix_merges = 0
    for short, longer in R.prefix_merges(opening_of):
        groups_uf.union(opening_of[short], opening_of[longer])
        prefix_merges += 1
    group_sizes = sorted((len(members) for members in groups_uf.components().values()),
                         reverse=True)
    print(f"      {indexed:,} 条，唯一指纹 {len(first_seen):,}，"
          f"首问分组 {len(group_sizes):,} 组（最大 {group_sizes[0] if group_sizes else 0:,}），"
          f"{time.time() - started:.0f}s", flush=True)

    # ---- pass 2: classify
    print("[2/2] 规则判定…", flush=True)
    rule_examples = {code: Reservoir(EXAMPLES_PER_RULE, args.seed)
                     for code in (*R.DROP_RULES, *R.S2_RULES, *R.FLAG_RULES)}
    by_source: dict = collections.defaultdict(
        lambda: {"cases": 0, "turns": 0, "doctor_turns": 0, "patient_turns": 0,
                 "turn_counts": [], "doctor_counts": []})
    drop_hits: collections.Counter = collections.Counter()
    drop_by_source: dict = collections.defaultdict(collections.Counter)
    repair_hits: collections.Counter = collections.Counter()
    repair_by_source: dict = collections.defaultdict(collections.Counter)
    repair_cases: collections.Counter = collections.Counter()
    flag_hits: collections.Counter = collections.Counter()
    flag_by_source: dict = collections.defaultdict(collections.Counter)
    primary: collections.Counter = collections.Counter()
    primary_by_source: dict = collections.defaultdict(collections.Counter)

    multi_hit = dropped_cases = dropped_turns = 0
    kept_turns = kept_doctor_turns = kept_cases = quality_passed = manual_queue = 0
    cases = turns_total = doctor_total = patient_total = 0
    turn_counts_all: list = []
    doctor_counts_all: list = []
    cjk_ratios: list = []
    first_role: collections.Counter = collections.Counter()
    startup_leading: list = []
    privacy_patterns: collections.Counter = collections.Counter()
    placeholder_forms: collections.Counter = collections.Counter()
    punct_two = punct_three = cross_turn_repeat = 0
    invalid_entities = 0
    entity_fields: collections.Counter = collections.Counter()
    image_exchange_cases = 0
    platform_blank_cases = 0
    department_topics: collections.Counter = collections.Counter()
    post_conclusion_cases = 0
    post_conclusion_new_info = 0

    for path, line_no, row in R.iter_rows(args.input, limit=args.limit):
        cases += 1
        source = str(row.get("source") or path.stem)
        turns = R.turns_of(row)
        texts = [str(t.get("text") or "") for t in turns]
        roles = [str(t.get("role") or "") for t in turns]
        doctor_turns = roles.count("doctor")
        stats = by_source[source]
        stats["cases"] += 1
        stats["turns"] += len(turns)
        stats["doctor_turns"] += doctor_turns
        stats["patient_turns"] += roles.count("patient")
        stats["turn_counts"].append(len(turns))
        stats["doctor_counts"].append(doctor_turns)
        turns_total += len(turns)
        doctor_total += doctor_turns
        patient_total += roles.count("patient")
        turn_counts_all.append(len(turns))
        doctor_counts_all.append(doctor_turns)
        first_role[roles[0] if roles else "none"] += 1

        # threshold evidence, recorded whether or not a rule fires
        ratio, _, _ = R.letters_ratio(texts)
        if ratio is not None:
            cjk_ratios.append(ratio)
        if roles and roles[0] != "patient":
            lead = next((i for i, role in enumerate(roles) if role == "patient"), None)
            startup_leading.append(R.substantive_len(texts[lead]) if lead is not None else 0)
        for text in texts:
            if R.PUNCT_RUN.search(text):
                punct_three += 1
            elif REPEAT_TWO.search(text):
                punct_two += 1
            for match in R.PLACEHOLDER.finditer(text):
                placeholder_forms[match.group(0)] += 1
        if any(R.V2.IMAGE_EXCHANGE.search(text) for text in texts):
            image_exchange_cases += 1
        if any(R.PLATFORM_BLANK.search(text) for text in texts):
            platform_blank_cases += 1
        if B.is_department_name((row.get("labels") or {}).get("topic")):
            department_topics[source] += 1
        for name in R.hitting_privacy(texts):
            privacy_patterns[name] += 1
        previous_doctor = None
        run = 1
        for index, role in enumerate(roles):
            if role != "doctor":
                continue
            key = re.sub(r"[\s，。、；：！？,.!?;:]+", "", texts[index])
            run = run + 1 if key and key == previous_doctor else 1
            previous_doctor = key
            if run == 2:
                cross_turn_repeat += 1
        for turn in turns:
            for key, value in turn.items():
                if key in ("role", "text") or not isinstance(value, list):
                    continue
                for item in value:
                    if isinstance(item, dict) and "name" in item:
                        entity_fields[key] += 1
                        if R.invalid_entity(item):
                            invalid_entities += 1

        result = R.analyse(row, dup_of=first_seen)
        if args.progress_every and cases % args.progress_every == 0:
            rate = (time.time() - started) / cases * 1000
            print(f"      {cases:,} 条判定完毕，{time.time() - started:.0f}s"
                  f"（{rate:.1f}ms/条，含索引）", flush=True)
        context = {"source": source, "id": str(row.get("id")), "line": line_no}

        drops = result["drops"] or result["post_drops"]
        if drops:
            dropped_cases += 1
            dropped_turns += len(turns)
            code = next(code for code in R.DROP_RULES if code in drops)
            primary[code] += 1
            primary_by_source[source][code] += 1
            if len(drops) > 1:
                multi_hit += 1
            for hit_code, detail in drops.items():
                drop_hits[hit_code] += 1
                drop_by_source[hit_code][source] += 1
                if result["post_drops"]:
                    detail = "post_repair|" + detail
                rule_examples[hit_code].add({
                    **context, "detail": detail,
                    "text": excerpt(example_for_drop(hit_code, texts, roles))})
            continue

        kept_cases += 1
        conclusion_index = B.conclusion_of(R.turns_of(row))
        if conclusion_index is not None:
            tail = B.post_conclusion(R.turns_of(row), conclusion_index)
            if tail["post_conclusion_turns"] > 0:
                post_conclusion_cases += 1
            if tail["post_conclusion_has_new_info"]:
                post_conclusion_new_info += 1
        kept_turns += len(result["turns"])
        kept_doctor_turns += sum(1 for t in result["turns"] if t.get("role") == "doctor")
        if result["quality"].get("passed"):
            quality_passed += 1
        manual_queue += len(result["manual"])
        touched = set()
        for op in result["ops"]:
            code = op["rule_code"]
            touched.add(code)
            repair_hits[code] += 1
            repair_by_source[code][source] += 1
            index = op["turn_index"]
            rule_examples[code].add({
                **context, "detail": op["detail"], "turn_index": index,
                "text": excerpt(texts[index] if index < len(texts) else "")})
        for code in touched:
            repair_cases[code] += 1
        for code in result["flags"]:
            flag_hits[code] += 1
            flag_by_source[code][source] += 1
            rule_examples[code].add({
                **context, "detail": "flag",
                "text": excerpt(texts[0] if texts else "")})

    elapsed = time.time() - started
    source_names = sorted(by_source)
    print(f"      {cases:,} 条，用时 {elapsed:.0f}s", flush=True)

    def block(codes, totals, per_source):
        out = {}
        for code in codes:
            out[code] = {
                "title": R.RULE_TITLES[code],
                "hits": totals.get(code, 0),
                "by_source": {src: per_source[code].get(src, 0) for src in source_names},
                "share_of_cases": round(totals.get(code, 0) / max(cases, 1), 4),
                "examples": rule_examples[code].as_list(),
            }
        return out

    census = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": "scripts/audit_corpus_v3.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "input": rel(args.input),
        "input_resolved": str(Path(args.input).resolve()),
        "limit": args.limit,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "files": [{"name": p.name, "bytes": p.stat().st_size} for p in files],
        "input_bytes": file_bytes,
        "rule_titles": R.RULE_TITLES,
        "blocking_flags": sorted(R.BLOCKING_FLAGS),
        "totals": {
            "cases": cases,
            "turns": turns_total,
            "doctor_turns": doctor_total,
            "patient_turns": patient_total,
            "mean_turns": round(turns_total / max(cases, 1), 3),
            "median_turns": median(turn_counts_all),
            "mean_doctor_turns": round(doctor_total / max(cases, 1), 3),
            "median_doctor_turns": median(doctor_counts_all),
            "turns_histogram": histogram(turn_counts_all, TURN_BUCKETS),
            "doctor_turns_histogram": histogram(doctor_counts_all, DOCTOR_BUCKETS),
            "first_turn_role": dict(first_role),
        },
        "by_source": {src: {
            "cases": stats["cases"],
            "turns": stats["turns"],
            "doctor_turns": stats["doctor_turns"],
            "patient_turns": stats["patient_turns"],
            "cases_share": round(stats["cases"] / max(cases, 1), 4),
            "mean_turns": round(stats["turns"] / max(stats["cases"], 1), 3),
            "median_doctor_turns": median(stats["doctor_counts"]),
            "turns_histogram": histogram(stats["turn_counts"], TURN_BUCKETS),
            "doctor_turns_histogram": histogram(stats["doctor_counts"], DOCTOR_BUCKETS),
        } for src, stats in sorted(by_source.items())},
        "drops": block(R.DROP_RULES, drop_hits, drop_by_source),
        "repairs": block(R.S2_RULES, repair_hits, repair_by_source),
        "repair_cases": {code: repair_cases.get(code, 0) for code in R.S2_RULES},
        "flags": block(R.FLAG_RULES, flag_hits, flag_by_source),
        "drop_decision": {
            "cases_dropped": dropped_cases,
            "cases_dropped_share": round(dropped_cases / max(cases, 1), 4),
            "turns_dropped": dropped_turns,
            "cases_with_multiple_drop_rules": multi_hit,
            "primary_rule": dict(primary),
            "primary_rule_by_source": {src: dict(primary_by_source[src])
                                       for src in source_names},
        },
        "projection": {
            "cases_kept": kept_cases,
            "turns_kept": kept_turns,
            "doctor_turns_kept": kept_doctor_turns,
            "quality_passed": quality_passed,
            "quality_blocked": kept_cases - quality_passed,
            "manual_review_turns": manual_queue,
        },
        "first_ask_groups": {
            "count": len(group_sizes),
            "largest": group_sizes[0] if group_sizes else 0,
            "mean_size": round(sum(group_sizes) / max(len(group_sizes), 1), 3),
            "multi_member": sum(1 for size in group_sizes if size > 1),
            "prefix_merges": prefix_merges,
            "definition": "首问规范化整句（前 120 字）、起始 0/40/80 处的 40 字窗口、"
                          "前缀包含（≥20 字）、轮次序列指纹，任一相同即同组",
        },
        "duplicates": {
            "unique_fingerprints": len(first_seen),
            "cases_dropped_by_fingerprint": drop_hits.get("D07", 0),
            "duplicate_excess": cases - len(first_seen),
        },
        "measurements": {
            "cjk_ratio_histogram": histogram(cjk_ratios, CJK_BUCKETS),
            "privacy_pattern_cases": dict(privacy_patterns),
            "image_placeholder_forms": dict(placeholder_forms.most_common(20)),
            "image_exchange_cases": image_exchange_cases,
            "platform_blank_cases": platform_blank_cases,
            "department_topic_cases": dict(department_topics),
            "conclusion_not_last_turn": post_conclusion_cases,
            "post_conclusion_new_info": post_conclusion_new_info,
            "punct_repeat_turns_2": punct_two,
            "punct_repeat_turns_3plus": punct_three,
            "consecutive_repeated_doctor_turns": cross_turn_repeat,
            "invalid_named_entities": invalid_entities,
            "annotation_list_fields": dict(entity_fields),
            "leading_patient_chars_when_first_turn_is_doctor": {
                "cases": len(startup_leading),
                "median": median(startup_leading),
                "max": max(startup_leading) if startup_leading else 0,
                "histogram": histogram(startup_leading, LEAD_BUCKETS),
            },
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    census_path = args.output / "census.json"
    census_path.write_text(json.dumps(census, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"普查: {census_path}")

    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(render_markdown(census, source_names), encoding="utf-8")
    print(f"报告: {args.doc}")
    return 0


def table_header(source_names) -> str:
    head = "| 规则 | 说明 | 命中 | " + " | ".join(source_names) + " |"
    rule = "| --- | --- | ---: | " + " | ".join("---:" for _ in source_names) + " |"
    return head + "\n" + rule


def rule_rows(entries, source_names) -> str:
    lines = []
    for code, entry in entries.items():
        cells = " | ".join(str(entry["by_source"].get(src, 0)) for src in source_names)
        lines.append(f"| {code} | {entry['title']} | {entry['hits']:,} | {cells} |")
    return "\n".join(lines)


def examples_block(entries, codes) -> str:
    lines = []
    for code in codes:
        entry = entries.get(code) or {}
        if not entry.get("examples"):
            continue
        lines.append(f"\n**{code} {entry['title']}**（命中 {entry['hits']:,}）\n")
        lines.append("| 来源 | id | 行号 | detail | 原文片段 |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for item in entry["examples"]:
            text = item["text"].replace("|", "\\|")
            lines.append(f"| {item['source']} | {item['id']} | {item['line']} | "
                         f"{item.get('detail', '')} | {text} |")
    return "\n".join(lines)


def render_markdown(census: dict, source_names) -> str:
    totals = census["totals"]
    drops = census["drops"]
    repairs = census["repairs"]
    flags = census["flags"]
    decision = census["drop_decision"]
    projection = census["projection"]
    measurements = census["measurements"]

    parts: list = []
    parts.append("# 语料 v3 清洗前置普查（before）\n")
    parts.append(f"- 输入：{census['input']}"
                 + (f"（仅前 {census['limit']:,} 条，非全量）" if census["limit"] else "（全量）")
                 + f"，{census['input_bytes'] / 1e6:.1f} MB")
    parts.append(f"- 生成时间：{census['generated_at']}；耗时 {census['elapsed_seconds']}s")
    parts.append(f"- 普查器：{census['script']['path']}，"
                 f"sha256 {census['script']['sha256'][:16]}…")
    parts.append(f"- 示例抽样：每条规则最多 {EXAMPLES_PER_RULE} 个，蓄水池抽样，"
                 f"固定种子 {census['seed']}")
    parts.append("- 判定引擎与清洗器共用 scripts/corpus_v3_rules.py，"
                 "因此本表命中数就是清洗器将要执行的命中数。\n")

    parts.append("## 1. 语料规模\n")
    parts.append("| 来源 | 对话数 | 占比 | 轮数 | 平均轮数 | 医生轮 | 医生轮中位 |")
    parts.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for src in source_names:
        entry = census["by_source"][src]
        parts.append(f"| {src} | {entry['cases']:,} | {entry['cases_share']:.1%} | "
                     f"{entry['turns']:,} | {entry['mean_turns']} | "
                     f"{entry['doctor_turns']:,} | {entry['median_doctor_turns']} |")
    parts.append(f"| **合计** | {totals['cases']:,} | 100% | {totals['turns']:,} | "
                 f"{totals['mean_turns']} | {totals['doctor_turns']:,} | "
                 f"{totals['median_doctor_turns']} |\n")

    parts.append("### 对话轮数分布\n")
    parts.append("| 轮数区间 | 对话数 |\n| --- | ---: |")
    for label, value in totals["turns_histogram"].items():
        parts.append(f"| {label} | {value:,} |")
    parts.append("\n### 医生轮数分布\n")
    parts.append("| 医生轮 | 对话数 |\n| --- | ---: |")
    for label, value in totals["doctor_turns_histogram"].items():
        parts.append(f"| {label} | {value:,} |")
    parts.append("")

    parts.append("## 2. S1 丢弃规则（全量命中）\n")
    parts.append(table_header(source_names))
    parts.append(rule_rows(drops, source_names))
    parts.append("")
    parts.append(f"命中任一丢弃规则的对话 {decision['cases_dropped']:,} 条"
                 f"（占 {decision['cases_dropped_share']:.2%}），"
                 f"其中 {decision['cases_with_multiple_drop_rules']:,} 条同时命中多条；"
                 "清洗器按 D01→D09 顺序取第一条记账（多条命中全部记入普查，不重复删除）。")
    parts.append("\n按主规则分布：\n")
    parts.append("| 主规则 | 删除对话数 |\n| --- | ---: |")
    for code, value in sorted(decision["primary_rule"].items()):
        parts.append(f"| {code} {R.RULE_TITLES[code]} | {value:,} |")
    parts.append("")

    parts.append("## 3. S2 修复规则（存活对话内，按轮计数）\n")
    parts.append(table_header(source_names))
    parts.append(rule_rows(repairs, source_names))
    parts.append("")
    parts.append("| 规则 | 受影响对话数 |\n| --- | ---: |")
    for code, value in census["repair_cases"].items():
        parts.append(f"| {code} | {value:,} |")
    parts.append("")

    parts.append("## 4. S3 标记规则（存活对话内，按对话计数）\n")
    parts.append(table_header(source_names))
    parts.append(rule_rows(flags, source_names))
    parts.append("")
    parts.append(f"质量门：阻断标记集合 {census['blocking_flags']}；"
                 f"quality.passed=false 的对话 {projection['quality_blocked']:,} 条不进 SFT，"
                 f"passed=true {projection['quality_passed']:,} 条。"
                 "F08/F10 只标注不阻断。\n")

    parts.append("## 5. 清洗后规模预测（清洗器实测须与本表对齐）\n")
    parts.append("| 口径 | 清洗前 | 清洗后（预测） |\n| --- | ---: | ---: |")
    parts.append(f"| 对话数 | {totals['cases']:,} | {projection['cases_kept']:,} |")
    parts.append(f"| 轮数 | {totals['turns']:,} | {projection['turns_kept']:,} |")
    parts.append(f"| 医生轮数 | {totals['doctor_turns']:,} | {projection['doctor_turns_kept']:,} |")
    parts.append("")

    parts.append("## 6. 阈值与测量证据\n")
    parts.append("- 首轮角色分布 "
                 f"{json.dumps(totals['first_turn_role'], ensure_ascii=False)}："
                 "RL 的 prompt 取首轮文本，首轮不是患者发言时医生的话会进入患者可见上下文，"
                 "故 D08 直接把这类对话判为切分残段。")
    lead = measurements["leading_patient_chars_when_first_turn_is_doctor"]
    parts.append(f"- 首轮为医生的残段 {lead['cases']:,} 条，其首个患者发言实质字数中位 "
                 f"{lead['median']}、最大 {lead['max']}，分布 "
                 f"{json.dumps(lead['histogram'], ensure_ascii=False)}")
    parts.append("- 中文占比分布（D04 阈值 0.40）："
                 f"{json.dumps(measurements['cjk_ratio_histogram'], ensure_ascii=False)}")
    parts.append("- 隐私命中（按模式计对话数）："
                 f"{json.dumps(measurements['privacy_pattern_cases'], ensure_ascii=False)}")
    parts.append("- 图片占位形态：" + json.dumps(measurements["image_placeholder_forms"],
                                            ensure_ascii=False))
    parts.append(f"- 承接图片的对话（v2 的 IMAGE_EXCHANGE 口径）{measurements['image_exchange_cases']:,} 条")
    parts.append(f"- 连续标点：重复 2 次的轮 {measurements['punct_repeat_turns_2']:,}，"
                 f"3 次及以上 {measurements['punct_repeat_turns_3plus']:,}；"
                 "R07 只收敛 3 次及以上（2 次是说话人的语气，与 v2 同口径）。")
    parts.append("- 医生连续重复同一问题（跨轮）"
                 f"{measurements['consecutive_repeated_doctor_turns']:,} 处；"
                 "R03 只做同轮内去重，跨轮重复保留并记为 F08——删掉医生第二次追问等于改写事实。")
    parts.append("- 标注列表字段 "
                 f"{json.dumps(measurements['annotation_list_fields'], ensure_ascii=False)}，"
                 f"其中 name 为 undefined/空 的实体 "
                 f"{measurements['invalid_named_entities']:,} 个（R09 丢弃）。")
    parts.append("")

    parts.append("## 7. 切分与首问分组\n")
    groups = census["first_ask_groups"]
    parts.append(f"- 首问分组 {groups['count']:,} 组，最大 {groups['largest']:,} 条，"
                 f"平均 {groups['mean_size']} 条，多成员组 {groups['multi_member']:,} 个，"
                 f"前缀合并 {groups['prefix_merges']:,} 对")
    parts.append(f"- 分组定义：{groups['definition']}")
    parts.append("- 清洗器按组分配 split（组内取最小组键做哈希切分），因此同一组的对话不会跨 split。")
    parts.append("- 结局轮之后仍有轮次的对话 "
                 f"{measurements['conclusion_not_last_turn']:,} 条，"
                 f"其中后续患者轮带来新信息 {measurements['post_conclusion_new_info']:,} 条")
    parts.append("- presenting_problem 实为科室名的对话 "
                 f"{json.dumps(measurements['department_topic_cases'], ensure_ascii=False)}")
    parts.append(f"- 平台占位（语音/撤回/仅自己可见）对话 {measurements['platform_blank_cases']:,} 条\n")

    parts.append("## 8. 重复对话\n")
    dup = census["duplicates"]
    parts.append(f"- 唯一指纹 {dup['unique_fingerprints']:,}；"
                 f"命中已有指纹（D07 删除）{dup['cases_dropped_by_fingerprint']:,} 条；"
                 f"重复超额 {dup['duplicate_excess']:,}；"
                 "保留规则：按文件名排序后首个出现的对话胜出。\n")

    parts.append("## 9. 原文示例（每条 ≤80 字）\n")
    parts.append("### S1 丢弃\n")
    parts.append(examples_block(drops, list(drops)))
    parts.append("\n### S2 修复\n")
    parts.append(examples_block(repairs, list(repairs)))
    parts.append("\n### S3 标记\n")
    parts.append(examples_block(flags, list(flags)))
    parts.append("")

    parts.append("## 10. 已知限制\n")
    parts.append("- D08 以「首轮不是患者发言」为判据，隐含假设：切分残段的共同特征正是丢了开场。"
                 "若某源本就以医生开场，该源会被整源删除，需人工复核（本报告第 6 节给出该源规模）。")
    parts.append("- F07（多个主诉）用「患者出现 ≥2 次会话开场白」作为代理指标，漏判与误判都存在，"
                 "只标记不删除。")
    parts.append("- R03 不做跨轮去重（理由见第 6 节）。")
    parts.append("- 本报告是普查基线，示例仅为抽样展示；被删除或改写的每一条都逐条记录在 "
                 "data/processed_v3/_ledger.jsonl。\n")
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
