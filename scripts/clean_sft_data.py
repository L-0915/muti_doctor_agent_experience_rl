"""Build source-faithful SFT turns from real consultation dialogues.

Only observed patient/doctor turns are used. Post-hoc reports, diagnosis labels,
future patient turns, and inferred patient profiles are never inserted into the
doctor's prompt or used as an assistant target. The system prompt stays in
scripts/sys_prompt.py, outside the JSONL records.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/sft_clean"
DEFAULT_AUDIT = ROOT / "data/audit/sft_cleaning"
MIN_RAW_TURNS = 6  # Doctor-R1 filters KaMed SFT dialogues to >5 turns.
MIN_EXCHANGES = 3
MAX_TURNS = 30
MAX_DIALOGUE_CHARS = 4200
MAX_TURN_CHARS = 1200

QUESTION_MARKS = "?？"
ADVICE_CUES = (
    "建议", "需要", "最好", "可以", "应当", "应该", "就诊", "医院", "急诊",
    "治疗", "检查", "复查", "观察", "服用", "口服", "避免", "休息", "随访",
    "注意", "监测", "停药", "尽快",
)
DIAGNOSIS_CUES = (
    "诊断", "考虑", "怀疑", "倾向", "不排除", "可能是", "可能为", "提示", "符合",
    "属于", "判断为", "更像", "感染", "炎症", "疾病",
)
EMERGENCY_CUES = (
    "大量咯血", "咯血", "吐血", "呼吸困难", "胸痛", "意识不清", "昏厥", "抽搐",
    "一侧肢体无力", "言语不清", "大量出血", "剧烈腹痛",
)
NOISE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"(?:给个|给一下|留下|点个|帮忙).{0,12}(?:好评|五星|评价|满意)",
    r"(?:提高好评率|满意度评价|五星好评|好评率|麻烦.{0,12}评价|(?:请|麻烦|求|给).{0,12}(?:好评|满意|送心意)|(?:好评|满意).{0,12}送心意)",
    r"(?:系统|平台|本次).{0,12}(?:规定|限制|上限).{0,16}(?:轮|次|问诊|提问|问答)",
    r"(?:最多|还剩|剩余)\s*\d{1,2}\s*(?:次|轮)",
    r"自动回复|现在比较忙.{0,20}(?:留言|稍后回复)|刚才在忙|这里听不了语音|无法听清语音|在线等了一段时间.{0,20}先离开",
    r"有病.{0,8}为什么.{0,8}(?:喝酒|抽烟)|你有病.{0,8}(?:喝酒|抽烟)",
    r"(?:针对|对于)本次问诊.{0,50}总结建议|医生更新了总结建议",
    r"关注(?:我的|本人|医生|诊所|主页)|点击.{0,12}头像|搜索我的姓名|扫码.{0,12}(?:咨询|关注)",
    r"(?:加|添加).{0,8}(?:微信|QQ)|(?:微信|QQ)号.{0,8}(?:联系|咨询)",
))
MISSING_MEDIA = re.compile(
    r"图片因隐私问题无法显示|图片无法显示|图片未显示|无法查看图片|"
    r"患者曾上传图片，但本数据没有图像内容"
)
GARBLED = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]|</?[a-z][^>]*>|<\|[^|]{1,40}\|>|[?？!！。．]{3,}")
SUMMARY_TEMPLATE = re.compile(r"(?:针对|对于)本次问诊.{0,50}总结建议|医生更新了总结建议")
MALFORMED_DEMOGRAPHIC = re.compile(r"[（(]\s*[，,]\s*[）)]")
DIRECT_PII = (
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9](?:[- ]?\d){9}(?!\d)"), "[电话已隐去]"),
    (re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"), "[证件号已隐去]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[邮箱已隐去]"),
    (re.compile(r"https?://[^\s<>，。；;]+", re.I), "[链接已隐去]"),
    (re.compile(r"(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)"), "[电话已隐去]"),
    (re.compile(r"(?:微信|QQ|wx|vx)(?:号|账号)?\s*[:：]?\s*[A-Za-z0-9_-]{5,20}", re.I), "[账号已隐去]"),
    (re.compile(r"(?:病历号|住院号|就诊号|门诊号)\s*[:：]?\s*[A-Za-z0-9-]{5,24}"), "[病历编号已隐去]"),
    (re.compile(r"(?:家庭住址|居住地址|地址|住址)\s*[:：]\s*[^，。；;\n]{3,100}"), "[地址已隐去]"),
)
RESIDUAL_PII = tuple(pattern for pattern, _ in DIRECT_PII)
SOCIAL_ONLY = (
    "谢谢", "感谢", "不客气", "不用谢", "好的", "好吧", "嗯", "嗯嗯", "知道了",
    "明白了", "祝您", "祝你", "早日康复", "再见", "拜拜", "没有其他问题", "有问题再咨询",
)
SAFE_TYPO_FIXES = (
    ("辈子盖的很少", "被子盖得很少"),
    ("不号买药", "不好买药"),
    ("弄一下试试卡", "弄一下试试看"),
    ("早歺", "早餐"), ("午歺", "午餐"), ("晚歺", "晚餐"),
    ("耳隆", "耳聋"),
)
TYPO_COUNTS: Counter = Counter()


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")
    for wrong, right in SAFE_TYPO_FIXES:
        value, count = value.replace(wrong, right), value.count(wrong)
        TYPO_COUNTS[wrong] += count
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def privacy_names(case_id: str, rules: dict) -> list[str]:
    return rules.get("names_by_original_case", {}).get(case_id, [])


def redact(text: str, names: list[str]) -> str:
    for name in sorted(names, key=lambda value: (-len(value), value)):
        if name:
            text = text.replace(name, "[姓名已隐去]")
    for pattern, replacement in DIRECT_PII:
        text = pattern.sub(replacement, text)
    return clean_text(text)


def source_groups(case: dict, names: list[str]) -> list[dict] | None:
    groups: list[dict] = []
    for turn in case.get("turns") or []:
        role = turn.get("role")
        text = turn.get("text_zh")
        if role not in {"patient", "doctor"} or not isinstance(text, str):
            return None
        text = redact(text, names)
        if not text:
            return None
        if groups and groups[-1]["role"] == role:
            groups[-1]["content"] += "\n" + text
        else:
            groups.append({"role": role, "content": text})
    return groups or None


def completed_exchanges(groups: list[dict]) -> int:
    return sum(groups[i]["role"] == "patient" and groups[i + 1]["role"] == "doctor"
               for i in range(len(groups) - 1))


def case_rejection(case: dict, groups: list[dict] | None) -> str | None:
    if case.get("source") == "meddialog":
        return "excluded_medialog_per_user"
    if case.get("split") not in {"train", "dev"}:
        return "not_train_or_dev"
    if case.get("synthetic"):
        return "synthetic_not_in_real_dialogue_sft"
    if case.get("source_dialogue_truncated"):
        return "source_dialogue_truncated"
    if not groups:
        return "malformed_dialogue"
    if len(case.get("turns") or []) < MIN_RAW_TURNS:
        return "fewer_than_six_source_turns"
    if len(groups) > MAX_TURNS:
        return "overlong_dialogue"
    if groups[0]["role"] != "patient" or groups[-1]["role"] != "doctor":
        return "incomplete_dialogue_boundary"
    if completed_exchanges(groups) < MIN_EXCHANGES:
        return "fewer_than_three_completed_exchanges"
    if not final_is_clinical(groups[-1]["content"]):
        return "missing_observed_terminal_diagnosis_and_advice"
    if any(SUMMARY_TEMPLATE.search(group["content"]) for group in groups):
        return "posthoc_summary_template"
    if sum(len(g["content"]) for g in groups) > MAX_DIALOGUE_CHARS:
        return "overlong_dialogue_chars"
    for group in groups:
        if len(group["content"]) > MAX_TURN_CHARS:
            return "overlong_turn"
        if MALFORMED_DEMOGRAPHIC.search(group["content"]):
            return "malformed_demographic_fragment"
        if GARBLED.search(group["content"]):
            return "garbled_or_corrupt_text"
        if MISSING_MEDIA.search(group["content"]):
            return "unavailable_media_content"
        if group["role"] == "doctor" and any(p.search(group["content"]) for p in NOISE_PATTERNS):
            return "platform_rating_promo_or_limit_noise"
        if any(pattern.search(group["content"]) for pattern in RESIDUAL_PII):
            return "residual_direct_identifier"
    return None


def single_question(text: str) -> bool:
    text = text.strip()
    if not 5 <= len(text) <= 150 or "\n" in text:
        return False
    if sum(text.count(mark) for mark in QUESTION_MARKS) != 1:
        return False
    if text[-1] not in QUESTION_MARKS:
        return False
    if any(cue in text for cue in ADVICE_CUES):
        return False
    if re.search(r"(?:有什么|还有什么|还有其他|有没有其他).{0,8}(?:不适|症状|问题).{0,3}[吗呢？?]?$", text):
        return False
    if re.search(r"[，,].{0,16}(?:有没有|是否|多大年龄|多长时间|几天了|什么症状|哪里不舒服)", text):
        return False
    symptom_cues = re.findall(
        r"反酸|烧心|打嗝|嗳气|恶心|想吐|呕吐|腹泻|便秘|发热|咳嗽|咳痰|胸闷|气短|头晕|尿频|尿急|疼痛",
        text,
    )
    if len(symptom_cues) >= 3:
        return False
    return True


def emergency_in_history(groups: list[dict]) -> bool:
    return any(group["role"] == "patient" and any(cue in group["content"] for cue in EMERGENCY_CUES)
               for group in groups)


def final_is_clinical(text: str) -> bool:
    text = text.strip()
    if not 20 <= len(text) <= 500 or any(mark in text for mark in QUESTION_MARKS):
        return False
    if re.search(r"有没有|是否|怎么回事|什么原因|多大年龄|多长时间|几天了|如何处理|怎么办|\\w吗(?:$|[，。])", text):
        return False
    disease = r"(?:[\u4e00-\u9fff]{1,10}(?:病|炎|症|感染|结石|肿瘤|综合征|癌|过敏)|高血压|糖尿病|冠心病)"
    diagnosis_trigger = r"(?:诊断(?:为|是)?|考虑(?:是|为|患有)?|怀疑|倾向(?:于)?|不排除|可能(?:是|为)|符合(?:于)?|提示)"
    has_diagnosis = bool(
        re.search(diagnosis_trigger + r".{0,12}" + disease, text)
        or re.search(disease + r".{0,8}" + diagnosis_trigger, text)
    )
    has_advice = bool(re.search(
        r"建议|需要|应当|应该|最好|就诊|医院|急诊|治疗|检查|复查|观察|服用|口服|避免|休息|随访|注意|监测|尽快|可(?:以)?(?:先)?(?:到|去|做|服|口服|观察|复查)",
        text,
    ))
    emergency_referral = any(cue in text for cue in ("立即就医", "急诊", "拨打120", "呼叫急救"))
    return (has_diagnosis and has_advice) or emergency_referral


def chat_history(groups: list[dict]) -> list[dict]:
    role_map = {"patient": "user", "doctor": "assistant"}
    return [{"role": role_map[group["role"]], "content": group["content"]}
            for group in groups]


def example(groups: list[dict], target_index: int, action: str) -> dict:
    target = groups[target_index]["content"].strip()
    answer = json.dumps({"action": action, "message": target},
                        ensure_ascii=False, separators=(",", ":"))
    messages = chat_history(groups[:target_index])
    messages.append({"role": "assistant", "content": answer})
    if not messages or messages[0]["role"] != "user" or messages[-2]["role"] != "user":
        raise ValueError("SFT example must begin with patient and target a doctor turn")
    return {"messages": messages}


def make_examples(case: dict, groups: list[dict]) -> list[dict]:
    candidates = []
    patient_rounds = 0
    for index, group in enumerate(groups):
        if group["role"] == "patient":
            patient_rounds += 1
            continue
        if index == 0 or groups[index - 1]["role"] != "patient":
            continue
        if (index + 1 < len(groups) and groups[index + 1]["role"] == "patient"
                and single_question(group["content"]) and patient_rounds >= 2):
            if not (emergency_in_history(groups[:index]) and
                    not any(cue in group["content"] for cue in ("急诊", "立即就医", "拨打120", "呼叫急救"))):
                candidates.append((index, "ASK"))
        if index == len(groups) - 1 and final_is_clinical(group["content"]):
            candidates.append((index, "FINAL"))

    # Keep no more than one example per action per source case; choose a later
    # answered question to teach decisions from multi-turn history.
    selected = []
    ask = [item for item in candidates if item[1] == "ASK"]
    final = [item for item in candidates if item[1] == "FINAL"]
    if ask:
        selected.append(max(ask, key=lambda item: item[0]))
    if final:
        selected.extend(final[:1])
    return [example(groups, index, action) for index, action in selected]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def example_fingerprint(row: dict) -> str:
    payload = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--spotcheck-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing normalized source dialogue file: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")

    rules_path = ROOT / "data/local/public_release_rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}
    accepted: dict[str, list[tuple[str, dict]]] = {"train": [], "dev": []}
    rejects: dict[str, Counter] = {"train": Counter(), "dev": Counter()}
    source_counts: dict[str, Counter] = {"train": Counter(), "dev": Counter()}
    seen_fingerprints: dict[str, set[str]] = {"train": set(), "dev": set()}
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            split = case.get("split")
            if split not in accepted:
                continue
            source_counts[split]["source_cases"] += 1
            case_id = str(case.get("case_id", ""))
            groups = source_groups(case, privacy_names(case_id, rules))
            reason = case_rejection(case, groups)
            if reason:
                rejects[split][reason] += 1
                continue
            fingerprint = hashlib.sha256("\n".join(
                f"{g['role']}:{g['content']}" for g in groups
            ).encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints[split]:
                rejects[split]["duplicate_dialogue"] += 1
                continue
            seen_fingerprints[split].add(fingerprint)
            samples = make_examples(case, groups)
            if not samples:
                rejects[split]["no_high_precision_observed_action"] += 1
                continue
            source_counts[split]["accepted_cases"] += 1
            source_counts[split]["ask_examples"] += sum(
                json.loads(row["messages"][-1]["content"])["action"] == "ASK" for row in samples)
            source_counts[split]["final_examples"] += sum(
                json.loads(row["messages"][-1]["content"])["action"] == "FINAL" for row in samples)
            accepted[split].extend((case_id, row) for row in samples)

    # Different source cases can still produce the exact same state/action
    # example. Deduplicate within each split, then prevent train/dev overlap.
    seen_examples: dict[str, set[str]] = {"train": set(), "dev": set()}
    train_unique = []
    for case_id, row in accepted["train"]:
        fingerprint = example_fingerprint(row)
        if fingerprint in seen_examples["train"]:
            rejects["train"]["duplicate_sft_example_within_split"] += 1
            continue
        seen_examples["train"].add(fingerprint)
        train_unique.append((case_id, row))
    accepted["train"] = train_unique

    unique_dev = []
    for case_id, row in accepted["dev"]:
        fingerprint = example_fingerprint(row)
        if fingerprint in seen_examples["train"]:
            rejects["dev"]["duplicate_sft_example_cross_split"] += 1
        elif fingerprint in seen_examples["dev"]:
            rejects["dev"]["duplicate_sft_example_within_split"] += 1
        else:
            seen_examples["dev"].add(fingerprint)
            unique_dev.append((case_id, row))
    accepted["dev"] = unique_dev

    for split in ("train", "dev"):
        source_counts[split]["accepted_cases"] = len({case_id for case_id, _ in accepted[split]})
        source_counts[split]["ask_examples"] = sum(
            json.loads(row["messages"][-1]["content"])["action"] == "ASK"
            for _, row in accepted[split]
        )
        source_counts[split]["final_examples"] = sum(
            json.loads(row["messages"][-1]["content"])["action"] == "FINAL"
            for _, row in accepted[split]
        )

    # Do not expose source names, case IDs, or system prompts in the published records.
    public_rows = {}
    for split in ("train", "dev"):
        accepted[split].sort(key=lambda item: hashlib.sha256(
            ("heapo-sft-v3:" + item[0] + ":" + item[1]["messages"][-1]["content"]).encode("utf-8")
        ).hexdigest())
        public_rows[split] = [row for _, row in accepted[split]]
        atomic_jsonl(args.output / f"{split}.openai.jsonl", public_rows[split])

    # Output audit: no embedded system message, profile template, report label,
    # raw source IDs, or residual direct identifiers.
    residual = Counter()
    case_sets = {split: {case_id for case_id, _ in accepted[split]} for split in accepted}
    for split, rows in public_rows.items():
        for row in rows:
            messages = row.get("messages") or []
            if not messages or any(message.get("role") == "system" for message in messages):
                raise ValueError("System prompt must stay outside data records")
            if not messages[-1]["content"].startswith('{"action":'):
                raise ValueError("Target is not a structured action response")
            for message in messages:
                text = message["content"]
                if ("患者就诊病历" in text or "既往病历" in text or "已公开患者档案" in text
                        or SUMMARY_TEMPLATE.search(text)):
                    residual["fabricated_profile_template"] += 1
                if any(pattern.search(text) for pattern in RESIDUAL_PII):
                    residual["direct_identifier"] += 1
    if residual:
        raise ValueError(f"Output quality gate failed: {dict(residual)}")
    if case_sets["train"] & case_sets["dev"]:
        raise ValueError("Case leakage between train and dev")
    if seen_fingerprints["train"] & seen_fingerprints["dev"]:
        raise ValueError("Duplicate dialogue leakage between train and dev")

    all_rows = [(split, row) for split in ("train", "dev") for row in public_rows[split]]
    sample_n = min(max(args.spotcheck_size, 0), len(all_rows))
    randomizer = random.Random(args.seed)
    by_action = {action: [(split, row) for split, row in all_rows
                          if json.loads(row["messages"][-1]["content"])["action"] == action]
                 for action in ("ASK", "FINAL")}
    audit_counts = {"ASK": min(sample_n // 2, len(by_action["ASK"])),
                    "FINAL": min(sample_n - sample_n // 2, len(by_action["FINAL"]))}
    if sum(audit_counts.values()) < sample_n:
        remaining = [item for item in all_rows
                     if item not in by_action["ASK"][:audit_counts["ASK"]]
                     and item not in by_action["FINAL"][:audit_counts["FINAL"]]]
        fill = randomizer.sample(remaining, min(sample_n - sum(audit_counts.values()), len(remaining)))
    else:
        fill = []
    sample = []
    for action in ("ASK", "FINAL"):
        sample.extend(randomizer.sample(by_action[action], audit_counts[action]))
    sample.extend(fill)
    args.audit.mkdir(parents=True, exist_ok=True)
    spotcheck = args.audit / f"spotcheck{sample_n}.csv"
    with spotcheck.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "split", "action", "dialogue"))
        writer.writeheader()
        for index, (split, row) in enumerate(sample, 1):
            messages = row["messages"]
            dialogue = "\n".join(
                ("患者：" if item["role"] == "user" else "医生：") + item["content"]
                for item in messages[:-1]
            )
            target = json.loads(messages[-1]["content"])
            writer.writerow({"sample_id": f"review_{index:02d}", "split": split,
                             "action": target["action"],
                             "dialogue": dialogue + "\n医生：" + target["message"]})

    report = {
        "source": "data/processed/cases.jsonl",
        "method": "rules_only_observed_dialogue_turns",
        "api_calls": 0,
        "patient_profile": "not_available_in_source_and_not_injected",
        "posthoc_report_or_diagnosis_label_used": False,
        "excluded_sources": ["meddialog"],
        "split_case_leakage": False,
        "system_prompt_embedded": False,
        "criteria": {
            "min_source_turns": MIN_RAW_TURNS,
            "min_completed_exchanges": MIN_EXCHANGES,
        "complete_dialogue_and_nontruncated": True,
        "requires_observed_terminal_diagnosis_and_advice": True,
            "exclude_rating_platform_limits_promo_missing_media_and_corruption": True,
            "ask_target_is_one_observed_question": True,
            "final_target_is_observed_clinical_close": True,
            "direct_identifiers_redacted_and_rescanned": True,
            "typos_autocorrected": "only a small explicit safe-replacement list",
        },
        "known_typo_fixes": dict(TYPO_COUNTS),
        "accepted": {split: dict(source_counts[split]) for split in accepted},
        "rejected": {split: dict(sorted(counts.items())) for split, counts in rejects.items()},
        "files": {split: {"path": str(args.output / f"{split}.openai.jsonl"),
                          "examples": len(rows), "sha256": sha256(args.output / f"{split}.openai.jsonl")}
                  for split, rows in public_rows.items()},
        "spotcheck": {"path": str(spotcheck), "examples": sample_n},
    }
    (args.audit / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(public_rows["train"]), "dev": len(public_rows["dev"]),
                      "train_actions": dict(Counter(json.loads(r["messages"][-1]["content"])["action"]
                                                     for r in public_rows["train"])),
                      "dev_actions": dict(Counter(json.loads(r["messages"][-1]["content"])["action"]
                                                   for r in public_rows["dev"])),
                      "spotcheck": str(spotcheck)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
