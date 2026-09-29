"""Build complete, case-level SFT episodes from IMCS-V2-MRG without LLM calls.

The source includes an annotated six-section medical report. Diagnosis and
recommendations are excluded from the doctor-visible profile. When the logged
dialogue lacks a diagnostic close, those source annotations provide a clearly
counted, deterministic final answer. No LLM calls are made.
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
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/raw/IMCS-V2-MRG.zip"
DEFAULT_OUTPUT = ROOT / "data/processed/sft_clean"
DEFAULT_AUDIT = ROOT / "data/audit/sft_cleaning"

REPORT_FIELDS = ("主诉", "现病史", "辅助检查", "既往史", "诊断", "建议")
PROFILE_FIELDS = ("主诉", "现病史", "辅助检查", "既往史")
UNKNOWN_HISTORY = re.compile(r"不详|未知|无记录|不清楚|不明确|资料缺如|病史不详")
EMPTY_FINAL_FIELD = re.compile(r"^(?:暂无?|无|未查|待查|不详|未知|无特殊|无异常)[。！!；;，, ]*$")
DIAGNOSIS_CUE = re.compile(
    r"诊断|考虑|倾向|怀疑|初步判断|初步考虑|不排除|可能是|可能为|符合"
)
ADVICE_CUE = re.compile(
    r"建议|需要|应当|应该|最好|可先|就诊|检查|复查|治疗|观察|随访|"
    r"注意|避免|休息|尽快|急诊|门诊|医院|监测"
)
NOISE_PATTERNS = (
    re.compile(r"(?:给个|给一下|留下|点个|帮忙).{0,8}(?:好评|五星|评价|满意)"),
    re.compile(r"(?:麻烦|请|劳烦|希望).{0,10}(?:给个|点个|留下|评价|好评|五星)"),
    re.compile(r"(?:好评|五星好评|服务评价|满意度评价).{0,8}(?:谢谢|感谢|哦|哟)?"),
    re.compile(
        r"(?:问诊|问答|提问|回复).{0,8}(?:次数|上限|限制)|"
        r"(?:最多|还剩|仅剩|剩余)\s*\d{1,2}\s*(?:次|轮)|"
        r"(?:系统|平台|本次).{0,8}(?:规定|限制|上限).{0,12}(?:次|轮|提问|问诊|问答)"
    ),
)
MEDIA_DEPENDENCE = re.compile(
    r"图片因隐私问题无法显示|图片(?:未显示|无法显示|看不清)|"
    r"(?:请|麻烦)?(?:上传|发一下|看一下)(?:图片|照片|报告|影像)|"
    r"无法查看图片|看图才能|根据图片"
)
GARBLED = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000e-\u001f]|</?[a-z][^>]*>|锟斤拷")
DIRECT_PII = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{15}(?:\d{2}[\dXx])?(?!\d)|"
    r"(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|https?://\S+"
)
REPEATED_PUNCTUATION = re.compile(r"[！？?!，。；、]{4,}")
NAME_INTRO = re.compile(
    r"(?:本人(?:的名字叫|名叫|叫|姓名)|我(?:的名字叫|名叫|叫)|"
    r"(?:患者|就诊人)姓名)\s*[:：]?\s*(?P<name>[\u4e00-\u9fff]{2,4})"
    r"(?=[，。；;：:、,.\s]|$)"
)
NAME_STOP = {"医生", "主任", "朋友", "妈妈", "爸爸", "老师", "不要", "帮我", "患者"}
PRIVACY_RULES = (
    ("url", re.compile(r"https?://[^\s<>，。；;]+", re.I), "[链接已隐去]"),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[邮箱已隐去]"),
    ("id_card", re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"), "[证件号已隐去]"),
    ("mobile", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9](?:[- ]?\d){9}(?!\d)"), "[电话已隐去]"),
    ("landline", re.compile(r"(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)"), "[电话已隐去]"),
    ("account", re.compile(r"(?:微信|QQ|扣扣|wx|vx)(?:号|号码)?\s*(?:是|为|[:：])?\s*[A-Za-z0-9_-]{5,20}", re.I), "[账号已隐去]"),
    ("record_number", re.compile(r"(?:病历号|住院号|就诊号|门诊号)\s*[:：]?\s*[A-Za-z0-9-]{5,24}"), "[病历编号已隐去]"),
    ("birth_date", re.compile(r"(?:出生日期|出生年月|生日)\s*[:：]?\s*\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?"), "[出生日期已隐去]"),
    ("address", re.compile(r"(?:家庭住址|居住地址|地址|住址)\s*[:：]\s*[^，。；;\n]{3,100}"), "[地址已隐去]"),
)
PRIVACY_COUNTS: Counter = Counter()
EXPLICIT_NAME_CASES = 0
TRAILING_CLOSER = re.compile(
    r"^(?:不客气|不用谢|再见|祝好|谢谢|感谢|有问题.{0,12}(?:再问|联系)|"
    r"希望对您有帮助)[！!。．.\s]*$"
)
ASSESSMENT_BEFORE = re.compile(
    r"(?:诊断(?:为|是)?|考虑(?:为|是)?|怀疑(?:为|是)?|倾向于?|可能(?:是|为)|"
    r"不排除|符合|判断(?:为|是)?|症状(?:是|考虑|为|还是)|"
    r"(?:目前|现在).{0,8}(?:是|属于)|就是|属于|患有)"
)
ASSESSMENT_AFTER = re.compile(r"(?:可能性大|考虑|提示|倾向)")
ROLE_MAP = {"医生": "assistant", "患者": "user"}


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def redact_case(case: dict) -> dict:
    global EXPLICIT_NAME_CASES
    names = set()
    for text in iter_strings(case):
        for match in NAME_INTRO.finditer(text):
            name = match.group("name")
            if name not in NAME_STOP and len(name) >= 2:
                names.add(name)
    if names:
        EXPLICIT_NAME_CASES += 1

    def visit(value):
        if isinstance(value, str):
            for name in sorted(names, key=lambda item: (-len(item), item)):
                if name in value:
                    value = value.replace(name, "[姓名已隐去]")
                    PRIVACY_COUNTS["name"] += 1
            for label, pattern, replacement in PRIVACY_RULES:
                value, count = pattern.subn(replacement, value)
                PRIVACY_COUNTS[label] += count
            return value
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return visit(case)


def informative_history(report: dict) -> bool:
    if not all(clean_text(report.get(field)) for field in REPORT_FIELDS):
        return False
    history = clean_text(report["既往史"])
    if UNKNOWN_HISTORY.search(history):
        return False
    if EMPTY_FINAL_FIELD.fullmatch(clean_text(report["诊断"])):
        return False
    if EMPTY_FINAL_FIELD.fullmatch(clean_text(report["建议"])):
        return False
    return True


def choose_report(reports: object) -> dict | None:
    if not isinstance(reports, list):
        return None
    candidates = [r for r in reports if isinstance(r, dict) and informative_history(r)]
    if not candidates:
        return None
    # Prefer the source report with the most patient-history detail.
    return max(
        candidates,
        key=lambda report: sum(len(clean_text(report.get(key))) for key in PROFILE_FIELDS),
    )


def make_groups(case: dict) -> list[dict] | None:
    groups: list[dict] = []
    for turn in case.get("dialogue") or []:
        if not isinstance(turn, dict):
            return None
        role = ROLE_MAP.get(turn.get("speaker"))
        text = clean_text(turn.get("sentence"))
        if role is None or not text:
            return None
        if groups and groups[-1]["role"] == role:
            groups[-1]["content"] += "\n" + text
        else:
            groups.append({"role": role, "content": text})
    if not groups:
        return None
    return groups


def remove_terminal_closers(text: str) -> str:
    lines = text.splitlines()
    while len(lines) > 1 and TRAILING_CLOSER.fullmatch(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def diagnosis_matches(final_text: str, diagnosis: str) -> bool:
    if ("?" in final_text or "？" in final_text
            or re.search(r"(?:吗|呢)[。.!！?？\s]*$", final_text)):
        return False
    diagnosis_prefix = re.compile(
        r"^(?:诊断为|考虑|初步诊断为?|可能为?|倾向于?|新生儿|婴儿|小儿|儿童|"
        r"急性|慢性|病毒性|细菌性)+"
    )
    for label in re.split(r"[、，,；;及或/]", diagnosis):
        label = diagnosis_prefix.sub("", label.strip())
        label = re.sub(r"[\W_]+", "", label)
        if len(label) < 2:
            continue
        for sentence in re.split(r"[。！？?!；;\n]", final_text):
            sentence = re.sub(r"[\W_]+", "", sentence)
            position = sentence.find(label)
            if position < 0:
                continue
            before = sentence[max(0, position - 24):position]
            after = sentence[position + len(label):position + len(label) + 10]
            if (ASSESSMENT_BEFORE.search(before) or ASSESSMENT_AFTER.search(after)):
                if not re.search(r"(?:如果|假如|若).{0,12}$", before):
                    return True
    return False


def is_clean_text(text: str) -> str | None:
    if not text:
        return "empty_text"
    if GARBLED.search(text) or REPEATED_PUNCTUATION.search(text):
        return "garbled_or_corrupt_text"
    if DIRECT_PII.search(text):
        return "direct_identifier_or_link"
    if MEDIA_DEPENDENCE.search(text):
        return "unavailable_image_or_attachment"
    return None


def inspect_case(case: dict) -> tuple[dict | None, str]:
    case = redact_case(case)
    report = choose_report(case.get("report"))
    if report is None:
        return None, "missing_or_incomplete_medical_record"

    groups = make_groups(case)
    if groups is None:
        return None, "invalid_dialogue"
    full_text = "\n".join(group["content"] for group in groups)
    if any(pattern.search(full_text) for pattern in NOISE_PATTERNS):
        return None, "rating_request_or_turn_limit"
    for text in [full_text, *(clean_text(report.get(k)) for k in PROFILE_FIELDS)]:
        reason = is_clean_text(text)
        if reason:
            return None, reason

    complete_rounds = sum(
        group["role"] == "user"
        and index + 1 < len(groups)
        and groups[index + 1]["role"] == "assistant"
        for index, group in enumerate(groups)
    )
    if complete_rounds < 3:
        return None, "fewer_than_three_complete_rounds"
    if len(groups) > 40 or sum(len(group["content"]) for group in groups) > 8000:
        return None, "overlong_dialogue"
    if any(len(group["content"]) > 2000 for group in groups):
        return None, "overlong_turn"

    observed_final = (
        remove_terminal_closers(groups[-1]["content"])
        if groups[-1]["role"] == "assistant"
        else ""
    )
    diagnosis = clean_text(report.get("诊断"))
    recommendation = clean_text(report.get("建议"))
    use_observed_final = bool(
        observed_final
        and len(observed_final) >= 25
        and len(observed_final) <= 1200
        and diagnosis_matches(observed_final, diagnosis)
        and ADVICE_CUE.search(observed_final)
    )
    if use_observed_final:
        final_text = observed_final
        final_origin = "observed_doctor_response"
        history_groups = groups[:-1]
    else:
        final_text = f"诊断：{diagnosis}\n建议：{recommendation}"
        final_origin = "structured_report_annotation"
        # Replace an incomplete final doctor utterance with the report-derived
        # clinical close, rather than placing two assistant turns back to back.
        history_groups = groups[:-1] if groups[-1]["role"] == "assistant" else groups
    if len(final_text) < 25 or len(final_text) > 1200:
        return None, "final_answer_length"
    if not diagnosis_matches(final_text, diagnosis):
        return None, "final_answer_without_diagnosis"
    if not ADVICE_CUE.search(final_text):
        return None, "final_answer_without_advice"

    target_record = f"{diagnosis}\n{recommendation}"
    for text in (diagnosis, recommendation):
        reason = is_clean_text(text)
        if reason:
            return None, reason
    if any(pattern.search(target_record) for pattern in NOISE_PATTERNS):
        return None, "rating_request_or_turn_limit"

    profile = "\n".join(
        f"{field}：{clean_text(report[field])}" for field in PROFILE_FIELDS
    )
    profile = "患者就诊病历（本次诊断和建议不放入输入）：\n" + profile
    visible_chars = len(profile) + len(final_text) + sum(
        len(group["content"]) for group in history_groups
    )
    if visible_chars > 3600:
        return None, "overlong_case_for_4096_context"

    messages: list[dict[str, str]] = []
    start = 0
    if groups[0]["role"] == "user":
        messages.append({
            "role": "user",
            "content": profile + "\n\n患者发言：\n" + groups[0]["content"],
        })
        start = 1
    else:
        messages.append({"role": "user", "content": profile})

    for group in history_groups[start:]:
        messages.append({"role": group["role"], "content": group["content"]})
    if not messages or messages[-1]["role"] != "user":
        return None, "invalid_sft_message_order"
    target = json.dumps(
        {"action": "FINAL", "message": final_text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages.append({"role": "assistant", "content": target})

    record = {
        "messages": messages,
        "profile": profile,
        "dialogue": history_groups + [{"role": "assistant", "content": final_text}],
        "final_origin": final_origin,
        "fingerprint": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return record, "accepted"


def load_split(archive: ZipFile, member: str):
    data = json.loads(archive.read(member).decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {member}")
    return data


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--spotcheck-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"IMCS-V2-MRG source archive not found: {args.source}. "
            "Download the official source archive locally and rerun."
        )

    accepted: dict[str, list[dict]] = {"train": [], "dev": []}
    rejected: dict[str, Counter] = {"train": Counter(), "dev": Counter()}
    members = {
        "train": "IMCS-V2-MRG/IMCS-V2_train.json",
        "dev": "IMCS-V2-MRG/IMCS-V2_dev.json",
    }
    with ZipFile(args.source) as archive:
        for split, member in members.items():
            for _, case in load_split(archive, member).items():
                candidate, reason = inspect_case(case)
                if candidate is None:
                    rejected[split][reason] += 1
                else:
                    candidate["split"] = split
                    accepted[split].append(candidate)

    train_fingerprints = {row["fingerprint"] for row in accepted["train"]}
    accepted["dev"] = [
        row for row in accepted["dev"] if row["fingerprint"] not in train_fingerprints
    ]
    accepted["train"].sort(key=lambda row: row["fingerprint"])
    accepted["dev"].sort(key=lambda row: row["fingerprint"])

    residual_privacy = Counter()
    for split_rows in accepted.values():
        for row in split_rows:
            for text in iter_strings(row["messages"]):
                if NAME_INTRO.search(text):
                    residual_privacy["explicit_name"] += 1
                if DIRECT_PII.search(text):
                    residual_privacy["direct_identifier_or_link"] += 1
                for label, pattern, _ in PRIVACY_RULES:
                    if pattern.search(text):
                        residual_privacy[label] += 1
    if residual_privacy:
        raise ValueError(f"Residual direct-identifier patterns found: {dict(residual_privacy)}")

    for split in ("train", "dev"):
        atomic_jsonl(args.output / f"{split}.openai.jsonl", [
            {"messages": row["messages"]} for row in accepted[split]
        ])

    reason_counts = {
        split: dict(sorted(counts.items())) for split, counts in rejected.items()
    }
    report = {
        "source": "IMCS-V2-MRG",
        "method": "rules_only_no_llm",
        "source_record_counts": {
            split: len(accepted[split]) + sum(rejected[split].values())
            for split in accepted
        },
        "criteria": {
            "complete_six_section_source_report": True,
            "past_history_must_not_be_unknown": True,
            "at_least_three_patient_doctor_rounds": True,
            "conversation_must_end_with_doctor": True,
            "final_answer_must_include_diagnosis_and_advice": True,
            "incomplete_observed_final_replaced_by_source_report_annotation": True,
            "reject_turn_limit_rating_noise_pii_and_missing_media": True,
            "diagnosis_and_recommendation_excluded_from_visible_profile": True,
            "test_split_used": False,
        },
        "accepted": {split: len(rows) for split, rows in accepted.items()},
        "final_answer_origins": {
            split: dict(Counter(row["final_origin"] for row in rows))
            for split, rows in accepted.items()
        },
        "privacy": {
            "profile": "rule_based_direct_identifier_scrub_v1",
            "explicit_name_cases_scrubbed": EXPLICIT_NAME_CASES,
            "redactions_by_type": dict(sorted(PRIVACY_COUNTS.items())),
            "residual_direct_identifier_patterns": {},
        },
        "rejected": reason_counts,
    }
    args.audit.mkdir(parents=True, exist_ok=True)
    (args.audit / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_rows = [
        (split, row)
        for split in ("train", "dev")
        for row in accepted[split]
    ]
    sample_n = min(max(args.spotcheck_size, 0), len(all_rows))
    sampled = random.Random(args.seed).sample(all_rows, sample_n)
    spotcheck_path = args.audit / f"spotcheck{sample_n}.csv"
    with spotcheck_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id", "split", "final_answer_origin", "patient_record",
                "dialogue", "final_answer",
            ),
        )
        writer.writeheader()
        for index, (split, row) in enumerate(sampled, 1):
            dialogue = []
            for message in row["dialogue"]:
                speaker = "患者" if message["role"] == "user" else "医生"
                dialogue.append(f"{speaker}：{message['content']}")
            target = json.loads(row["messages"][-1]["content"])["message"]
            writer.writerow({
                "sample_id": f"review_{index:02d}",
                "split": split,
                "final_answer_origin": row["final_origin"],
                "patient_record": row["profile"],
                "dialogue": "\n".join(dialogue),
                "final_answer": target,
            })

    print(
        f"Complete-case SFT data ready: train={len(accepted['train'])}, "
        f"dev={len(accepted['dev'])}; no LLM calls."
    )
    print(f"Files: {args.output / 'train.openai.jsonl'}")
    print(f"       {args.output / 'dev.openai.jsonl'}")
    print(f"Review: {spotcheck_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
