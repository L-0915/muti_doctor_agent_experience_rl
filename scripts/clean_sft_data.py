"""Conservatively select clean cold-start SFT conversations with zero API calls.

Rules remove obvious dialogue artifacts and unsuitable targets. They do not
verify diagnoses or certify clinical quality. Source JSONL is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/train.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/sft_clean"
DEFAULT_AUDIT = ROOT / "data/audit/sft_cleaning"
RULE_VERSION = "sft-clean-v5"
ACTION_KEYS = {"action", "message"}
SYSTEM_PROFILE_MARKER = "\n\n已公开患者档案（仅来自当前可见患者发言）："
HAN = re.compile(r"[\u3400-\u9fff]")
DIRECT_PII = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[\dXx](?!\d)|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|"
    r"https?://\S+|(?:微信|qq|QQ)[:：号\s]*[A-Za-z0-9_-]{5,}"
)
GARBLED = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000e-\u001f]|<\/?(?:div|span|br|p)\b|\\u[0-9a-fA-F]{4}")
TARGET_TYPO = re.compile(r"[，,]{2,}|[。\.]{2,}|[？?]{2,}|有不部分|慢长|即刻住院胸|复发不会比以前严重")
RATING = re.compile(r"好评|五星|满意.{0,8}评价|评价.{0,8}满意|给.{0,5}评价|打分|点赞|收藏|关注.{0,8}(?:我|医生)")
PROMOTION = re.compile(r"私人医生|扫码|二维码|微信公众号|关注.{0,8}(?:头像|微博|公众号)|点击头像|优惠券|复诊卡|加微信|添加微信")
PLATFORM = re.compile(
    r"\[自动回复\]|【自动回复】|医生留言|医生给您发来一个提醒|"
    r"针对本次问诊.{0,8}(?:总结|建议)|病情资料我已详细阅读|"
    r"报到来源|患病多久[:：]|希望获得的帮助[:：]|病情变化情况[:：]|"
    r"是否开了检查[:：]|是否拿到检查结果[:：]|开药需求[:：]|"
    r"您好.{0,30}长时间没反馈|有问题请留言|真情寄语|微信上传"
)
MEDIA_DEPENDENCE = re.compile(
    r"图片因隐私问题无法显示|患者曾上传图片|看图|上图|这张图|图片上|照片上|"
    r"图片|照片|相片|片子|影像资料|上传.{0,6}(?:图|照片)|报告单|化验单|"
    r"(?:拍|发).{0,6}(?:照|图)|(?:发|上传).{0,12}(?:报告|病历|影像|ct|CT|b超|B超)|"
    r"(?:初步|乍).{0,4}看(?:来|着)|从外观|"
    r"语音因隐私问题无法显示|音频因隐私问题无法显示|视频因隐私问题无法显示"
)
COURTESY_ONLY = re.compile(
    r"^(?:嗯+|哦+|好(?:的|吧|了)?|知道了|明白了|收到|谢谢(?:你|您|医生|大夫)?|"
    r"多谢(?:你|您|医生|大夫)?|辛苦了|不客气|再见)[!！。,.，~～\s]*$"
)
ACK_ONLY = re.compile(r"^(?:是(?:的)?|对(?:的)?|没错|不是|没有|有|嗯+|好(?:的|吧)?)[!！。,.，~～\s]*$")
LOW_INFO_ASK = re.compile(
    r"^(?:您好|你好|请问|医生)?[,，:\s]*(?:还有吗|什么症状|有其他症状吗|"
    r"还有其他症状吗|还有别的不舒服吗|有其他不舒服吗|还有什么问题吗|"
    r"想咨询什么问题呢|有什么问题呢|怎么了)[?？]$"
)
GENERIC_SYMPTOM_ASK = re.compile(r"(?:还|另|其).{0,5}(?:什么|其他|别的).{0,5}(?:症状|不舒服)|(?:有|出现).{0,3}(?:什么|其他).{0,3}症状")
QUESTION_CUE = re.compile(r"有没有|是否|有无|什么|怎么|哪里|哪|多久|多长|多少|几|吗|呢|用过|做过|会不会|能否")
ASK_ADVICE = re.compile(
    r"建议|应该|可以吃|可以用|需要服用|先吃|先用|口服|治疗|诊断为|考虑是|"
    r"不用担心|没有问题|多喝水|注意休息|去医院|到医院|药店"
)
FINAL_FALLBACK = re.compile(r"不客气|祝.{0,12}(?:健康|康复|愉快)|随时联系|有问题再问|我的解答|谢谢.{0,8}信任")
FINAL_CONTENT = re.compile(r"建议|需要|应当|注意|观察|复查|检查|就医|医院|门诊|急诊|如果|若|可能|风险|避免")
FINAL_ACTION = re.compile(r"建议|需要|应当|注意|观察|复查|检查|就医|急诊|及时|尽快|请到|请去")
FINAL_CAUTION = re.compile(r"可能|考虑|建议|如果|若|需要|应当|注意|风险|暂时|目前|请|尽快")
FINAL_NEXT_STEP = re.compile(r"复查|检查|就医|医院|门诊|急诊|观察|留意|监测|随访")
INSTITUTIONAL_REPLY = re.compile(r"我们医院|我院|本院|我科|门诊在周|地址是|费用|预约|挂.{0,4}号|专家号|我现在在.{0,8}下乡|身份证|医保卡|就诊卡|特需门诊|我每周.{0,12}门诊|周[一二三四五六日].{0,8}门诊|(?:上海|北京|浙江|江苏|中山|瑞金).{0,8}医院")
TREATMENT_CERTAINTY = re.compile(r"(?:可以只|只要|完全可以|不需要|没有必要|不用).{0,12}(?:疫苗|检查|手术|治疗|复查|住院|就医)")
PROCEDURE_DIRECTIVE = re.compile(r"(?:建议|需要|可以|应当).{0,8}(?:手术|移植|化疗|放疗|注射|输液)|最好拔|建议拔|可以一起吃|没有任何副作用|一刀永逸|放粒子|伽马刀|粒子植入")
UNPROFESSIONAL_REPLY = re.compile(r"呵呵|哈哈|亲[，,\s]|我个头|快给|科普文章|来我这|已经给你说过|我都说了|懒得解释|自己思考|首先.{0,6}我说了|没有医生不")
DIRECT_DRUG_VERB = re.compile(
    r"(?:吃|服|用|涂抹|注射|打).{0,12}(?:药|片|胶囊|颗粒|丸|乳膏|软膏|针|"
    r"霉素|唑|林|汀|丁啉|沙坦|普利)|(?:奥美拉唑|吗丁啉|华法林|阿司匹林|激素)"
)
UNSUPPORTED_DIAGNOSIS = re.compile(r"(?:就是|属于|确诊为|我(?:的)?诊断是).{0,8}(?:炎|病|癣|癌|结核|感染|障碍)")
DIRECT_DRUG_ADVICE = re.compile(
    r"(?:口服|服用|吃|使用|应用|注射|停用|停药|换药|减量|加量).{0,12}"
    r"(?:药|片|胶囊|颗粒|糖浆|滴剂|抗生素)|"
    r"(?:每次|每日|每天|一天|每晚|每早).{0,18}(?:片|粒|次|毫克|mg|ml|毫升)|"
    r"\d+(?:\.\d+)?\s*(?:毫克|mg|ml|毫升|片|粒).{0,18}(?:服|吃|用|次)"
)
ANY_MEDICATION_IN_FINAL = re.compile(r"药|服用|口服|抗生素|激素|胶囊|颗粒|软膏|乳膏|注射|输液|打针")
OVERCONFIDENT = re.compile(r"绝对|肯定没|一定没|肯定是|就是.{0,12}(?:癌|炎|病)|不用检查|完全不用担心|不可能有事|不可能查出|可以基本确诊|100%|百分之百")
# These are high-risk review cues, not a diagnosis. Matched cases are kept only
# when the target is a FINAL action with explicit urgent-care instructions.
RED_FLAG = re.compile(
    r"咳血|咯血|痰中带血|血痰|吐血|呕血|大量出血|血流不止|意识模糊|神志不清|"
    r"突然意识|呼吸困难|喘不过气|胸痛|胸口剧痛|昏厥|昏迷|抽搐|"
    r"自杀|轻生|想死|肢体突然无力|突然说不出话"
)
NEGATION_BEFORE = re.compile(r"(?:没有|没|无|否认|未|不|排除|不是|并无)$")
URGENT_CARE = re.compile(r"(?:立即|立刻|马上|尽快|及时).{0,8}(?:急诊|就医|去医院)|急诊|拨打120|叫救护车")
DURATION_MENTION = re.compile(r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半])\s*(?:个)?(?:小时|分钟|天|日|周|星期|月|年)")
GENERIC_DURATION_ASK = re.compile(r"(?:这种情况|这个情况|这情况|症状|不舒服)?.{0,6}(?:多久|多长时间|什么时候开始|几天了|几个月了)")
AGE_MENTION = re.compile(r"(?:\d{1,3}|[一二两三四五六七八九十]+)\s*(?:岁|个月|月龄)")
AGE_ASK = re.compile(r"(?:几岁|多大|年龄|几个月|才.{0,3}(?:岁|个月))")
ASSERTIVE_ASK_PREFIX = re.compile(r"(?:初步来看|看起来|可以确定|应该是|就是|会的|是的|属于).{2,40}[，,]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def action_from(message: dict) -> dict | None:
    try:
        target = json.loads(message["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if (not isinstance(target, dict) or set(target) != ACTION_KEYS
            or target.get("action") not in {"ASK", "FINAL"}
            or not isinstance(target.get("message"), str)):
        return None
    return target


def red_flag_present(patient_text: str) -> bool:
    for match in RED_FLAG.finditer(patient_text):
        preceding = patient_text[max(0, match.start() - 5):match.start()]
        if not NEGATION_BEFORE.search(preceding):
            return True
    return False


def screen(record: dict) -> tuple[dict | None, list[str]]:
    reasons = []
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
        return None, ["invalid_structure"]
    if (messages[0].get("role") != "system"
            or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                   or m["role"] != ("user" if index % 2 else "assistant")
                   or not isinstance(m["content"], str) or not m["content"].strip()
                   for index, m in enumerate(messages[1:], 1))):
        return None, ["invalid_structure"]
    target = action_from(messages[-1])
    if target is None:
        return None, ["invalid_target"]
    doctor_text = normalize(target["message"])
    if not doctor_text or len(doctor_text) > 350:
        return None, ["target_length"]
    patient = [normalize(m["content"]) for m in messages if m["role"] == "user"]
    history_doctor = [normalize(m["content"]) for m in messages[1:-1]
                      if m["role"] == "assistant"]
    all_visible = [messages[0]["content"]] + patient + history_doctor + [doctor_text]
    if any(DIRECT_PII.search(x) for x in all_visible):
        reasons.append("direct_identifier_or_link")
    if any(GARBLED.search(x) for x in all_visible):
        reasons.append("garbled_text_or_markup")
    if TARGET_TYPO.search(doctor_text):
        reasons.append("obvious_target_typo_or_repeated_punctuation")
    if any(RATING.search(x) or PROMOTION.search(x) for x in history_doctor + [doctor_text]):
        reasons.append("rating_or_promotion")
    if any(PLATFORM.search(x) for x in all_visible):
        reasons.append("platform_template")
    if any(MEDIA_DEPENDENCE.search(x) for x in all_visible):
        reasons.append("unavailable_media")
    if len(messages) > 13 or sum(len(x) for x in all_visible) > 1400:
        reasons.append("overlong_context")
    if any(len(x) > 500 for x in patient) or any(len(x) > 350 for x in history_doctor):
        reasons.append("overlong_turn")
    if len(HAN.findall(patient[-1])) < 2 or len(HAN.findall(doctor_text)) < 4:
        reasons.append("low_information_turn")
    if COURTESY_ONLY.fullmatch(patient[-1]) or ACK_ONLY.fullmatch(patient[-1]):
        reasons.append("courtesy_only_last_patient_turn")
    if (re.fullmatch(r".{0,8}(?:医生|大夫|主任|教授)[：:]?", patient[-1])
            or (len(patient[-1]) < 35 and re.search(r"(?:晚上好|您好|你好).{0,12}(?:打扰|请问)", patient[-1]))):
        reasons.append("greeting_only_last_patient_turn")
    if red_flag_present("\n".join(patient)) and not (target["action"] == "FINAL" and URGENT_CARE.search(doctor_text)):
        reasons.append("red_flag_without_urgent_final")
    if target["action"] == "ASK":
        if (len(doctor_text) < 6 or len(doctor_text) > 55
                or doctor_text.count("？") + doctor_text.count("?") != 1
                or not doctor_text.endswith(("？", "?"))):
            reasons.append("ask_not_single_short_question")
        if re.search(r"[。；;！!\n]", doctor_text) or doctor_text.count("吗") > 1:
            reasons.append("ask_contains_multiple_sentences")
        clauses = re.split(r"[,，]", doctor_text.rstrip("？?"))
        if sum(bool(QUESTION_CUE.search(clause)) for clause in clauses) > 1:
            reasons.append("ask_multiple_question_clauses")
        if LOW_INFO_ASK.fullmatch(doctor_text):
            reasons.append("generic_ask")
        if GENERIC_SYMPTOM_ASK.search(doctor_text):
            reasons.append("generic_symptom_ask")
        if ASK_ADVICE.search(doctor_text):
            reasons.append("ask_contains_advice_or_diagnosis")
        if re.search(r"买.{0,12}吃|买.{0,12}用", doctor_text):
            reasons.append("ask_implies_medication_purchase")
        if re.search(r"出现多少岁|发病多少岁", doctor_text):
            reasons.append("ask_garbled_question")
        if RED_FLAG.search(doctor_text):
            reasons.append("ask_continues_red_flag_inquiry")
        if ASSERTIVE_ASK_PREFIX.search(doctor_text) or len(re.findall(r"你好|您好", doctor_text)) > 1:
            reasons.append("ask_contains_assertion_or_repeated_greeting")
        if DURATION_MENTION.search(patient[-1]) and GENERIC_DURATION_ASK.search(doctor_text):
            reasons.append("ask_repeats_known_duration")
        if AGE_MENTION.search(patient[-1]) and AGE_ASK.search(doctor_text):
            reasons.append("ask_repeats_known_age")
        if re.search(r"检查结果|结果.{0,5}(?:正常|没有|无|没|出来)", patient[-1]) and re.search(
                r"检查结果如何|检查结果.{0,4}怎么样|结果怎么样|结果如何", doctor_text):
            reasons.append("ask_repeats_known_test_result")
    else:
        if (len(doctor_text) < 30 or len(doctor_text) > 180 or "?" in doctor_text
                or "？" in doctor_text or doctor_text.endswith(("，", ",", "；", ";", "、", "：", ":"))):
            reasons.append("final_too_short_or_question")
        if FINAL_FALLBACK.search(doctor_text):
            reasons.append("final_courtesy_or_closer")
        if not FINAL_CONTENT.search(doctor_text):
            reasons.append("final_no_actionable_content")
        if not FINAL_ACTION.search(doctor_text) or not FINAL_CAUTION.search(doctor_text):
            reasons.append("final_lacks_action_or_caution")
        if not FINAL_NEXT_STEP.search(doctor_text):
            reasons.append("final_lacks_observation_or_followup")
        if (DIRECT_DRUG_ADVICE.search(doctor_text) or DIRECT_DRUG_VERB.search(doctor_text)
                or ANY_MEDICATION_IN_FINAL.search(doctor_text)):
            reasons.append("final_direct_drug_instruction")
        if OVERCONFIDENT.search(doctor_text):
            reasons.append("final_overconfident_claim")
        if INSTITUTIONAL_REPLY.search(doctor_text):
            reasons.append("final_institutional_reply")
        if TREATMENT_CERTAINTY.search(doctor_text):
            reasons.append("final_categorical_treatment_claim")
        if PROCEDURE_DIRECTIVE.search(doctor_text):
            reasons.append("final_direct_procedure_or_absolute_claim")
        if UNPROFESSIONAL_REPLY.search(doctor_text):
            reasons.append("final_unprofessional_tone_or_promotion")
        if UNSUPPORTED_DIAGNOSIS.search(doctor_text):
            reasons.append("final_categorical_diagnosis")
        if re.search(r"(?:建议|治疗)[:：]\s*$", doctor_text):
            reasons.append("final_truncated_recommendation")
    if reasons:
        return None, list(dict.fromkeys(reasons))
    cleaned = {"messages": [{"role": m["role"], "content": normalize(m["content"])}
                            for m in messages]}
    cleaned["messages"][-1]["content"] = json.dumps(
        {"action": target["action"], "message": doctor_text},
        ensure_ascii=False, separators=(",", ":"))
    return cleaned, []


def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()


def split_for_case(case_id: str) -> str:
    rank = int.from_bytes(hashlib.sha256(
        ("heapo-sft-v1:" + case_id).encode("utf-8")).digest()[:8], "big")
    return "dev" if rank / (1 << 64) < 0.1 else "train"


def source_candidate(row: dict) -> tuple[dict | None, str | None]:
    metadata = row["metadata"]
    quality = metadata["quality"]
    if metadata["synthetic"]:
        return None, "synthetic_row"
    if row["start_type"] != "continuation" or metadata["start_reason"] != "labeled":
        return None, "no_aligned_logged_action"
    if quality["target_passed"] is not True or quality["target_issues"]:
        return None, "upstream_target_flagged"
    if metadata["dialogue_truncated"]:
        return None, "truncated_source_dialogue"
    if quality["missing_image"]:
        return None, "missing_image"
    if quality["history_issues"]:
        return None, "upstream_history_flagged"
    reference = row["environment"]["reference"]
    action, text = reference["logged_action"], reference["logged_response"]
    if action not in {"ASK", "FINAL"} or not isinstance(text, str) or not text.strip():
        return None, "missing_aligned_logged_response"
    target = json.dumps({"action": action, "message": text}, ensure_ascii=False,
                        separators=(",", ":"))
    return {"messages": [*row["messages"], {"role": "assistant", "content": target}]}, None


def reservoir_add(pool: list, seen_count: int, item: tuple, rng: random.Random) -> None:
    if len(pool) < 20:
        pool.append(item)
    else:
        position = rng.randrange(seen_count)
        if position < 20:
            pool[position] = item


def run(input_parquet: Path, output_dir: Path, audit_dir: Path, seed: int = 20260929) -> dict:
    if not input_parquet.is_file():
        raise FileNotFoundError(input_parquet)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    reasons = Counter()
    randomizer = random.Random(seed)
    seen = set()
    accepted_for_review = {"ASK": [], "FINAL": []}
    rejected_for_review = []
    cases_by_split = {"train": set(), "dev": set()}
    paths = {split: output_dir / f"{split}.openai.jsonl" for split in ("train", "dev")}
    with (paths["train"].open("w", encoding="utf-8", newline="\n") as train_out,
          paths["dev"].open("w", encoding="utf-8", newline="\n") as dev_out):
        writers = {"train": train_out, "dev": dev_out}
        columns = ["case_id", "start_type", "messages", "environment", "metadata"]
        for batch in pq.ParquetFile(input_parquet).iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                record, preliminary_reason = source_candidate(row)
                if preliminary_reason:
                    counts["excluded_before_text_rules"] += 1
                    reasons[preliminary_reason] += 1
                    continue
                counts["aligned_candidates"] += 1
                split = split_for_case(row["case_id"])
                clean, flags = screen(record)
                if clean is not None:
                    key = digest(clean)
                    if key in seen:
                        flags = ["duplicate_conversation"]
                    else:
                        seen.add(key)
                if flags:
                    counts[f"{split}_rejected"] += 1
                    reasons.update(flags)
                    item = (split, counts["source_rows"], ";".join(flags), record)
                    reservoir_add(rejected_for_review,
                                  counts["train_rejected"] + counts["dev_rejected"],
                                  item, randomizer)
                    continue
                writers[split].write(json.dumps(clean, ensure_ascii=False) + "\n")
                cases_by_split[split].add(row["case_id"])
                counts[f"{split}_accepted"] += 1
                action = action_from(clean["messages"][-1])["action"]
                counts[f"{split}_{action.lower()}"] += 1
                item = (split, counts["source_rows"], "accepted", clean)
                reservoir_add(accepted_for_review[action],
                              counts[f"train_{action.lower()}"] + counts[f"dev_{action.lower()}"],
                              item, randomizer)
    if cases_by_split["train"] & cases_by_split["dev"]:
        raise ValueError("SFT train/dev case overlap")
    review_groups = {
        "accepted_review20.csv": [
            *randomizer.sample(accepted_for_review["ASK"], min(10, len(accepted_for_review["ASK"]))),
            *randomizer.sample(accepted_for_review["FINAL"], min(10, len(accepted_for_review["FINAL"]))),
        ],
        "rejected_review20.csv": randomizer.sample(rejected_for_review,
                                                   min(20, len(rejected_for_review))),
    }
    for filename, sample in review_groups.items():
        with (audit_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("split", "source_line", "result", "last_patient", "target"))
            for split, line_number, label, record in sample:
                messages = record["messages"]
                target = action_from(messages[-1])
                writer.writerow((split, line_number, label,
                                 next((m["content"] for m in reversed(messages[:-1])
                                       if m["role"] == "user"), "")[:500],
                                 (target["message"] if target else messages[-1]["content"])[:500]))
    report = {"rule_version": RULE_VERSION,
              "source_sha256": hashlib.sha256(input_parquet.read_bytes()).hexdigest(),
              "counts": dict(counts), "rejection_reasons": dict(reasons),
              "unique_cases": {split: len(cases) for split, cases in cases_by_split.items()},
              "limitations": "Deterministic filtering cannot certify clinical correctness; review accepted examples before training."}
    (audit_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.audit_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
