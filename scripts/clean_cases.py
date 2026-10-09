"""Turn the raw parsed sources into one homogeneous, non-overlapping corpus.

Three things happen here, in order.

1. Segment-level cleaning. Real consultation text mixes clinical reasoning with
   platform boilerplate inside the same turn: a doctor explains a growth-hormone
   protocol and then appends "请参阅我网站首页的文章《…》《…》". Removing the
   whole turn would discard the reasoning. So a turn is kept whole and only the
   offending spans are cut out of it. A turn disappears only when nothing is
   left after cleaning.

2. Structural normalisation. Four of the six corpora let a speaker send several
   messages in a row, and one records the patient's own account of the problem
   in a field outside the dialogue. Consecutive turns are merged, and that
   account is restored as the opening patient turn.

3. Overlap resolution. MedDG and MedDG-Session are two published views of the
   same consultations, and 40.7% of ReMeDi duplicates KaMed. MedDG-Session is
   folded into MedDG, and later sources yield to earlier ones on duplicates.

What is deliberately *not* removed: anything a clinician could have said. Turn
counts and lengths are not capped; requests for records, the patient's own
questionnaire entries, and courtesies such as "祝早日康复" all stay.

Output: data/processed/<Source>.jsonl and data/processed/_report.json

Usage:
    python scripts/clean_cases.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/sft_rl"
DEFAULT_OUTPUT = ROOT / "data/processed"

SOURCE_ORDER = ["IMCS-21", "KaMed", "MedDG", "CHIP-MDCFNPC", "MedDialog", "ReMeDi"]
MERGED_INTO = {"MedDG-Session": "MedDG"}

# Source page URLs are kept out of the corpus and written here instead.
URL_AUDIT_PATH = ROOT / "data/local/source_page_urls.jsonl"
url_sink: list[dict] = []

# --------------------------------------------------------------------------
# Spans cut out of a turn. Each pattern is anchored on a promotion or platform
# cue and stops at the end of its sentence, so the clinical text around it
# survives.
# --------------------------------------------------------------------------
# The lead-in to a request is arbitrary prose: "如果帮助到你麻烦给个满意评价吧"
# pads the cue with words no fixed list would enumerate. So the window is simply
# "whatever precedes the cue in the same breath", bounded by three things: it
# never crosses sentence punctuation, never crosses a comma, and never crosses a
# space. That last limit is what keeps clinical text safe, because a doctor who
# writes "不要喝酒 你可以点一下我的头像" separates the two with a space, while
# "如果帮助到你麻烦给个满意评价吧" is one unbroken run.
LEAD = r"[^。！？!?，,；;\s\n]{0,16}?"
TAIL = r"[^。！？!?\n]*[。！？!?\n]?"
BRACKET = r"[“\"'［\[]?\s*"
BRACKET_END = r"\s*[”\"'］\]]?"
VERB = r"(?:点击|点|戳|长按|给|留|写|打|投|来个|查看|阅读|浏览|参阅|参见|详见)?"

# Rating words split by how safely they identify promotion. "评分" is a clinical
# term, not a cue: 88% of its occurrences are Apgar scores, fetal monitoring
# scores, CHA2DS2 scores. "满意" is worse than it looks too, because "勃起硬度
# 不满意" is a symptom. Only words that never appear in clinical prose stand
# alone; the rest need a giving verb or a partner word.
RATING_ALONE = r"(?:好评|五星|评价)"
RATING_GIVEN = (r"(?:给|留|点|写|打|来个|投|送)\s*(?:个|一下|一个)?\s*"
                r"(?:满意|好评|五星|评价|评分|打分)")
RATING_PAIRED = r"(?:满意|评分|打分)\s*(?:评价|好评|评分|打分)"

PROMOTION_SPANS = [
    # Rating solicitation, with or without a verb in front of the noun.
    re.compile(LEAD + VERB + r"\s*(?:个|一下|一个)?\s*" + BRACKET +
               rf"(?:{RATING_ALONE}|{RATING_GIVEN}|{RATING_PAIRED})" + BRACKET_END + TAIL),
    re.compile(LEAD + BRACKET + r"(?:五星好评|满意评价|非常满意|很满意)"
               r"\s*(?:评价|好评|评分)?" + BRACKET_END + TAIL),
    # Follow, add, or scan. A bare "关注" is deliberately not a cue: a doctor
    # may write "需要关注一下副作用". Only a bracketed "关注" or one aimed at a
    # person counts.
    re.compile(LEAD + VERB + r"\s*(?:一下)?\s*" + BRACKET +
               r"(?:我的)?(?:头像|二维码)" + BRACKET_END + r"\s*(?:我|一下)?" + TAIL),
    # The brackets here are required, not optional: "[关注]我" is a button, a
    # bare "关注" is a verb, and "需要关注一下副作用" is a doctor talking.
    re.compile(LEAD + r"[［\[]\s*关注\s*[］\]]" + r"\s*(?:我|一下)?" + TAIL),
    re.compile(LEAD + r"关注\s*(?:一下)?\s*我" + TAIL),
    re.compile(LEAD + r"(?:加|留|给)\s*(?:我|一下)?\s*(?:微信|QQ|好友|联系方式|电话)" + TAIL),
    re.compile(LEAD + r"(?:扫|识别)\s*(?:码|描二维码|二维码)" + TAIL),
    re.compile(LEAD + r"公众号" + TAIL),
    # The doctor's own site, articles and reading list. Every qualifier here is
    # required: making them optional let "文章" alone match, and a doctor may
    # write "这篇文章提到过" without meaning anything by it.
    re.compile(LEAD + r"(?:我|本人)(?:的)?\s*(?:网站|主页|博客|文章)" + TAIL),
    re.compile(LEAD + r"(?:查看|阅读|浏览|参阅|参见|详见)\s*(?:我|本人)?\s*(?:的)?\s*"
               r"(?:网站|主页|博客|文章|科普)" + TAIL),
    re.compile(LEAD + r"(?:网站首页|我的?公众号)" + TAIL),
    re.compile(LEAD + r"(?:相关|经典|其他)(?:的)?(?:咨询|文章|科普)" + TAIL),
    re.compile(LEAD + r"温馨提示" + TAIL),
    # Coupons and gifts.
    re.compile(LEAD + r"(?:复\s*[＊*]?\s*诊\s*[＊*]?\s*)?优惠[劵券]" + TAIL),
    re.compile(LEAD + r"送出?\s*心意" + TAIL),
    # Platform flow talk about the consultation mechanic itself.
    re.compile(r"由于一问一答" + TAIL),
    re.compile(r"(?:首次回复时|这里先做简单|这里先简单)" + TAIL),
    re.compile(r"(?:后面|稍后|下次)?再?(?:帮您|为您)?增加咨询次数" + TAIL),
    re.compile(r"详细回复请稍候" + TAIL),
    # The platform's own sign-off, appended to a doctor's closing turn.
    re.compile(LEAD + r"(?:问题已经解答完毕|如有疑问.{0,8}(?:再次)?追问"
               r"|要是没问题|欢迎再次咨询|本次(?:咨询|问诊)结束)" + TAIL),
    # A run of article titles, which is a reading list rather than a sentence.
    # Two or more are required: a single 《》 may hold a value, as in "血糖《8".
    re.compile(r"(?:《[^》]{2,60}》\s*[：:]?\s*){2,}"),
    re.compile(r"《[^》]{2,60}》\s*[：:]\s*(?=《|$)"),
]

# Turns about how the platform itself works: its app, its customer service, its
# pharmacy, who can see whose messages after following. These carry no clinical
# content at all, so the whole turn goes rather than just the phrase. The
# distinction matters: a clinical sentence that happens to name the platform
# stays, a sentence that is *about* the platform does not.
PLATFORM_NAME = r"(?:春雨|平台|在线|网站|本平台)"
PLATFORM_SERVICE = [
    re.compile(LEAD + PLATFORM_NAME + r"?\s*(?:客服|服务助手|小助手|健康助手|云药房|药房)" + TAIL),
    re.compile(LEAD + r"(?:联系|拨打|咨询|找|加)\s*" + PLATFORM_NAME + r"?\s*"
               r"(?:客服|服务助手|小助手)" + TAIL),
    re.compile(LEAD + PLATFORM_NAME + r"\s*(?:医生|大夫)?\s*(?:APP|app|应用|客户端|软件|公众号)" + TAIL),
    re.compile(LEAD + r"(?:关注|添加)后[^。！？!?\n]*[。！？!?\n]?"),
    re.compile(LEAD + r"(?:以前|之前|先前)\s*(?:我)?\s*发\s*的?[^。！？!?\n]*"
               r"(?:看不到|看不见|不显示|收不到)[^。！？!?\n]*[。！？!?\n]?"),
    re.compile(LEAD + r"(?:系统|平台|网站)\s*(?:规定|限制|要求|上限|规则|设置)" + TAIL),
]

# Corrupt characters and keyboard mashing, removed rather than kept.
INLINE_ARTIFACT = re.compile(
    r"图片因隐私问题无法显示|无法查看图片|图片.{0,6}无法显示"
    r"|无法听清语音|听不清语音|该图片.{0,4}已失效"
)
CONTROL_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
KEYBOARD_NOISE = re.compile(r"(?i)\b(?:hskskskdl|asdfghjkl|qwertyuiop|hskskskskdl)\b")
PERCENT_ENCODED = re.compile(r"%[0-9a-fA-F]{2}")

# Turns that are entirely boilerplate. These are dropped whole; there is no
# clinical content to preserve.
WHOLE_TURN_NOISE = re.compile(
    r"^\s*[\[【]?\s*自动回复\s*[\]】]?|暂时不在线|不在电脑旁|请您?留言"
    r"|^稍后.{0,12}(?:回复|回你)|^一会儿.{0,12}(?:回复|回你)|晚点.{0,12}回复"
    r"|我在开车|收到您的提问了|您已进入|排队中|针对本次问诊|医生更新了总结建议"
)
KEYBOARD_ONLY = re.compile(r"^[\s?？!！.。、,，~～\-—_*＊]+$")

DANGLING_PUNCT = re.compile(r"^[\s，。、！？!?；;：:]+|[\s，、；;：:]+$")

# Some corpora record the department as a path that begins at the hospital the
# patient attended, as in "复旦大学附属华山医院>神经外科". The hospital names a
# third party and is not part of the department, so those segments are dropped.
HOSPITAL_NAME = re.compile(r"医院|医科大学|医学院|附属|分院|门诊部|卫生服务中心|医疗中心")


def strip_hospital(path: str) -> str:
    parts = [p.strip() for p in str(path).split(">")]
    return ">".join(p for p in parts if p and not HOSPITAL_NAME.search(p))
PUNCT_ONLY = re.compile(r"[\s，。、！？；：,.!?;:\"'（）()【】\[\]—\-~·＊*]+")

DIRECT_IDENTIFIERS = (
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9](?:[- ]?\d){9}(?!\d)"), "[电话已隐去]"),
    (re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"), "[证件号已隐去]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[邮箱已隐去]"),
    (re.compile(r"https?://[^\s<>，。；;]+", re.I), "[链接已隐去]"),
    # Bare domains, as in "北肿有官网，你可以上去查查www.bjcancer.org". Doctors also
    # cite hospital sites and booking portals, so the sentence is kept and only
    # the address is replaced.
    (re.compile(r"(?:www\.)?[A-Za-z0-9][A-Za-z0-9-]{1,}(?:\.[A-Za-z0-9-]{2,})*"
                r"\.(?:com|cn|net|org|edu|gov|com\.cn|org\.cn|net\.cn)"
                r"(?:/[^\s<>，。；;]*)?", re.I), "[链接已隐去]"),
    (re.compile(r"(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)"), "[电话已隐去]"),
    (re.compile(r"(?:微信|QQ|wx|vx)(?:号|账号)?\s*[:：]?\s*[A-Za-z0-9_-]{5,20}", re.I), "[账号已隐去]"),
    (re.compile(r"(?:病历号|住院号|就诊号|门诊号)\s*[:：]?\s*[A-Za-z0-9-]{5,24}"), "[病历编号已隐去]"),
)


# A fallback for clauses the span patterns still miss. Only cues that never
# occur in clinical prose qualify: a bare "关注一下" is excluded on purpose,
# because "需要关注一下副作用" is a doctor talking, not a doctor advertising.
PROMOTION_CUE = re.compile(
    r"好评|五星|评价"
    r"|(?:给|留|点|写|打|来个|投|送)\s*(?:个|一下)?\s*(?:满意|好评|评价|评分|打分)"
    r"|(?:满意|评分|打分)\s*(?:评价|好评)"
    r"|关注\s*(?:一下)?\s*我|加\s*我\s*关注|加\s*我\s*(?:微信|QQ|好友)"
    r"|扫码|二维码|公众号|我的?头像"
    r"|(?:我|本人)(?:的)?(?:网站|主页|博客|文章)|网站首页|参阅|参见|详见|经典咨询|温馨提示|小贴士"
    r"|复诊\s*[＊*]?\s*优惠|优惠[劵券]|送出?\s*心意"
)

CLAUSE_SPLIT = re.compile(r"(?<=[，,。！？!?；;、\n])")
CLAUSE_EDGE = " \t，,。！？!?；;、\n"

# Removing "麻烦给个满意评价吧" can strand the conditional that introduced it,
# as in "如果我的回答对您有帮助，". The clause survives the pass above because
# it carries no cue of its own, so it is dropped here instead. Applied only to
# turns where something was removed, so a doctor's real "如果症状加重…" is safe.
ORPHAN_LEAD = re.compile(
    r"(?:^|[\s，,；;])\s*(?:"
    r"(?:如果|若|要是|假如|倘若)[^。！？!?；;\n]{0,40}"
    r"|(?:为方便|为了|方便)[^。！？!?；;\n]{0,30}"
    r"|[^。！？!?；;\n]{0,24}的话"
    r")[，,]?\s*$"
)


def is_platform_turn(text: str) -> bool:
    """A turn that opens by talking about the platform is about the platform.

    "不是春雨医生APP么？ 具体叫什么名字？" loses its subject the moment the app
    is cut out, and what is left refers to nothing. Such turns are dropped whole
    rather than trimmed.
    """
    return any(pattern.match(text) for pattern in PLATFORM_SERVICE)


def strip_promotion(text: str) -> tuple[str, int]:
    """Cut non-clinical matter out of a turn, leaving clinical text untouched.

    Two passes. The first removes spans the fixed patterns recognise precisely,
    which keeps clinical text sharing a clause with the noise, as in
    "一般两周会见效，不要喝酒 你可以点一下我的头像，关注我". The second works by
    clause and catches lead-ins the patterns cannot enumerate. Anything it
    removes is already free of clinical content.
    """
    removed = 0
    for pattern in (*PROMOTION_SPANS, *PLATFORM_SERVICE):
        text, count = pattern.subn(" ", text)
        removed += count

    kept = []
    for clause in CLAUSE_SPLIT.split(text):
        body = clause.strip(CLAUSE_EDGE)
        if body and PROMOTION_CUE.search(body):
            removed += 1
            continue
        kept.append(clause)
    return "".join(kept), removed


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    for pattern, replacement in DIRECT_IDENTIFIERS:
        text = pattern.sub(replacement, text)
    text = INLINE_ARTIFACT.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def repair_seams(text: str) -> str:
    """Clean up what cutting a span out of a sentence leaves behind.

    Applied only where something was actually cut. Repeated punctuation is left
    alone: "好的，，谢谢" and "还在吗？？？" are how people type, and a
    question mark at the start of a turn is how a patient follows up. Those are
    the speaker's own expression, not defects to be corrected.
    """
    text = re.sub(r"([，、；;])\s*(?=[。！？!?])", "", text)
    text = DANGLING_PUNCT.sub("", text)
    return text.strip()


def annotations_of(turn: dict) -> dict:
    return {k: v for k, v in turn.items() if k not in ("role", "text")}


def combine(previous, value):
    """Keep every annotation value when two turns by one speaker are merged."""
    if previous == value:
        return previous
    if isinstance(previous, list) and isinstance(value, list):
        return previous + value
    if isinstance(previous, list):
        return [*previous, value]
    if isinstance(value, list):
        return [previous, *value]
    return [previous, value]


def merge_runs(turns: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["text"] = f"{merged[-1]['text']} {turn['text']}".strip()
            for key, value in annotations_of(turn).items():
                if key in merged[-1]:
                    merged[-1][key] = combine(merged[-1][key], value)
                else:
                    merged[-1][key] = value
        else:
            merged.append({"role": turn["role"], "text": turn["text"], **annotations_of(turn)})
    return merged


def clean_dialogue(row: dict, stats: collections.Counter) -> tuple[dict | None, str | None]:
    turns: list[dict] = []
    for turn in row["turns"]:
        text = normalize_text(turn.get("text") or "")
        if not text:
            stats["空发言跳过"] += 1
            continue
        if CONTROL_CHARS.search(text) or KEYBOARD_NOISE.search(text):
            stats["乱码轮丢弃"] += 1
            continue
        if KEYBOARD_ONLY.match(text):
            stats["无内容轮丢弃"] += 1
            continue
        if WHOLE_TURN_NOISE.search(text):
            stats["整轮模板丢弃"] += 1
            continue
        if PERCENT_ENCODED.search(text):
            stats["链接编码轮丢弃"] += 1
            continue
        if is_platform_turn(text):
            stats["平台功能轮丢弃"] += 1
            continue

        text, removed = strip_promotion(text)
        if removed:
            stats["轮内非临床片段删除"] += removed
            text = repair_seams(ORPHAN_LEAD.sub("", text))
        if not text or KEYBOARD_ONLY.match(text):
            stats["清理后空轮丢弃"] += 1
            continue
        turns.append({
            "role": turn["role"],
            "text": text,
            **annotations_of(turn),
        })

    if not turns:
        return None, "清洗后无有效轮次"

    # One corpus keeps the patient's account of the problem outside the
    # dialogue; restore it as the opening turn so the consultation has a premise.
    opening = (row.get("labels") or {}).get("self_report")
    if isinstance(opening, str):
        opening = normalize_text(opening)
        if opening and turns[0]["role"] != "patient":
            turns.insert(0, {"role": "patient", "text": opening})
            stats["补入患者自述"] += 1

    turns = merge_runs(turns)
    patients = sum(1 for t in turns if t["role"] == "patient")
    if not patients or patients == len(turns):
        return None, "缺少患者或医生发言"

    cleaned = dict(row)
    cleaned["turns"] = turns
    # dict(row) is shallow: without this copy the label handling below would
    # reach back into the caller's row and strip its page_url too.
    if isinstance(cleaned.get("labels"), dict):
        cleaned["labels"] = dict(cleaned["labels"])
    # Source page URLs carry the consulting doctor's account name, which is a
    # third party's identifier and has no place in a training corpus. They are
    # held in a local audit file instead.
    labels = cleaned.get("labels")
    if isinstance(labels, dict):
        if "page_url" in labels:
            url_sink.append({
                "corpus": row["source"],
                "original_id": str(row["id"]),
                "page_url": labels.pop("page_url"),
            })
        if labels.get("disease_grad"):
            labels["disease_grad"] = strip_hospital(labels["disease_grad"])
        if not any(v not in (None, "", [], {}) for v in labels.values()):
            cleaned.pop("labels", None)
    return cleaned, None


def overlap_key(turns: list[dict]) -> str:
    parts = [f"{t['role']}:{PUNCT_ONLY.sub('', t['text'])}" for t in turns]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def head_key(turns: list[dict], size: int = 6) -> str:
    parts = [f"{t['role']}:{PUNCT_ONLY.sub('', t['text'])}" for t in turns[:size]]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def same_turn(left: dict, right: dict) -> bool:
    return (left["role"] == right["role"]
            and PUNCT_ONLY.sub("", left["text"]) == PUNCT_ONLY.sub("", right["text"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    cleaned_by_source: dict[str, list[dict]] = {}
    rejects_by_source: dict[str, collections.Counter] = {}

    # Only the parsed corpora are inputs. The same directory also receives the
    # RL parquet files and the exported SFT jsonl; globbing *.jsonl there would
    # feed the SFT export back in as if it were a raw corpus.
    source_paths = [args.input / f"{name}.jsonl"
                    for name in [*SOURCE_ORDER, *MERGED_INTO]]
    for path in sorted(p for p in source_paths if p.is_file()):
        kept: list[dict] = []
        rejects: collections.Counter = collections.Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                cleaned, reason = clean_dialogue(json.loads(line), stats)
                if cleaned is None:
                    rejects[reason] += 1
                    continue
                kept.append(cleaned)
        cleaned_by_source[path.stem] = kept
        rejects_by_source[path.stem] = rejects
        print(f"清洗 {path.stem:<18}{len(kept):>8,} 保留")

    # Fold MedDG-Session's doctor action labels onto the matching MedDG dialogues.
    donor = next((n for n, target in MERGED_INTO.items() if target == "MedDG"), None)
    if donor and donor in cleaned_by_source:
        target_rows: dict[str, dict] = {}
        for row in cleaned_by_source["MedDG"]:
            target_rows.setdefault(head_key(row["turns"]), row)
        matched = 0
        for row in cleaned_by_source[donor]:
            match = target_rows.get(head_key(row["turns"]))
            if match is None:
                continue
            for src_turn, dst_turn in zip(row["turns"], match["turns"]):
                if not same_turn(src_turn, dst_turn):
                    break
                if src_turn["role"] == "doctor" and dst_turn.get("type") is None:
                    dst_turn["type"] = src_turn.get("type")
            matched += 1
        print(f"\n标注合并 MedDG-Session → MedDG:  匹配 {matched:,} 个对话")
        for name in MERGED_INTO:
            cleaned_by_source.pop(name, None)

    # Cross-source deduplication. Earlier sources win on a duplicate.
    seen: dict[str, str] = {}
    order = [n for n in SOURCE_ORDER if n in cleaned_by_source]
    order += [n for n in cleaned_by_source if n not in order]
    for name in order:
        kept = []
        dropped = 0
        for row in cleaned_by_source[name]:
            key = overlap_key(row["turns"])
            if key in seen:
                dropped += 1
                continue
            seen[key] = name
            kept.append(row)
        cleaned_by_source[name] = kept
        print(f"去重 {name:<18}{len(kept):>8,} 保留   跨源剔除 {dropped:,}")

    print()
    report: dict[str, dict] = {}
    total_turns = 0
    for name in order:
        rows = cleaned_by_source[name]
        out_path = args.output / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        turns = sum(len(r["turns"]) for r in rows)
        total_turns += turns
        report[name] = {
            "dialogues": len(rows),
            "turns": turns,
            "rejected": dict(rejects_by_source.get(name, {})),
        }
        print(f"写出 {name:<18}{len(rows):>8,} 对话   {turns:>10,} 轮")

    total_dialogues = sum(v["dialogues"] for v in report.values())
    print()
    print("=== 清洗动作统计 ===")
    for key, value in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"   {key:<20}{value:>10,}")
    print()
    print(f"合计 {total_dialogues:,} 个对话 / {total_turns:,} 轮")
    print("=== 各来源淘汰原因 ===")
    for name, entry in report.items():
        if entry["rejected"]:
            detail = ", ".join(f"{k} {v:,}" for k, v in entry["rejected"].items())
            print(f"   {name:<16}{detail}")

    report["_summary"] = {
        "sources": order,
        "merged": dict(MERGED_INTO),
        "total_dialogues": total_dialogues,
        "total_turns": total_turns,
        "cleaning_actions": dict(stats),
    }
    (args.output / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    URL_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with URL_AUDIT_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in url_sink:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n来源页面网址（本地审计，不随数据集发布）: {URL_AUDIT_PATH}  {len(url_sink):,} 条")
    print(f"报告: {args.output / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
