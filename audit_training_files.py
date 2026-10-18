#!/usr/bin/env python
"""Independent audit of the training artifacts (episodes and SFT records).

The auditor reads whatever is in --input (parquet and jsonl are detected by
content), reports measured numbers per check, and never modifies the data it
reads. Every finding carries a locator (file, line or row number, sample id)
plus a content hash so a row can be traced back to the file it came from.

Checks:
  a) schema: missing required fields, null / empty rates, internal consistency
  b) split leakage: case id, normalised first patient question, near duplicate
     (first 40 characters) across train / dev / test
  c) hidden information: future patient turns, reference conclusion and
     diagnosis text appearing inside the visible prompt
  d) conclusion integrity: empty / short / pleasantry-only conclusions, and
     surface overlap between presenting problem and conclusion
  e) distribution: turn counts, action counts, source share, length quantiles
  f) exact duplicate episodes (prompt + ground_truth fingerprint)
  g) SFT side: role alternation, assistant JSON validity, action vocabulary,
     case leakage across splits

Usage:
  python scripts/audit_training_files.py --input data/sft_rl \
      --output data/audit_v3/baseline_existing.json \
      --report-md docs/audit/baseline_existing.md

Sampling is deterministic: a row is included when
sha256("<seed>:<key>") % 10000 < sample_rate * 10000, so the same seed and rate
always reproduce the same subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "sft_rl"
DEFAULT_OUTPUT = ROOT / "data" / "audit_v3" / "audit.json"

AUDIT_VERSION = "1.0"
DEFAULT_SEED = 20261016
DEFAULT_SAMPLE_RATE = 1.0

SPLIT_NAMES = ("train", "dev", "test")
ACTIONS = ("WEN", "TALK", "DIG")
VALID_ROLES = ("user", "assistant", "system", "patient", "doctor")

# A conclusion shorter than this teaches the reward nothing (the builder uses
# the same threshold); fewer than MIN_CONCLUSION_ABS characters is a blank.
MIN_CONCLUSION_CHARS = 10
MIN_CONCLUSION_ABS = 5

# Leak detection ignores questions shorter than this after normalisation: a
# bare greeting repeats across unrelated patients and would drown the signal.
MIN_LEAK_QUESTION_CHARS = 12
NEAR_DUP_PREFIX = 40
NGRAM = 8
MIN_FUTURE_TURN_CHARS = 8

PUNCT = re.compile(
    r"[\s\u3000\u200b，。、！？；：“”‘’（）()\[\]【】《》〈〉…—~～,.!?;:'\"\x60"
    r"/\\|_*#=+]+")
PLEASANTRY_ONLY = re.compile(
    r"^(?:好的?|嗯+|哦+|行|谢谢|谢了|感谢|不客气|再见|拜拜|"
    r"祝您?|祝早日|祝康复|祝宝宝|欢迎随时|随时咨询|"
    r"对本次咨询满意|满意|好评|给个好评|早日康复|"
    r"有问题随时|随时联系)+$")

RULES: dict[str, tuple[str, str]] = {
    # code: (severity, description in Chinese)
    "file_unreadable": ("blocker", "文件无法解析，整份产物不可用"),
    "schema_missing_field": ("blocker", "必填字段缺失"),
    "prompt_empty": ("blocker", "prompt 为空或首条发言为空"),
    "prompt_role_invalid": ("high", "prompt 中出现非法角色名"),
    "prompt_not_user_first": ("high", "prompt 首条不是患者/user 发言"),
    "prompt_multiple_turns": ("high", "prompt 含多条消息（可见上下文已越过首轮）"),
    "gt_missing": ("blocker", "ground_truth 整块为空"),
    "dialogue_empty": ("blocker", "patient_state.full_dialogue 为空"),
    "dialogue_role_invalid": ("high", "full_dialogue 中出现非法角色名"),
    "dialogue_odd_roles": ("low", "full_dialogue 相邻发言角色相同"),
    "id_empty": ("blocker", "case_id 或 sample_id 为空"),
    "id_duplicate_within_split": ("high", "同一 split 内 sample_id 重复"),
    "index_mismatch": ("low", "index 列与行序不一致"),
    "data_source_unexpected": ("low", "data_source 取值与预期不符"),
    "profile_dialogue_turns_mismatch": ("high", "profile.dialogue_turns 与真实轮数不符"),
    "profile_doctor_turns_mismatch": ("high", "profile.doctor_turns 与真实医生轮数不符"),
    "profile_action_sum_mismatch": ("high", "profile 中 wen+dig+talk 与医生轮数不符"),
    "profile_negative": ("high", "profile 中出现负数计数"),
    "profile_dig_zero": ("medium", "profile.dig 为 0，该 episode 没有结案轮"),
    "conclusion_empty": ("blocker", "reference_conclusion 为空"),
    "conclusion_too_short": ("high", "reference_conclusion 少于 5 字"),
    "conclusion_below_builder_min": ("medium", "reference_conclusion 少于 10 字（低于构建器门槛）"),
    "conclusion_pleasantry_only": ("high", "reference_conclusion 只是寒暄或确认语"),
    "conclusion_not_in_dialogue": ("blocker", "reference_conclusion 不在 full_dialogue 中"),
    "conclusion_by_patient": ("high", "reference_conclusion 与患者某轮原文相同"),
    "conclusion_superseded_by_later_turn": ("high", "结局之后仍有实质性医生发言（回扫选错了轮次）"),
    "conclusion_promo_boilerplate": ("blocker", "reference_conclusion 是推广/索评/联系方式模板"),
    "dialogue_starts_with_doctor": ("high", "对话首轮不是患者发言，prompt 由医生轮填充并被标成 user"),
    "conclusion_intake_boilerplate": ("high", "reference_conclusion 是患者挂号表单原文而非临床结论"),
    "presenting_problem_is_department": ("high", "presenting_problem 取值是科室名，字段语义与名字不符"),
    "department_is_path": ("medium", "department 取值是 >> 分层的科室路径（来自 disease_grad 分级字段）"),
    "conclusion_duplicate_cross_split": ("blocker", "同一段实质性结局原文出现在多个 split（测试集的结局参考在训练集里可复现）"),
    "conclusion_duplicate_within_split": ("high", "同一段结局原文在同一 split 内重复"),
    "conclusion_leak_in_prompt": ("blocker", "结局原文出现在可见 prompt 中"),
    "conclusion_partial_leak_in_prompt": ("high", "结局与 prompt 有至少 2 个 8 字片段重合"),
    "diagnosis_leak_in_prompt": ("high", "diagnosis 文本出现在可见 prompt 中"),
    "future_patient_leak_in_prompt": ("blocker", "未来患者话语出现在可见 prompt 中"),
    "pp_conclusion_no_overlap": ("medium", "presenting_problem 与结局无表面字符重合（仅统计）"),
    "case_id_cross_split": ("blocker", "同一 case_id 出现在多个 split"),
    "first_question_cross_split": ("blocker", "同一首问出现在多个 split"),
    "near_duplicate_cross_split": ("high", "首问前 40 字相同（近似重复）且跨 split"),
    "episode_duplicate_within_split": ("high", "同一 split 内 episode 完全重复"),
    "episode_duplicate_cross_split": ("blocker", "完全重复的 episode 跨 split"),
    "sft_messages_missing": ("blocker", "SFT 记录缺少 messages"),
    "sft_messages_too_short": ("high", "SFT messages 少于 2 条"),
    "sft_sample_id_missing": ("blocker", "SFT 记录缺少 sample_id"),
    "sft_role_alternation_invalid": ("high", "SFT messages 角色未交替"),
    "sft_first_message_not_user": ("high", "SFT 首条消息不是 user"),
    "sft_empty_content": ("high", "SFT 消息内容为空"),
    "sft_assistant_invalid_json": ("blocker", "assistant 内容不是合法 JSON"),
    "sft_assistant_missing_key": ("high", "assistant JSON 缺少 action 或 message"),
    "sft_action_invalid": ("blocker", "assistant action 不在 WEN/TALK/DIG 内"),
    "sft_message_empty": ("high", "assistant JSON 的 message 为空"),
    "sft_sample_id_cross_split": ("blocker", "同一 SFT sample_id 出现在多个 split"),
    "sft_first_question_cross_split": ("blocker", "SFT 首问出现在多个 split"),
    "sft_duplicate_within_split": ("high", "同一 split 内 SFT 记录完全重复"),
    "sft_duplicate_cross_split": ("blocker", "完全相同的 SFT 记录跨 split"),
    "sft_last_message_is_user": ("low", "SFT 记录以 user 结尾（非决策点）"),
    "cross_artifact_split_conflict": ("blocker", "同一首问在 parquet 与 SFT 产物中归属的 split 集合不一致（两份产物覆盖的行不同）"),
}

SEVERITY_RANK = {"blocker": 4, "high": 3, "medium": 2, "low": 1}


# --------------------------------------------------------------------------
# helpers


def sha(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:length]


def normalize(text: str) -> str:
    return PUNCT.sub("", text).lower()


def ngrams(text: str, size: int = NGRAM) -> set:
    flat = normalize(text)
    if len(flat) < size:
        return {flat} if flat else set()
    return {flat[i:i + size] for i in range(len(flat) - size + 1)}


DEPARTMENT_LIKE = re.compile(r"(?:科|室|门诊|中心)$")
PROMO_STRONG = re.compile(
    r"好评|评价我|给个满意|满意.*评价|公众号|微信|二维码|扫描|我的诊所|小诊所|"
    r"关注我|科普文章|留言|优惠|预约挂号|点击|电话咨询|联系我")
INTAKE_FORM = re.compile(r"预约医生|病情描述|希望获得的帮助|患病多久|已就诊医院|过敏史|用药情况")

PLATFORM_TURN = re.compile(
    r"语音因隐私|图片因隐私|患者上传|因隐私问题|无法显示|已撤回|该消息|"
    r"消息已|系统提示|^s*$")


def is_promo(text: str) -> bool:
    """Marketing or solicitation boilerplate in a doctor turn."""
    if not text:
        return False
    if PROMO_STRONG.search(text):
        return True
    return bool(PLEASANTRY_ONLY.match(normalize(text))) and len(normalize(text)) < 25


def leak_kind(text: str) -> str:
    """Classify a leaked turn so the report can name the mechanism."""
    if PLATFORM_TURN.search(text):
        return "平台功能轮"
    if PLEASANTRY_ONLY.match(normalize(text)):
        return "寒暄/确认"
    return "患者正文"


def shares_surface(a: str, b: str, size: int = 2) -> bool:
    """Any shared character run of the given size: the surface-link test."""
    left = normalize(a)
    right = normalize(b)
    if len(left) < size or len(right) < size:
        return bool(left) and bool(right) and left in right or right in left
    return any(left[i:i + size] in right for i in range(len(left) - size + 1))


def head_hash(text: str, size: int = NEAR_DUP_PREFIX) -> str:
    return sha(normalize(text)[:size], 20)


def is_blank(value) -> bool:
    return value is None or value == "" or value == [] or value == {}


def short(text: str, limit: int = 100) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat[:limit]


def quantiles(values: list, points=(50, 90, 99)) -> dict:
    if not values:
        return {f"p{point}": None for point in points} | {"mean": None, "max": None}
    ordered = sorted(values)
    last = len(ordered) - 1
    out = {}
    for point in points:
        position = min(last, max(0, int(round(point / 100 * last))))
        out[f"p{point}"] = ordered[position]
    out["mean"] = round(sum(ordered) / len(ordered), 2)
    out["max"] = ordered[-1]
    return out


class Findings:
    """Counts every hit, keeps at most N examples per rule for the report."""

    def __init__(self, max_per_rule: int = 5):
        self.max_per_rule = max_per_rule
        self.counts: Counter = Counter()
        self.examples: dict = defaultdict(list)
        self.files_hit: dict = defaultdict(Counter)

    def hit(self, rule: str, file: str = "", locator: str = "", snippet: str = "",
            detail: str = "") -> None:
        self.counts[rule] += 1
        self.files_hit[rule][file] += 1
        bucket = self.examples[rule]
        if len(bucket) < self.max_per_rule:
            bucket.append({
                "file": file,
                "locator": locator,
                "snippet": short(snippet),
                "text_sha": sha(snippet or locator, 16),
                "detail": detail,
            })

    def to_dict(self) -> dict:
        out = {}
        for rule, count in self.counts.items():
            severity, description = RULES.get(rule, ("medium", rule))
            out[rule] = {
                "count": count,
                "severity": severity,
                "description": description,
                "files": dict(self.files_hit[rule]),
                "examples": self.examples.get(rule, []),
            }
        return out


class Sampler:
    """Deterministic hash sampling so a rerun keeps the identical subset."""

    def __init__(self, seed: int, rate: float):
        self.seed = seed
        self.rate = rate
        self.threshold = int(round(rate * 10000))

    def keep(self, key: str) -> bool:
        if self.rate >= 1.0:
            return True
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()[:8]
        return int(digest, 16) % 10000 < self.threshold


def read_jsonl_first_record(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if line.strip():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    return None


def classify(path: Path, first) -> str:
    if path.suffix.lower() == ".parquet":
        return "episode_parquet"
    if first is None:
        return "unknown"
    if "messages" in first:
        return "sft_jsonl"
    if "turns" in first:
        return "corpus_jsonl"
    if "prompt" in first and "ground_truth" in first:
        return "episode_jsonl"
    return "unknown"


# --------------------------------------------------------------------------
# audit state shared by every file


class AuditState:
    def __init__(self, seed: int, rate: float, max_examples: int):
        self.findings = Findings(max_examples)
        self.sampler = Sampler(seed, rate)
        self.seed = seed
        self.rate = rate
        self.files: list = []
        self.skipped: list = []

        # cross split maps: key -> {split: count}
        self.case_splits = defaultdict(lambda: defaultdict(int))
        self.q1_splits = defaultdict(lambda: defaultdict(int))
        self.q1_text: dict = {}
        self.head_splits = defaultdict(lambda: defaultdict(int))
        self.head_text: dict = {}
        self.head_questions = defaultdict(set)
        self.episode_fp_splits = defaultdict(lambda: defaultdict(int))
        self.conclusion_splits = defaultdict(lambda: defaultdict(int))
        self.conclusion_text: dict = {}
        self.promo_splits = defaultdict(lambda: defaultdict(int))
        self.promo_text: dict = {}
        # member locators so a cross-split finding names the rows it joins
        self.q1_members: dict = {}
        self.sft_q1_members: dict = {}
        self.conclusion_members: dict = {}
        self.sft_sample_splits = defaultdict(lambda: defaultdict(int))
        self.sft_q1_splits = defaultdict(lambda: defaultdict(int))
        self.sft_q1_text: dict = {}
        self.sft_fp_splits = defaultdict(lambda: defaultdict(int))

        # uniqueness inside one split
        self.sample_id_seen = Counter()
        self.sft_sample_id_seen = Counter()

        # field emptiness, keyed by (file, field)
        self.field_blank = Counter()
        self.field_total = Counter()
        self.current_file = ""

        # distributions
        self.global_dist = defaultdict(list)
        self.metrics = Counter()
        self.action_totals = Counter()
        self.sft_action_totals = Counter()
        self.data_sources = Counter()

    def field(self, name: str, blank: bool) -> None:
        key = (self.current_file, name)
        self.field_total[key] += 1
        if blank:
            self.field_blank[key] += 1


MEMBER_CAP = 4


def remember(store: dict, key: str, locator: str) -> None:
    bucket = store.setdefault(key, [])
    if len(bucket) < MEMBER_CAP:
        bucket.append(locator)


def register_question(state: AuditState, text: str, split: str, locator: str = "") -> None:
    flat = normalize(text)
    if not flat:
        return
    state.q1_splits[flat][split] += 1
    state.q1_text.setdefault(flat, text)
    if locator:
        remember(state.q1_members, flat, locator)
    head = head_hash(text)
    state.head_splits[head][split] += 1
    state.head_text.setdefault(head, text)
    state.head_questions[head].add(flat)


def register_sft_question(state: AuditState, text: str, split: str, locator: str = "") -> None:
    flat = normalize(text)
    if not flat:
        return
    state.sft_q1_splits[flat][split] += 1
    state.sft_q1_text.setdefault(flat, text)
    if locator:
        remember(state.sft_q1_members, flat, locator)


# --------------------------------------------------------------------------
# episode records (parquet and jsonl)


def audit_episode(record: dict, split: str, file_name: str, row_no: int,
                  state: AuditState, dist: dict) -> None:
    findings = state.findings
    prompt = record.get("prompt") if isinstance(record, dict) else None
    ground_truth = record.get("ground_truth") if isinstance(record, dict) else None
    extra = record.get("extra_info") if isinstance(record, dict) else None
    if not isinstance(extra, dict):
        extra = {}
    patient_state = extra.get("patient_state") or {}
    if not isinstance(patient_state, dict):
        patient_state = {}
    full_dialogue = patient_state.get("full_dialogue") or []
    profile = extra.get("profile") or {}
    if not isinstance(profile, dict):
        profile = {}

    case_id = extra.get("case_id")
    sample_id = extra.get("sample_id")
    locator = f"{file_name}#row={row_no}" + (f" sample_id={sample_id}" if sample_id else "")
    if isinstance(ground_truth, dict) and ground_truth.get("reference_conclusion"):
        locator += f" case_id={case_id}"

    state.field("prompt", is_blank(prompt))
    state.field("ground_truth", is_blank(ground_truth))
    state.field("extra_info", is_blank(extra))
    state.field("extra_info.sample_id", is_blank(sample_id))
    state.field("extra_info.case_id", is_blank(case_id))
    state.field("extra_info.patient_state.full_dialogue", is_blank(full_dialogue))
    state.field("extra_info.patient_state.symptoms",
                is_blank(patient_state.get("symptoms")))
    if not isinstance(ground_truth, dict):
        state.field("ground_truth.reference_conclusion", True)
        ground_truth = {}
    for key in ("reference_conclusion", "diagnosis", "presenting_problem", "department"):
        state.field(f"ground_truth.{key}", is_blank(ground_truth.get(key)))

    if is_blank(sample_id) or is_blank(case_id):
        findings.hit("id_empty", file_name, locator, detail="sample_id/case_id 为空")

    if isinstance(prompt, list) and prompt:
        prompt_text = "\n".join(str(m.get("content") or "") for m in prompt if isinstance(m, dict))
        bad_roles = [m.get("role") for m in prompt if isinstance(m, dict) and m.get("role") not in VALID_ROLES]
        if bad_roles:
            findings.hit("prompt_role_invalid", file_name, locator, prompt_text,
                         detail=f"roles={bad_roles[:3]}")
        dist["prompt_message_count"].append(len(prompt))
        if len(prompt) > 1:
            findings.hit("prompt_multiple_turns", file_name, locator,
                         detail=f"n={len(prompt)}")
        first_role = prompt[0].get("role") if isinstance(prompt[0], dict) else None
        if first_role not in ("user", "patient"):
            findings.hit("prompt_not_user_first", file_name, locator, prompt_text,
                         detail=f"first role={first_role}")
    else:
        prompt_text = ""
        findings.hit("prompt_empty", file_name, locator, detail="prompt 为空")
    if not prompt_text.strip():
        findings.hit("prompt_empty", file_name, locator, detail="prompt 内容为空")
    flat_prompt_all = normalize(prompt_text)
    dist["prompt_chars"].append(len(prompt_text))

    if not isinstance(ground_truth, dict) or not ground_truth:
        findings.hit("gt_missing", file_name, locator, detail="ground_truth 为空")

    turns: list = []
    if isinstance(full_dialogue, list) and full_dialogue:
        for turn in full_dialogue:
            if isinstance(turn, dict):
                turns.append((str(turn.get("role") or ""), str(turn.get("content") or "")))
            else:
                turns.append(("", ""))
    else:
        findings.hit("dialogue_empty", file_name, locator, detail="full_dialogue 为空")

    if turns and turns[0][0] != "patient":
        state.metrics["rows_dialogue_starts_with_doctor"] += 1
        findings.hit("dialogue_starts_with_doctor", file_name, locator, prompt_text,
                     detail=f"first_role={turns[0][0]}")

    if turns:
        bad = {role for role, _ in turns if role not in VALID_ROLES}
        if bad:
            findings.hit("dialogue_role_invalid", file_name, locator, str(sorted(bad)))
        repeats = [i for i in range(1, len(turns)) if turns[i][0] == turns[i - 1][0]]
        if repeats:
            findings.hit("dialogue_odd_roles", file_name, locator,
                         f"{turns[repeats[0]-1][0]}->{turns[repeats[0]][0]}",
                         detail=f"{len(repeats)} 处相邻同角色")
        dist["dialogue_turns"].append(len(turns))
        dist["dialogue_chars"].append(sum(len(text) for _, text in turns))
        doctor_turns = sum(1 for role, _ in turns if role == "doctor")
    else:
        doctor_turns = 0
    dist["doctor_turns"].append(doctor_turns)

    if profile:
        dialogue_turns_meta = profile.get("dialogue_turns")
        doctor_turns_meta = profile.get("doctor_turns")
        wen = profile.get("wen") or 0
        dig = profile.get("dig") or 0
        talk = profile.get("talk") or 0
        if isinstance(dialogue_turns_meta, int) and dialogue_turns_meta != len(turns):
            findings.hit("profile_dialogue_turns_mismatch", file_name, locator, detail=f"meta={dialogue_turns_meta} real={len(turns)}")
        if isinstance(doctor_turns_meta, int) and doctor_turns_meta != doctor_turns:
            findings.hit("profile_doctor_turns_mismatch", file_name, locator, detail=f"meta={doctor_turns_meta} real={doctor_turns}")
        if wen + dig + talk != doctor_turns:
            findings.hit("profile_action_sum_mismatch", file_name, locator, detail=f"wen+dig+talk={wen+dig+talk} doctor_turns={doctor_turns}")
        if min(wen, dig, talk) < 0:
            findings.hit("profile_negative", file_name, locator, detail=f"wen={wen} dig={dig} talk={talk}")
        if dig == 0:
            findings.hit("profile_dig_zero", file_name, locator, detail="dig=0")
        for key, value in (("wen", wen), ("dig", dig), ("talk", talk)):
            state.action_totals[(state.current_file, key)] += value
        state.action_totals[(state.current_file, "doctor_turns")] += (
            doctor_turns_meta if isinstance(doctor_turns_meta, int) else 0)

    conclusion = str(ground_truth.get("reference_conclusion") or "")
    flat_conclusion = normalize(conclusion)
    raw_length = len(conclusion.strip())
    dist["conclusion_chars"].append(raw_length)
    if not conclusion.strip():
        findings.hit("conclusion_empty", file_name, locator, detail="reference_conclusion 为空")
    else:
        if raw_length < MIN_CONCLUSION_ABS:
            findings.hit("conclusion_too_short", file_name, locator, conclusion,
                         detail=f"len={raw_length}")
        elif raw_length < MIN_CONCLUSION_CHARS:
            findings.hit("conclusion_below_builder_min", file_name, locator, conclusion,
                         detail=f"len={raw_length}")
        if PLEASANTRY_ONLY.match(flat_conclusion):
            findings.hit("conclusion_pleasantry_only", file_name, locator, conclusion)

        if turns:
            stripped = conclusion.strip()
            exact = any(text.strip() == stripped for _, text in turns)
            contained = exact or any(
                flat_conclusion in normalize(text) for _, text in turns if text)
            if not contained:
                findings.hit("conclusion_not_in_dialogue", file_name, locator, conclusion)
            if len(flat_conclusion) >= 8:
                for role, text in turns:
                    if role == "patient" and normalize(text) == flat_conclusion:
                        findings.hit("conclusion_by_patient", file_name, locator, conclusion)
                        break

            doctor_indexes = [i for i, (role, _) in enumerate(turns) if role == "doctor"]
            if doctor_indexes:
                chosen = None
                for index, (role, text) in enumerate(turns):
                    if role == "doctor" and text.strip() == stripped:
                        chosen = index
                if chosen is not None:
                    gap = doctor_indexes[-1] - chosen
                    dist["conclusion_gap"].append(gap)
                    if gap == 0:
                        state.metrics["rows_conclusion_is_final_doctor_turn"] += 1
                    substantive_after = [
                        index for index in doctor_indexes
                        if index > chosen
                        and len(turns[index][1].strip()) >= MIN_CONCLUSION_CHARS
                        and not PLEASANTRY_ONLY.match(normalize(turns[index][1]))]
                    if substantive_after:
                        state.metrics["rows_conclusion_superseded"] += 1
                        findings.hit("conclusion_superseded_by_later_turn", file_name, locator,
                                     conclusion,
                                     detail=f"conclusion_turn={chosen} "
                                            f"later_substantive={substantive_after[:3]} "
                                            f"last_doctor_turn={doctor_indexes[-1]}")

            if len(INTAKE_FORM.findall(conclusion)) >= 2:
                findings.hit("conclusion_intake_boilerplate", file_name, locator, conclusion)
            if is_promo(conclusion):
                state.metrics["rows_conclusion_promo"] += 1
                findings.hit("conclusion_promo_boilerplate", file_name, locator, conclusion)

        flat_prompt = flat_prompt_all
        if len(flat_conclusion) >= 6 and flat_conclusion in flat_prompt:
            findings.hit("conclusion_leak_in_prompt", file_name, locator, conclusion)
        elif flat_prompt and len(ngrams(conclusion) & ngrams(prompt_text)) >= 2:
            findings.hit("conclusion_partial_leak_in_prompt", file_name, locator, conclusion)

        diagnosis = str(ground_truth.get("diagnosis") or "")
        if diagnosis:
            flat_dx = normalize(diagnosis)
            if len(flat_dx) >= 3 and flat_dx in flat_prompt:
                findings.hit("diagnosis_leak_in_prompt", file_name, locator, diagnosis)

        presenting = str(ground_truth.get("presenting_problem") or "")
        if presenting:
            if not shares_surface(presenting, conclusion):
                findings.hit("pp_conclusion_no_overlap", file_name, locator,
                             f"pp={presenting} | concl={conclusion[:40]}")
            if DEPARTMENT_LIKE.search(presenting.strip()):
                findings.hit("presenting_problem_is_department", file_name, locator,
                             presenting)
        department = str(ground_truth.get("department") or "")
        if ">" in department or "＞" in department:
            findings.hit("department_is_path", file_name, locator, department)

    if flat_prompt_all:
        for index, (role, text) in enumerate(turns):
            if index == 0 or role != "patient":
                continue
            flat_turn = normalize(text)
            if len(flat_turn) >= MIN_FUTURE_TURN_CHARS and flat_turn in flat_prompt_all:
                kind = leak_kind(text)
                state.metrics[f"future_leak_kind_{kind}"] += 1
                findings.hit("future_patient_leak_in_prompt", file_name, locator, text,
                             detail=f"turn={index} kind={kind}")
                break

    if not is_blank(ground_truth.get("presenting_problem")):
        state.metrics["rows_with_presenting_problem"] += 1
    if not is_blank(ground_truth.get("diagnosis")):
        state.metrics["rows_with_diagnosis"] += 1
    if not is_blank(ground_truth.get("department")):
        state.metrics["rows_with_department"] += 1
    if not is_blank(patient_state.get("symptoms")):
        state.metrics["rows_with_symptoms"] += 1
    if turns:
        state.metrics["rows_with_dialogue"] += 1
        if any(PLEASANTRY_ONLY.match(normalize(text)) for _, text in turns):
            state.metrics["rows_with_pleasantry_turn"] += 1

    data_source = record.get("data_source") if isinstance(record, dict) else None
    state.data_sources[(state.current_file, str(data_source))] += 1
    if is_blank(data_source):
        findings.hit("data_source_unexpected", file_name, locator, detail="data_source 为空")

    index_value = record.get("index") if isinstance(record, dict) else None
    if isinstance(index_value, int) and index_value != row_no:
        findings.hit("index_mismatch", file_name, locator, detail=f"index={index_value} row={row_no}")

    if not is_blank(sample_id):
        state.sample_id_seen[(split, str(sample_id))] += 1
    if not is_blank(case_id):
        state.case_splits[str(case_id)][split] += 1
    register_question(state, prompt_text, split, locator)
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        dist.setdefault("first_prompt_chars", []).append(len(str(prompt[0].get("content") or "")))

    if len(flat_conclusion) >= 15:
        conclusion_key = sha(flat_conclusion, 20)
        if is_promo(conclusion):
            state.promo_splits[conclusion_key][split] += 1
            state.promo_text.setdefault(conclusion_key, conclusion)
        else:
            state.conclusion_splits[conclusion_key][split] += 1
            state.conclusion_text.setdefault(conclusion_key, conclusion)
            remember(state.conclusion_members, conclusion_key, locator)
    fingerprint = sha(json.dumps([prompt, ground_truth], ensure_ascii=False, sort_keys=True), 20)
    state.episode_fp_splits[fingerprint][split] += 1


# --------------------------------------------------------------------------
# SFT records


def audit_sft(record: dict, split: str, file_name: str, line_no: int,
              state: AuditState, dist: dict) -> None:
    findings = state.findings
    messages = record.get("messages") if isinstance(record, dict) else None
    sample_id = record.get("sample_id") if isinstance(record, dict) else None
    locator = f"{file_name}#line={line_no}" + (f" sample_id={sample_id}" if sample_id else "")

    state.field("sample_id", is_blank(sample_id))
    state.field("messages", is_blank(messages))
    if is_blank(sample_id):
        findings.hit("sft_sample_id_missing", file_name, locator, detail="缺少 sample_id")
    if not isinstance(messages, list) or not messages:
        findings.hit("sft_messages_missing", file_name, locator, detail="缺少 messages")
        return
    dist["message_count"].append(len(messages))
    if len(messages) < 2:
        findings.hit("sft_messages_too_short", file_name, locator, detail=f"n={len(messages)}")

    roles = [str(m.get("role") or "") if isinstance(m, dict) else "" for m in messages]
    contents = [str(m.get("content") or "") if isinstance(m, dict) else "" for m in messages]
    if roles[0] != "user":
        findings.hit("sft_first_message_not_user", file_name, locator, detail=f"first={roles[0]}")
    unexpected = {role for role in roles if role not in ("user", "assistant")}
    if unexpected:
        findings.hit("sft_role_alternation_invalid", file_name, locator,
                     detail=f"非法角色 {sorted(unexpected)}")
    breaks = [i for i in range(1, len(roles)) if roles[i] == roles[i - 1]]
    if breaks:
        findings.hit("sft_role_alternation_invalid", file_name, locator,
                     f"{roles[breaks[0]-1]}->{roles[breaks[0]]}",
                     detail=f"{len(breaks)} 处相邻同角色")
    if roles[-1] == "user":
        findings.hit("sft_last_message_is_user", file_name, locator, detail=f"n={len(messages)}")

    empty = [i for i, text in enumerate(contents) if not text.strip()]
    if empty:
        findings.hit("sft_empty_content", file_name, locator,
                     detail=f"{len(empty)} 条空内容 index={empty[:3]}")

    if roles[0] == "user":
        dist["user_chars"].append(len(contents[0]))
        register_sft_question(state, contents[0], split, locator)

    for role, text in zip(roles, contents):
        if role != "assistant":
            continue
        dist["assistant_chars"].append(len(text))
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            findings.hit("sft_assistant_invalid_json", file_name, locator, text)
            continue
        if not isinstance(payload, dict):
            findings.hit("sft_assistant_invalid_json", file_name, locator, text,
                         detail=f"JSON 顶层是 {type(payload).__name__}")
            continue
        action = payload.get("action")
        message = payload.get("message")
        if "action" not in payload or "message" not in payload:
            findings.hit("sft_assistant_missing_key", file_name, locator, text,
                         detail=f"keys={sorted(payload)}")
        if action not in ACTIONS:
            findings.hit("sft_action_invalid", file_name, locator, text, detail=f"action={action}")
        else:
            state.sft_action_totals[(state.current_file, str(action))] += 1
        if not str(message or "").strip():
            findings.hit("sft_message_empty", file_name, locator, text)

    if not is_blank(sample_id):
        state.sft_sample_id_seen[(split, str(sample_id))] += 1
        state.sft_sample_splits[str(sample_id)][split] += 1
    fingerprint = sha(json.dumps(messages, ensure_ascii=False, sort_keys=True), 20)
    state.sft_fp_splits[fingerprint][split] += 1


# --------------------------------------------------------------------------
# cross split and duplicate analysis


def analyse_split_leakage(state: AuditState) -> dict:
    findings = state.findings
    metrics: dict = {}

    cross_cases = sorted(case_id for case_id, per in state.case_splits.items() if len(per) > 1)
    for case_id in cross_cases:
        per = state.case_splits[case_id]
        findings.hit("case_id_cross_split", "(cross-split)", f"case_id={case_id}",
                     case_id, detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
    metrics["case_id"] = {
        "distinct": len(state.case_splits),
        "in_multiple_splits": len(cross_cases),
        "examples_of_keys": cross_cases[:5],
    }

    exact_cross = 0
    exact_cross_any_len = 0
    for flat, per in state.q1_splits.items():
        if len(per) < 2:
            continue
        exact_cross_any_len += 1
        if len(flat) < MIN_LEAK_QUESTION_CHARS:
            continue
        exact_cross += 1
        findings.hit("first_question_cross_split", "(cross-split)",
                     f"q_sha={sha(flat, 20)} members={state.q1_members.get(flat, [])}",
                     state.q1_text.get(flat, flat),
                     detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
    repeated = sum(1 for per in state.q1_splits.values() if sum(per.values()) > 1)
    metrics["first_question"] = {
        "groups_repeated_any_split": repeated,
        "distinct": len(state.q1_splits),
        "in_multiple_splits_any_length": exact_cross_any_len,
        "in_multiple_splits_substantive": exact_cross,
        "min_chars_for_substantive": MIN_LEAK_QUESTION_CHARS,
    }

    near_cross = 0
    near_pairs = 0
    for head, per in state.head_splits.items():
        if len(per) < 2:
            continue
        if len(state.head_questions[head]) < 2:
            continue        # identical questions are counted as exact duplicates
        near_pairs += 1
        text = state.head_text.get(head, "")
        if len(normalize(text)) < MIN_LEAK_QUESTION_CHARS:
            continue
        near_cross += 1
        findings.hit("near_duplicate_cross_split", "(cross-split)", f"head_sha={head}", text,
                     detail=f"splits={ {k: v for k, v in sorted(per.items())} } variants={len(state.head_questions[head])}")
    metrics["near_duplicate"] = {
        "distinct_prefix": len(state.head_splits),
        "prefix_in_multiple_splits": near_cross,
        "prefix_pairs": near_pairs,
    }

    within = cross = 0
    for fingerprint, per in state.episode_fp_splits.items():
        if sum(per.values()) < 2:
            continue
        if len(per) > 1:
            cross += 1
            findings.hit("episode_duplicate_cross_split", "(cross-split)", f"fp={fingerprint}",
                         detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
        else:
            within += 1
            findings.hit("episode_duplicate_within_split", "(cross-split)", f"fp={fingerprint}",
                         detail="split=" + next(iter(per)))
    metrics["episode_duplicate"] = {
        "distinct_fingerprints": len(state.episode_fp_splits),
        "duplicated_within_split": within,
        "duplicated_across_split": cross,
    }

    conclusion_within = conclusion_cross = 0
    for key, per in state.conclusion_splits.items():
        if sum(per.values()) < 2:
            continue
        text = state.conclusion_text.get(key, key)
        if len(per) > 1:
            conclusion_cross += 1
            findings.hit("conclusion_duplicate_cross_split", "(cross-split)",
                         f"fp={key} members={state.conclusion_members.get(key, [])}", text,
                         detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
        else:
            conclusion_within += 1
            findings.hit("conclusion_duplicate_within_split", "(cross-split)", f"fp={key}", text,
                         detail="split=" + next(iter(per)))
    promo_within = sum(1 for per in state.promo_splits.values() if sum(per.values()) > 1 and len(per) == 1)
    promo_cross = sum(1 for per in state.promo_splits.values() if len(per) > 1)
    metrics["conclusion_promo_duplicate"] = {
        "distinct_promo_conclusions": len(state.promo_splits),
        "promo_duplicated_within_split": promo_within,
        "promo_duplicated_across_split": promo_cross,
        "note": "推广模板重复单独计数，不混入实质性重复结局",
    }
    metrics["conclusion_duplicate"] = {
        "distinct_long_conclusions": len(state.conclusion_splits),
        "duplicated_within_split": conclusion_within,
        "duplicated_across_split": conclusion_cross,
    }

    sft_cross = sum(1 for per in state.sft_sample_splits.values() if len(per) > 1)
    for sample_id, per in state.sft_sample_splits.items():
        if len(per) > 1:
            findings.hit("sft_sample_id_cross_split", "(cross-split)", f"sample_id={sample_id}",
                         detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
    sft_q_cross = 0
    for flat, per in state.sft_q1_splits.items():
        if len(per) < 2 or len(flat) < MIN_LEAK_QUESTION_CHARS:
            continue
        sft_q_cross += 1
        findings.hit("sft_first_question_cross_split", "(cross-split)",
                     f"q_sha={sha(flat, 20)} members={state.sft_q1_members.get(flat, [])}",
                     state.sft_q1_text.get(flat, flat),
                     detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
    sft_within = sft_fp_cross = 0
    for fingerprint, per in state.sft_fp_splits.items():
        if sum(per.values()) < 2:
            continue
        if len(per) > 1:
            sft_fp_cross += 1
            findings.hit("sft_duplicate_cross_split", "(cross-split)", f"fp={fingerprint}",
                         detail=f"splits={ {k: v for k, v in sorted(per.items())} }")
        else:
            sft_within += 1
            findings.hit("sft_duplicate_within_split", "(cross-split)", f"fp={fingerprint}",
                         detail="split=" + next(iter(per)))
    metrics["sft"] = {
        "distinct_sample_ids": len(state.sft_sample_splits),
        "sample_id_in_multiple_splits": sft_cross,
        "first_question_in_multiple_splits": sft_q_cross,
        "records_duplicated_within_split": sft_within,
        "records_duplicated_across_split": sft_fp_cross,
    }

    shared = set(state.q1_splits) & set(state.sft_q1_splits)
    conflicts = 0
    compared = 0
    for flat in shared:
        if len(flat) < MIN_LEAK_QUESTION_CHARS:
            continue
        compared += 1
        left = set(state.q1_splits[flat])
        right = set(state.sft_q1_splits[flat])
        if left != right:
            conflicts += 1
            findings.hit("cross_artifact_split_conflict", "(cross-split)",
                         f"q_sha={sha(flat, 20)}", state.q1_text.get(flat, flat),
                         detail=f"parquet={sorted(left)} sft={sorted(right)}")
    metrics["cross_artifact"] = {
        "shared_questions": len(shared),
        "compared_substantive": compared,
        "split_conflicts": conflicts,
    }

    for split, sample_id in state.sample_id_seen.items():
        if sample_id > 1:
            findings.hit("id_duplicate_within_split", "(cross-split)",
                         f"split={split[0]} sample_id={split[1]}",
                         detail=f"出现 {sample_id} 次")
    for split, sample_id in state.sft_sample_id_seen.items():
        if sample_id > 1:
            findings.hit("sft_duplicate_within_split", "(cross-split)",
                         f"split={split[0]} sample_id={split[1]}", detail=f"出现 {sample_id} 次")
    return metrics


# --------------------------------------------------------------------------
# file drivers


def file_report(path: Path, kind: str, split: str, state: AuditState, rows_total: int,
                rows_scanned: int, dist: dict, notes=None) -> dict:
    name = path.name
    rules = {rule: hits[name] for rule, hits in state.findings.files_hit.items() if hits.get(name)}
    empty = {}
    for (file_name, field), total in sorted(state.field_total.items()):
        if file_name != name:
            continue
        blank = state.field_blank.get((file_name, field), 0)
        empty[field] = {"blank": blank, "total": total,
                        "rate": round(blank / total, 6) if total else None}
    actions = {key: value for (file_name, key), value in state.action_totals.items() if file_name == name}
    sft_actions = {key: value for (file_name, key), value in state.sft_action_totals.items() if file_name == name}
    sources = {key: value for (file_name, key), value in state.data_sources.items() if file_name == name}

    for key, values in dist.items():
        state.global_dist[key].extend(values)
    report = {
        "name": name,
        "kind": kind,
        "split": split,
        "bytes": path.stat().st_size,
        "rows_total": rows_total,
        "rows_scanned": rows_scanned,
        "sample_rate": state.rate,
        "rules": rules,
        "field_empty": empty,
        "action_totals": actions,
        "sft_action_totals": sft_actions,
        "data_sources": sources,
        "distribution": {
            key: quantiles(values) for key, values in sorted(dist.items())
        },
        "notes": notes or [],
    }
    return report


def audit_parquet_file(path: Path, split: str, state: AuditState) -> dict:
    import pyarrow.parquet as pq

    state.current_file = path.name
    reader = pq.ParquetFile(path)
    rows_total = reader.metadata.num_rows
    dist: dict = defaultdict(list)
    audited = 0
    position = 0
    for batch in reader.iter_batches(batch_size=4000):
        columns = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        for offset in range(batch.num_rows):
            row_no = position + offset
            if not state.sampler.keep(f"{split}:{path.name}:{row_no}"):
                continue
            record = {name: columns[name][offset] for name in columns}
            audit_episode(record, split, path.name, row_no, state, dist)
            audited += 1
        position += batch.num_rows
    check_schema(state, path.name, reader.schema_arrow.names)
    notes = [f"顶层字段 {len(reader.schema_arrow)} 个，每批 4000 行流式读取"]
    return file_report(path, "episode_parquet", split, state, rows_total, audited, dist, notes)


def audit_jsonl_file(path: Path, kind: str, split: str, state: AuditState) -> dict:
    state.current_file = path.name
    dist: dict = defaultdict(list)
    rows_total = 0
    audited = 0
    bad_lines = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                state.findings.hit("file_unreadable", path.name, f"line={line_no}", line,
                                   detail="该行不是合法 JSON")
                continue
            if not state.sampler.keep(f"{split}:{path.name}:{line_no}"):
                continue
            audited += 1
            if kind == "sft_jsonl":
                audit_sft(record, split, path.name, line_no, state, dist)
            else:
                audit_episode(record, split, path.name, line_no - 1, state, dist)
    notes = [f"无法解析的行 {bad_lines} 行"]
    return file_report(path, kind, split, state, rows_total, audited, dist, notes)


# --------------------------------------------------------------------------
# discovery


def discover(input_dir: Path, include_sample_shards: bool):
    accepted = []
    skipped = []
    for path in sorted(input_dir.iterdir()):
        if path.is_dir():
            skipped.append({"name": path.name + "/",
                            "reason": "子目录，本工具只审目录顶层的产物文件"})
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".parquet", ".jsonl"):
            skipped.append({"name": path.name, "reason": "不是 parquet/jsonl 产物"})
            continue
        if path.name.startswith("_") or path.name.startswith("."):
            skipped.append({"name": path.name, "reason": "下划线/点前缀的旁挂文件（报告、缓存）"})
            continue
        stem = path.stem
        if not include_sample_shards and ("_sample" in stem or "_tiny" in stem):
            skipped.append({"name": path.name,
                            "reason": "抽样 shard，派生自 split 主文件，计入会重复"})
            continue
        first = read_jsonl_first_record(path) if path.suffix.lower() == ".jsonl" else None
        kind = classify(path, first)
        if kind in ("unknown", "corpus_jsonl"):
            skipped.append({"name": path.name, "reason": f"不是训练产物（识别为 {kind}）"})
            continue
        accepted.append((path, kind))
    return accepted, skipped


EPISODE_REQUIRED_TOP = ("prompt", "ground_truth", "extra_info", "data_source", "index")
EPISODE_REQUIRED_NESTED = {
    "ground_truth": ("reference_conclusion", "diagnosis", "presenting_problem", "department"),
    "extra_info": ("sample_id", "case_id", "patient_state", "profile"),
}


def check_schema(state: AuditState, file_name: str, field_names) -> None:
    """Declared schema must still carry every field the pipeline depends on."""
    present = set(field_names)
    for required in EPISODE_REQUIRED_TOP:
        if required not in present:
            state.findings.hit("schema_missing_field", file_name, "schema",
                               detail=f"缺少顶层字段 {required}")


def split_of_name(name: str) -> str:
    stem = Path(name).stem
    for split in SPLIT_NAMES:
        if stem == split or stem.startswith(split + ".") or stem.startswith(split + "_"):
            return split
    return stem


# --------------------------------------------------------------------------
# reporting

BACKTICK = chr(96)


def code(text: str) -> str:
    """Markdown code span without embedding a literal backtick in the source."""
    return BACKTICK + str(text) + BACKTICK


def severity_rank(rule: str) -> int:
    severity = RULES.get(rule, ("medium", ""))[0]
    return SEVERITY_RANK.get(severity, 0)


def summary_block(payload: dict) -> dict:
    counts = {rule: item["count"] for rule, item in payload["checks"].items()}
    ordered = sorted(counts.items(), key=lambda kv: (-severity_rank(kv[0]), -kv[1], kv[0]))
    blocking = [rule for rule, _ in ordered if severity_rank(rule) >= SEVERITY_RANK["blocker"]]
    return {
        "rules_hit": len(counts),
        "total_hits": sum(counts.values()),
        "blocker_rules": blocking,
        "blocker_hits": sum(counts[rule] for rule in blocking),
        "ranking": [{"rule": rule, "severity": RULES.get(rule, ("medium", ""))[0],
                     "count": count} for rule, count in ordered],
    }


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    lines: list = []
    add = lines.append
    fence = BACKTICK * 3

    add("# 训练产物独立审计基线")
    add("")
    add(f"- 生成时间：{meta['generated_at']}")
    add(f"- 工具：{code(meta['tool'])}（版本 {meta['version']}）")
    add(f"- 输入目录：{code(meta['input'])}")
    add(f"- 随机种子：{code(meta['seed'])}；抽样率：{meta['sample_rate']}"
        "（1.0 表示全量扫描）")
    add("- 抽样为哈希抽样：按 sha256(seed:文件:行号) 取模，同参数重跑得到同一子集")
    add("")

    add("## 一、文件总览")
    add("")
    add("| 文件 | 类型 | split | 行数 | 实际审计行数 | 命中规则数 |")
    add("| --- | --- | --- | ---: | ---: | ---: |")
    for item in payload["files"]:
        add(f"| {code(item['name'])} | {item['kind']} | {item['split']} | "
            f"{item['rows_total']:,} | {item['rows_scanned']:,} | {len(item['rules'])} |")
    add("")
    if payload["skipped_files"]:
        add("跳过的文件（逐条记录原因，未参与统计）：")
        add("")
        for item in payload["skipped_files"]:
            add(f"- {code(item['name'])}：{item['reason']}")
        add("")

    add("## 二、检查项实测结果")
    add("")
    if payload["checks"]:
        add("| 规则码 | 严重度 | 命中数 | 说明 |")
        add("| --- | --- | ---: | --- |")
        for entry in payload["summary"]["ranking"]:
            detail = payload["checks"][entry["rule"]]
            add(f"| {code(entry['rule'])} | {entry['severity']} | {entry['count']:,} | "
                f"{detail['description']} |")
    else:
        add("所有检查项均为 0 命中。")
    add("")

    add("## 三、分布")
    add("")
    add("| 指标 | p50 | p90 | p99 | 均值 | 最大值 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key, value in sorted(payload["distribution"].items()):
        add(f"| {key} | {value['p50']} | {value['p90']} | {value['p99']} | "
            f"{value['mean']} | {value['max']} |")
    add("")
    add("各文件的动作计数与来源：")
    add("")
    for item in payload["files"]:
        if item["action_totals"]:
            add(f"- {code(item['name'])} 动作计数：{item['action_totals']}")
        if item["sft_action_totals"]:
            add(f"- {code(item['name'])} SFT 动作计数：{item['sft_action_totals']}")
        if item["data_sources"]:
            add(f"- {code(item['name'])} data_source：{item['data_sources']}")
    add("")
    add("分母指标（用于解释上面的比率）：")
    add("")
    for key, value in sorted(payload["metrics"].items()):
        add(f"- {key}：{value:,}")
    add("")
    add("跨 split 结构指标：")
    add("")
    for name, block in sorted(payload["cross_split"].items()):
        for key, value in sorted(block.items()):
            if isinstance(value, list):
                rendered = ", ".join(str(v) for v in value) if value else "无"
            else:
                rendered = value
            add(f"- {name}.{key}：{rendered}")
    add("")

    add("## 四、字段空值率")
    add("")
    add("| 文件 | 字段 | 空值 | 总数 | 空值率 |")
    add("| --- | --- | ---: | ---: | ---: |")
    for item in payload["files"]:
        for field, stat in sorted(item["field_empty"].items()):
            rate = stat["rate"]
            shown = "n/a" if rate is None else f"{rate:.4%}"
            add(f"| {code(item['name'])} | {field} | {stat['blank']:,} | "
                f"{stat['total']:,} | {shown} |")
    add("")

    add("## 五、最严重的具体问题（按严重度与命中量排序，各附一条原文证据）")
    add("")
    shown = 0
    for entry in payload["summary"]["ranking"]:
        if shown >= 5:
            break
        rule = entry["rule"]
        item = payload["checks"][rule]
        example = item["examples"][0] if item["examples"] else None
        add(f"### {shown + 1}. {code(rule)}（{item['severity']}，命中 {item['count']:,} 次）")
        add("")
        add(f"- 说明：{item['description']}")
        add(f"- 涉及文件：{item['files']}")
        if example:
            add(f"- 定位：{code(example['file'])} {example['locator']}"
                f"（内容 sha256 前缀 {example['text_sha']}）")
            if example["snippet"]:
                add(f"- 原文证据：{code(example['snippet'].replace('|', ' '))}")
            if example.get("detail"):
                add(f"- 细节：{example['detail']}")
        add("")
        shown += 1

    add("## 六、复跑方法")
    add("")
    add(fence)
    add("python scripts/audit_training_files.py --input <产物目录> --output data/audit_v3/<名字>.json --report-md docs/audit/<名字>.md")
    add(fence)
    add("")
    add(f"默认种子 {payload['meta']['seed']}、默认抽样率 {payload['meta']['sample_rate']}。"
        "两个 JSON 的同名键可直接相减做前后对比。")
    add("")
    return "\n".join(lines) + "\n"


def console_summary(payload: dict) -> str:
    meta = payload["meta"]
    out = [f"input={meta['input']} seed={meta['seed']} rate={meta['sample_rate']}"]
    for item in payload["files"]:
        out.append(f"  {item['name']:<22} rows={item['rows_total']:>8,} "
                   f"audited={item['rows_scanned']:>8,} rules={len(item['rules'])}")
    for entry in payload["summary"]["ranking"]:
        out.append(f"  [{entry['severity']:<7}] {entry['rule']:<38} {entry['count']:,}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# entry point


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit training artifacts (parquet and jsonl) without modifying them.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="directory holding train/dev/test.parquet or *.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="path of the JSON report to write")
    parser.add_argument("--report-md", type=Path, default=None,
                        help="optional path of a Chinese markdown report to write")
    parser.add_argument("--append-md", type=Path, default=None,
                        help="markdown file appended verbatim after the generated report, "
                             "so a hand written judgment survives a rerun")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
                        help="1.0 audits every row; smaller values use hash sampling")
    parser.add_argument("--max-examples", type=int, default=5,
                        help="examples kept per rule for the report")
    parser.add_argument("--include-sample-shards", action="store_true",
                        help="also audit *_sample*.jsonl shards")
    parser.add_argument("--strict-exit", action="store_true",
                        help="exit 1 when a blocker rule has any hit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists() or not args.input.is_dir():
        print(f"input directory not found: {args.input}", file=sys.stderr)
        return 2
    if not 0 < args.sample_rate <= 1.0:
        print("--sample-rate must be in (0, 1]", file=sys.stderr)
        return 2

    state = AuditState(args.seed, args.sample_rate, args.max_examples)
    accepted, skipped = discover(args.input, args.include_sample_shards)
    state.skipped = skipped
    if not accepted:
        print(f"no training artifact found under {args.input}", file=sys.stderr)
        return 2

    for path, kind in accepted:
        split = split_of_name(path.name)
        if kind == "episode_parquet":
            report = audit_parquet_file(path, split, state)
        else:
            report = audit_jsonl_file(path, kind, split, state)
        state.files.append(report)
        if not args.quiet:
            print(f"[done] {path.name} rows={report['rows_total']:,} "
                  f"audited={report['rows_scanned']:,}", file=sys.stderr)

    cross = analyse_split_leakage(state)
    payload = {
        "meta": {
            "tool": "scripts/audit_training_files.py",
            "version": AUDIT_VERSION,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "input": str(args.input),
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "python": sys.version.split()[0],
        },
        "files": state.files,
        "cross_split": cross,
        "distribution": {key: quantiles(values) for key, values in sorted(state.global_dist.items())},
        "metrics": dict(sorted(state.metrics.items())),
        "checks": state.findings.to_dict(),
        "skipped_files": skipped,
    }
    payload["summary"] = summary_block(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        document = render_markdown(payload)
        if args.append_md:
            extra = args.append_md.read_text(encoding="utf-8").strip()
            if extra:
                document = document.rstrip() + "\n\n" + extra + "\n"
        args.report_md.write_text(document, encoding="utf-8")
    if not args.quiet:
        print(console_summary(payload))
        print(f"written: {args.output}"
              + (f" and {args.report_md}" if args.report_md else ""))
    if args.strict_exit and payload["summary"]["blocker_rules"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
