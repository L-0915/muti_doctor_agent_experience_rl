"""Rule engine shared by the v3 census and the v3 cleaner.

One implementation, two consumers. The census reports what each rule would
hit over the whole corpus; the cleaner applies the same functions and records
every change. Sharing the engine is what keeps the two number sets comparable:
a rule cannot drift between the measurement and the fix.

Groups follow the task book, extended by the independent audit's blocker list:

  S1 drop    D01..D11   the whole consultation leaves the corpus
  S2 repair  R01..R14   a turn or annotation is rewritten in place
  S3 flag    F01..F14   the case stays and carries a quality marker

The v2 patterns in scripts/clean_cases.py and the episode semantics in
scripts/build_training_data.py are imported rather than re-derived: the
promotion patterns were tuned against this corpus, and the conclusion is the
reward reference, so there must be exactly one definition of it.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402  conclusion_of / split_of / labels
import clean_cases as V2  # noqa: E402  v2 patterns, reused not re-derived

MAX_DOCTOR_TURNS = B.DEFAULT_MAX_DOCTOR_TURNS
MIN_DOCTOR_TURNS = 2
IMAGE_PLACEHOLDER = "<IMAGE_UNAVAILABLE>"

DROP_RULES = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09",
              "D10", "D11")
S2_RULES = ("R01", "R02", "R03", "R05", "R07", "R08", "R09", "R11", "R13", "R14")
FLAG_RULES = ("F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10", "F11",
              "F12", "F14")

# The independent audit's rule code for "first turn is not the patient" is D15;
# this engine implements it as D08 (see RULE_ALIASES) and drops rather than
# shifting the window forward, because the RL prompt is the opening patient
# turn and a mid-dialogue turn cannot stand in for a chief complaint.
RULE_ALIASES = {"D15": "D08"}

# A marker that makes the supervised demonstration unreliable or the
# environment unfair blocks the SFT copy. F05 (the conclusion is not the last
# turn), F08 (a doctor repeating a question), F10 (mixed script), F12 (a
# topic label that is really a department) and F14 (no conclusion turn at all,
# so there is no reward reference) describe how the text reads or how its
# labels are shaped, not whether the decision behind it can be trusted, so they
# only annotate: the case still serves as a consultation state to condition on.
BLOCKING_FLAGS = frozenset({"F01", "F02", "F03", "F04", "F06", "F07", "F11"})

# A doctor turn this short carries no clinical decision; three of them, or a
# doctor who never says anything longer, is a conversation with no supervision
# in it. One "嗯" is how people talk.
SHORT_DOCTOR_TURNS = 3
SHORT_TURN_CHARS = 3

# R14: the patient's opening message is treated as spliced when it shares this
# many consecutive characters with a later patient turn, and a remainder this
# short is no longer a chief complaint.
SPLICE_OVERLAP = 12
SPLICE_MIN_KEEP = 8

# D11: the conclusion reproduces the patient's opening message verbatim. A
# physician answering "细菌性阴道炎需要用药清洗阴道吗" with "细菌性阴道炎需要用
# 药清洗阴道的" is answering the question, not leaking it, so a shared run of
# a few characters proves nothing; only an opening this long contained whole in
# the conclusion (or an identical pair) is the leak the audit measured.
LEAK_OPENING_MIN_CHARS = 20

RULE_TITLES = {
    "D01": "有效轮少于 2 轮",
    "D02": "缺少患者发言",
    "D03": "隐私信息命中（手机号/身份证/邮箱/微信/QQ/银行卡）",
    "D04": "非中文占比超过 60% 或含乱码",
    "D05": "含 U+FFFD 或不可打印控制字符",
    "D06": "患者首问无实质文本",
    "D07": "与其它对话重复（按轮次文本序列指纹）",
    "D08": "会话切分残段（首轮不是患者发言，独立审计记作 D15）",
    "D09": "无任何医生发言",
    "D10": "所有医生轮都短于 8 字（无实质医疗内容）",
    "D11": "结局照抄患者首问原文",
    "R01": "医生轮残留平台模板/索评/联系方式/推广",
    "R02": "图片占位统一为 <IMAGE_UNAVAILABLE>",
    "R03": "同轮内重复句去重",
    "R05": "删除 URL 与站内链接",
    "R07": "空白/全角半角/零宽字符/连续标点规范化",
    "R08": "删除时间戳/署名/匿名标记",
    "R09": "标注中 name 为 undefined 或空的实体",
    "R11": "患者轮出现医生口吻（进人工队列，不改角色）",
    "R13": "平台占位（语音/撤回/仅自己可见）删除",
    "R14": "患者首问与后续患者轮拼接泄漏，截断",
    "F01": "医生轮少于 2 轮就给出结论",
    "F02": "医生轮超过 20 轮",
    "F03": "对话依赖图片",
    "F04": "医生轮可疑（全程过短或多次敷衍）",
    "F05": "结局不是末轮",
    "F06": "患者全程只有一句实质发言",
    "F07": "单个 session 内出现多个主诉",
    "F08": "医生连续重复同一问题至少 2 次",
    "F10": "中英混杂",
    "F11": "总轮数少于 3",
    "F12": "presenting_problem 实为科室名",
    "F14": "无实质结论轮（保留为问诊数据，不进结局奖励）",
}


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def input_files(input_dir) -> list:
    """Corpus files only. A leading underscore marks bookkeeping this project
    writes next to the corpus (_report.json, _ledger.jsonl), never a source."""
    return sorted(p for p in Path(input_dir).glob("*.jsonl")
                  if not p.name.startswith("_"))


def iter_rows(input_dir, limit: int = 0):
    seen = 0
    for path in input_files(input_dir):
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                yield path, line_no, json.loads(line)
                seen += 1
                if limit and seen >= limit:
                    return


def turns_of(row: dict) -> list:
    return [t for t in (row.get("turns") or []) if isinstance(t, dict)]


def texts_of(turns) -> list:
    return [str(t.get("text") or "") for t in turns]


# ---------------------------------------------------------------- measurement

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
SUBSTANTIVE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

PLACEHOLDER = re.compile(
    r"\[\s*(?:图片|照片|图|image|img)\s*\]"
    r"|【\s*(?:图片|照片|图)\s*】"
    r"|[(（]\s*(?:图片|照片|图)\s*[)）]"
    r"|<图片>|<\s*IMAGE_UNAVAILABLE\s*>", re.I)

# Media the platform blanks out. Text that survives them is what the patient
# actually said; text that does not is a turn with nothing in it.
PLATFORM_BLANK = re.compile(
    r"语音因?隐私问题?无法显示|因隐私问题无法显示|语音无法(?:播放|显示)|听不清(?:语音|您说)"
    r"|(?:图片|照片|视频)无法显示|该图片.{0,4}已失效"
    r"|该内容仅自己可见|仅自己可见|消息已撤回|撤回了一条消息|对方撤回"
    r"|对方正在输入|该用户已注销|\[(?:语音|视频|表情|位置|名片|红包|小程序)\]")

MOJIBAKE = re.compile(r"锟斤拷|ï¿½|[ÃÂâ€]{3,}|[\u0080-\u00ff]{4,}")


def substantive_len(text: str) -> int:
    """Letters and digits only: punctuation, emoji, blanked media and
    placeholders carry nothing a patient can be asked about."""
    return len(SUBSTANTIVE.findall(PLATFORM_BLANK.sub(" ", PLACEHOLDER.sub(" ", text))))


def spoken_len(text: str) -> int:
    """What the turn says, whitespace and blanked media removed but punctuation
    kept. This is the measure the independent audit used for "8 字" when it
    separated a physician who said nothing from one who said something and
    never concluded, so D10 reads it rather than substantive_len."""
    body = B.PLATFORM_PLACEHOLDER.sub(" ", str(text or ""))
    return len(re.sub(r"\s+", "", body))


def _ratio(text: str) -> tuple:
    cjk = len(CJK.findall(text))
    latin = len(LATIN.findall(text))
    return cjk, latin, len(re.sub(r"\s", "", text))


def rel(path) -> str:
    """Repo-relative when inside the project, so a report travels."""
    try:
        return str(Path(path).resolve().relative_to(B.ROOT))
    except ValueError:
        return str(path)


# The threshold is the one the task book fixes: a case is not Chinese when
# fewer than 40% of its characters are Han. The denominator is letters only:
# digits are language-neutral, and counting them would condemn a short case
# that is mostly lab values ("血糖5.6 血压120/80") as non-Chinese.
CJK_MIN_RATIO = 0.40
LETTERS_MIN = 10


def letters_ratio(texts) -> tuple:
    """(Han share of letters or None if there are too few letters, cjk, latin).

    Placeholders are stripped first: <IMAGE_UNAVAILABLE> is Latin script this
    pipeline wrote itself, and counting it as English would make a repaired
    case look like mixed-language text.
    """
    cjk, latin, _ = _ratio(" ".join(PLACEHOLDER.sub(" ", text) for text in texts))
    letters = cjk + latin
    if letters < LETTERS_MIN:
        return None, cjk, latin
    return cjk / letters, cjk, latin


# ----------------------------------------------------- first-ask grouping keys
#
# Two cases that open with the same question must land in the same split, or
# the evaluation set is the training set with different punctuation. The exact
# key catches identical openings; the window keys catch openings that differ
# only in an insertion or a suffix.

FIRST_ASK_MAX = 120
FIRST_ASK_WINDOW = 40
FIRST_ASK_STRIDE = 40
PUNCT_ALL = re.compile(r"[\s，。、；：！？,.!?;:（）()\[\]{}【】<>《》\-—_~～·＊*\"'“”‘’…]+")


def first_ask_text(turns) -> str:
    """The patient's opening message with punctuation and whitespace removed —
    the text the RL prompt will hand the policy."""
    for turn in turns:
        if str(turn.get("role") or "") == "patient":
            return PUNCT_ALL.sub("", str(turn.get("text") or ""))[:FIRST_ASK_MAX]
    return ""


def first_ask_keys(turns) -> set:
    """Exact opening key plus 40-character windows at fixed offsets."""
    text = first_ask_text(turns)
    if not text:
        return set()
    keys = {text}
    if len(text) > FIRST_ASK_WINDOW:
        for start in range(0, len(text) - FIRST_ASK_WINDOW + 1, FIRST_ASK_STRIDE):
            keys.add(text[start:start + FIRST_ASK_WINDOW])
    return keys


# One opening being the prefix of another is the other way two near-duplicate
# questions show up (the shorter message simply stops earlier). The floor keeps
# a greeting like "你好" from merging thousands of unrelated cases.
FIRST_ASK_PREFIX_MIN = 20


def prefix_merges(texts, floor: int = FIRST_ASK_PREFIX_MIN):
    """(short, long) pairs where one opening is a prefix of another."""
    ordered = sorted(set(texts))
    for text in ordered:
        if len(text) < floor or len(text) >= FIRST_ASK_MAX:
            continue
        start = bisect.bisect_left(ordered, text)
        end = bisect.bisect_left(ordered, text + "\uffff")
        for longer in ordered[start:end]:
            if longer != text:
                yield text, longer


class UnionFind:
    """Case grouping for split assignment: one question, one split."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> dict:
        out: dict = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return out


# --------------------------------------------------------------------- S1 drop
#
# Each gate below is a cheap pre-test: the detailed patterns only run when the
# gate shows something that could match them. Without gates a full corpus pass
# costs six regexes per turn instead of one.

PRIVACY_GATE = re.compile(r"\d{7,}|[@＠]|微信|[Qq]{2}|wx|vx|V信|薇信|银行卡|身份证")
PRIVACY_PATTERNS = (
    ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9](?:[- ]?\d){9}(?!\d)")),
    ("id_card", re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("wechat_qq", re.compile(
        r"(?:微信|薇信|V信|[Qq]{2}|wx|vx)\s*(?:号|账号)?\s*[:：]?\s*[A-Za-z0-9_-]{5,20}")),
    ("bank_card", re.compile(r"(?<!\d)(?:\d[ ]?){16,19}(?!\d)")),
    ("landline", re.compile(r"(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)")),
)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")


def shares_run(left: str, right: str, size: int) -> bool:
    """True when two texts share an unbroken run of "size" alphanumeric
    characters — the audit's operational definition of duplicated content."""
    def clean(text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text)

    a, b = clean(left), clean(right)
    if len(a) < size or len(b) < size:
        return bool(a) and a == b
    grams = {a[i:i + size] for i in range(len(a) - size + 1)}
    return any(b[i:i + size] in grams for i in range(len(b) - size + 1))


def normalized(text: str) -> str:
    """Letters and digits only, so punctuation and spacing cannot hide a copy."""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))


def leaks_opening(opening: str, conclusion: str) -> bool:
    """True when the opening message and the conclusion are the same text under
    the punctuation: identical, a long opening contained whole in the conclusion
    (the patient asked and the physician repeated the question), or a conclusion
    contained whole in the opening (the answer is already in the prompt, so the
    reward teaches copying)."""
    left, right = normalized(opening), normalized(conclusion)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= LEAK_OPENING_MIN_CHARS and left in right:
        return True
    return len(right) >= B.MIN_CONCLUSION_CHARS and right in left


def inspect_drops(row: dict, dup_of=None, texts=None, sub=None) -> dict:
    """Every S1 rule this case hits, as {rule_code: detail}.

    The cleaner drops on the first rule in DROP_RULES order and records that
    one in the ledger; the census counts all of them, because "how many cases
    are dirty" and "how many cases leave" are different numbers.

    texts / sub are optional precomputed values; analyse() passes them so a
    case is measured once instead of once per rule.
    """
    turns = turns_of(row)
    if texts is None:
        texts = texts_of(turns)
    if sub is None:
        sub = [substantive_len(text) for text in texts]
    hits: dict = {}

    hits.update(structural_drops(turns, texts, sub))

    privacy = hitting_privacy(texts)
    if privacy:
        hits["D03"] = "|".join(sorted(privacy))

    ratio, _, _ = letters_ratio(texts)
    if ratio is not None and ratio < CJK_MIN_RATIO:
        hits["D04"] = f"cjk_ratio={ratio:.2f}"
    elif MOJIBAKE.search(" ".join(texts)):
        hits["D04"] = "mojibake"

    for index, text in enumerate(texts):
        if CONTROL_CHARS.search(text):
            hits["D05"] = f"turn_index={index}"
            break

    first_patient = next((i for i, t in enumerate(turns)
                          if str(t.get("role") or "") == "patient"), None)
    if first_patient is not None and sub[first_patient] < 2:
        hits["D06"] = f"turn_index={first_patient}"

    if dup_of is not None:
        digest = fingerprint(turns)
        first = dup_of.get(digest)
        own = case_key(row)
        if first is not None and first != own:
            hits["D07"] = f"duplicate_of={first}"

    conclusion = B.conclusion_of(turns) if turns else None
    if conclusion is not None and texts \
            and leaks_opening(texts[0], texts[conclusion]):
        hits["D11"] = f"conclusion_turn={conclusion}"

    return hits


def structural_drops(turns, texts=None, sub=None) -> dict:
    """D01/D02/D08/D09/D10 — the shape of the dialogue itself.

    D02 is implemented as "no patient turn" and D09 as "no doctor turn" so
    that each case is attributed to exactly one rule; the task book words D02
    as "missing patient or doctor", and this split covers that set with D09.

    These are re-checked after repair, because emptying a turn can leave a case
    with no doctor left, with a doctor turn where the opening used to be — the
    opening turn is what the RL prompt is built from — or with no substantive
    doctor turn at all.
    """
    if texts is None:
        texts = texts_of(turns)
    if sub is None:
        sub = [substantive_len(text) for text in texts]
    roles = [str(t.get("role") or "") for t in turns]
    hits: dict = {}
    effective = sum(1 for size in sub if size >= 1)
    if effective < 2:
        hits["D01"] = f"effective_turns={effective}"
    if not any(role == "patient" for role in roles):
        hits["D02"] = "no_patient_turn"
    if not any(role == "doctor" for role in roles):
        hits["D09"] = "no_doctor_turn"
    if not roles or roles[0] != "patient":
        hits["D08"] = f"first_role={roles[0] if roles else 'none'}"
    # D10 keeps only the consultations with nothing clinical to supervise: every
    # doctor turn is shorter than a conclusion. A case where the physician said
    # something substantial but never concluded stays in the corpus and carries
    # F14, because it still shows what a patient reports and what was asked.
    doctor_chars = [spoken_len(texts[i]) for i, role in enumerate(roles)
                    if role == "doctor"]
    if doctor_chars and max(doctor_chars) < B.MIN_CONCLUSION_CHARS:
        hits["D10"] = f"max_doctor_chars={max(doctor_chars)}"
    return hits


def hitting_privacy(texts) -> set:
    found: set = set()
    for text in texts:
        if not PRIVACY_GATE.search(text):
            continue
        for name, pattern in PRIVACY_PATTERNS:
            if pattern.search(text):
                found.add(name)
    return found


def fingerprint(turns) -> str:
    """Turn-text sequence fingerprint: whitespace-insensitive, so a case that
    differs from another only in spacing is still the same case."""
    payload = "\x1f".join(re.sub(r"\s+", "", str(t.get("text") or "")) for t in turns)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------- S2 repair

ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\u00ad\ufeff]")
FULLWIDTH_ALNUM = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
MASK_TOKEN = re.compile(
    r"[\[［]\s*(?:电话|证件号|邮箱|链接|账号|病历编号)\s*已隐去\s*[\]］]")
# Two repetitions are how people type ("好的，，谢谢", "还在吗？？"); three or
# more is a defect. The census records both populations so the threshold is
# visible, and v2 kept the same line.
PUNCT_RUN = re.compile(r"([，。、；：！？,.!?;:])\1{2,}")
SPACE_RUN = re.compile(r"[ \t\u3000\x0b\x0c]+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([，。、；：！？,.!?;:)）】])")

URL = re.compile(r"(?:https?://|www\.)[^\s<>，。；;）)】]+", re.I)
BARE_DOMAIN = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9\-]{0,62}\.){1,4}"
    r"(?:com|cn|net|org|edu|gov|top|cc|xyz|info|vip)(?:\.cn)?"
    r"(?:/[^\s<>，。；;）)】]*)?", re.I)
LINK_INVITE = re.compile(
    r"(?:点击|复制|打开|查看|访问|长按)\s*(?:下方|下面|此)?\s*(?:链接|网址|地址|原文)\s*"
    r"(?:查看|了解|阅读|打开)?")
LINK_GATE = re.compile(r"http|www\.|\.(?:com|cn|net|org|edu|gov)\b|链接|网址|地址", re.I)

TIMESTAMP = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?"
    r"(?:\s*\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?)?")
CLOCK = re.compile(r"(?<![\d:])\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?(?![\d:])")
SPEAKER_LABEL = re.compile(
    r"^\s*[\[【（(]?\s*(?:患者|病人|医生|医师|用户|提问者|咨询者|回复人|楼主"
    r"|匿名用户|匿名)\s*[\]】）)]?\s*[:：]\s*")
ANON_TAG = re.compile(
    r"[\[【（(]\s*(?:匿名用户|匿名|佚名|用户已注销|已注销)\s*[\]】）)]")
APP_TAIL = re.compile(r"(?:来自|via)\s*[A-Za-z0-9._+-]{2,24}\s*(?:客户端|APP|app)?\s*$")
R08_GATE = re.compile(
    r"\d{1,2}\s*[:：]\s*\d{2}[-/.年]|匿名|佚名|已注销|来自|via"
    r"|^\s*[\[【（(]?(?:患者|病人|医生|医师|用户|提问者)")

SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")
INVALID_NAMES = {"", "undefined", "null", "none", "-", "nan"}

DOCTOR_VOICE = re.compile(
    r"^\s*(?:建议[你您]|建议口服|建议使用|建议加用|建议先|"
    r"[你您]可以(?:服用|口服|使用|尝试|吃)|[你您]应该|[你您]需要(?:服用|口服|做)|"
    r"可以(?:服用|口服)|推荐[你您](?:服|使|吃)|服用[^。]{0,8}即可)")
SESSION_OPENING = re.compile(
    r"^\s*(?:你好|您好|医生你好|医生您好|在吗|医生在吗|请问在吗|你好在吗|医生好)")

# v2 already stripped promotion once, so R01 only catches residue. The gate
# names the vocabulary those spans look for, so the expensive call runs on the
# few turns that could hold it instead of on all 2.19M.
R01_GATE = re.compile(
    r"好评|五星|评价|满意|评分|打分|关注|二维码|扫码|公众号|头像|微信|QQ|好友|"
    r"联系方式|电话|网站|主页|博客|文章|参阅|参见|详见|经典咨询|温馨提示|"
    r"小贴士|优惠|劵|券|心意|平台|春雨|客服|小助手|健康助手|药房|APP|app|"
    r"应用|客户端|软件|添加|系统|规定|限制|要求|上限|规则|设置|拨打|联系|"
    r"以前|之前|先前|一问一答|首次回复|简单|咨询次数|详细回复|解答完毕|追问|"
    r"《|点击|长按|链接|网址")


def apply_r01(text: str) -> tuple:
    """Residual platform template inside a doctor turn."""
    if not R01_GATE.search(text):
        return text, 0
    new, removed = V2.strip_promotion(text)
    if removed:
        new = V2.repair_seams(new)
    return new, removed


def apply_r02(text: str) -> tuple:
    new, count = PLACEHOLDER.subn(IMAGE_PLACEHOLDER, text)
    return new, count


def apply_r05(text: str) -> tuple:
    if not LINK_GATE.search(text):
        return text, 0
    new, count = URL.subn(" ", text)
    new, more = BARE_DOMAIN.subn(" ", new)
    count += more
    new, more = LINK_INVITE.subn(" ", new)
    count += more
    if count:
        new = V2.repair_seams(new)
    return new, count


def apply_r07(text: str) -> tuple:
    new = ZERO_WIDTH.sub("", text)
    new = new.translate(FULLWIDTH_ALNUM)
    new, masked = MASK_TOKEN.subn(" ", new)
    new = SPACE_RUN.sub(" ", new)
    new = SPACE_BEFORE_PUNCT.sub(r"\1", new)
    new = PUNCT_RUN.sub(r"\1", new).strip()
    count = masked + (1 if new != text else 0)
    if masked:
        new = V2.repair_seams(new).strip()
    return new, count


def apply_r08(text: str) -> tuple:
    if not R08_GATE.search(text):
        return text, 0
    new = SPEAKER_LABEL.sub("", text)
    new = ANON_TAG.sub(" ", new)
    new = TIMESTAMP.sub(" ", new)
    new = CLOCK.sub(" ", new)
    new = APP_TAIL.sub("", new)
    new = SPACE_RUN.sub(" ", new).strip()
    if new != text:
        new = V2.repair_seams(new).strip()
    return new, (1 if new != text else 0)


def apply_r13(text: str) -> tuple:
    """Platform blanks: the media the platform hid, and the placeholders it
    left behind. They read like content and carry none."""
    if not PLATFORM_BLANK.search(text):
        return text, 0
    new, count = PLATFORM_BLANK.subn(" ", text)
    new = V2.repair_seams(SPACE_RUN.sub(" ", new)).strip()
    return new, count


def apply_r03(text: str) -> tuple:
    """Sentence-level dedupe inside one turn only.

    Cross-turn repetition is deliberately left alone: a doctor asking the same
    question twice is a fact about that consultation (F08 marks it), and
    deleting the second ask would rewrite what actually happened.
    """
    parts = [p for p in SENTENCE_SPLIT.split(text) if p]
    if len(parts) < 2:
        return text, 0
    seen: set = set()
    kept: list = []
    removed = 0
    for part in parts:
        key = re.sub(r"[\s，。、；：！？,.!?;:]+", "", part)
        if len(key) >= 4 and key in seen:
            removed += 1
            continue
        if len(key) >= 4:
            seen.add(key)
        kept.append(part)
    if not removed:
        return text, 0
    return "".join(kept).strip(), removed


# R01 is a doctor-turn rule by definition: a patient naming a website is a
# symptom of the platform, not a promotion.
RULE_ROLES = {"R01": ("doctor",)}

TEXT_RULES = (
    ("R01", apply_r01),
    ("R02", apply_r02),
    ("R13", apply_r13),
    ("R05", apply_r05),
    ("R08", apply_r08),
    ("R07", apply_r07),
    ("R03", apply_r03),
)


def invalid_entity(item) -> bool:
    if not isinstance(item, dict):
        return False
    if "name" not in item:
        return False
    value = item.get("name")
    return str(value if value is not None else "").strip().lower() in INVALID_NAMES


def find_run_in(haystack: str, needle: str, size: int):
    """Start index in haystack of the first "size"-character run shared with
    needle, or None. Indices refer to the original string."""
    def clean(text: str) -> tuple:
        keep = [i for i, ch in enumerate(text)
                if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", ch)]
        return "".join(text[i] for i in keep), keep

    target, keep = clean(haystack)
    other, _ = clean(needle)
    if len(target) < size or len(other) < size:
        return None
    grams = {other[i:i + size] for i in range(len(other) - size + 1)}
    for start in range(len(target) - size + 1):
        if target[start:start + size] in grams:
            return keep[start] if start < len(keep) else len(haystack)
    return None


def trim_spliced_opening(turns, texts) -> tuple:
    """R14: cut a later patient turn's text out of the patient's opening.

    Some corpora concatenate the following exchanges into the first message, so
    the opening holds words the patient had not said yet. The prompt is built
    from that opening, which turns the leak into policy input. The overlap is
    cut at its first occurrence and the remainder is re-checked against the
    chief-complaint floor by analyse().
    """
    first_index = next((i for i, t in enumerate(turns)
                        if str(t.get("role") or "") == "patient"), None)
    if first_index is None:
        return texts, None
    opening = texts[first_index]
    if len(opening) < SPLICE_OVERLAP:
        return texts, None
    for index, turn in enumerate(turns):
        if index <= first_index or str(turn.get("role") or "") != "patient":
            continue
        later = texts[index]
        if len(later) < SPLICE_OVERLAP:
            continue
        cut = find_run_in(opening, later, SPLICE_OVERLAP)
        if cut is None:
            continue
        trimmed = V2.repair_seams(opening[:cut]).strip()
        if trimmed == opening:
            continue
        texts = list(texts)
        texts[first_index] = trimmed
        return texts, {"turn_index": first_index, "cut_at": cut,
                       "overlap_with": index, "before": opening, "after": trimmed}
    return texts, None


def repair(row: dict) -> dict:
    """Apply every S2 rule to a copy of the case.

    Returns the new turns plus one operation record per (rule, turn) that
    actually changed something, so the ledger can name the rule, the turn and
    both hashes. Text rules run per turn in TEXT_RULES order; R14 needs the
    whole dialogue and runs after them; R09 walks the annotations; R11 only
    marks, it never edits.
    """
    turns = [dict(t) for t in turns_of(row)]
    ops: list = []
    image_dependent = False
    doomed: set = set()

    for index, turn in enumerate(turns):
        before_all = str(turn.get("text") or "")
        role = str(turn.get("role") or "")
        text = before_all
        emptied_by = None
        for code, rule in TEXT_RULES:
            if code in RULE_ROLES and role not in RULE_ROLES[code]:
                continue
            if code == "R02":
                marked = bool(PLACEHOLDER.search(text)) or IMAGE_PLACEHOLDER in text
                image_dependent = image_dependent or marked
            new, count = rule(text)
            if count and new != text:
                ops.append({
                    "action": "repair",
                    "rule_code": code,
                    "turn_index": index,
                    "before_sha1": sha1(text),
                    "after_sha1": sha1(new),
                    "detail": f"hits={count}",
                })
            if text and not new:
                emptied_by = code
            text = new
        turn["text"] = text
        if before_all and not text:
            # An emptied turn carries nothing and would render as an empty
            # assistant or patient message, so it leaves the dialogue. The
            # ledger keeps the original turn index.
            doomed.add(index)
            ops.append({
                "action": "repair",
                "rule_code": emptied_by or "R07",
                "turn_index": index,
                "before_sha1": sha1(before_all),
                "after_sha1": sha1(""),
                "detail": "turn_emptied_by_repair_removed",
            })

    texts = texts_of(turns)
    texts, splice = trim_spliced_opening(turns, texts)
    if splice:
        turns[splice["turn_index"]]["text"] = texts[splice["turn_index"]]
        ops.append({
            "action": "repair",
            "rule_code": "R14",
            "turn_index": splice["turn_index"],
            "before_sha1": sha1(splice["before"]),
            "after_sha1": sha1(splice["after"]),
            "detail": f"cut_at={splice['cut_at']} overlaps_turn={splice['overlap_with']}",
        })
        if not texts[splice["turn_index"]]:
            doomed.add(splice["turn_index"])

    for index, turn in enumerate(turns):
        if index in doomed:
            continue
        for key, value in list(turn.items()):
            if key in ("role", "text") or not isinstance(value, list):
                continue
            kept = [item for item in value if not invalid_entity(item)]
            if len(kept) != len(value):
                before = json.dumps(value, ensure_ascii=False, sort_keys=True)
                after = json.dumps(kept, ensure_ascii=False, sort_keys=True)
                turn[key] = kept
                ops.append({
                    "action": "repair",
                    "rule_code": "R09",
                    "turn_index": index,
                    "before_sha1": sha1(before),
                    "after_sha1": sha1(after),
                    "detail": f"field={key} dropped_entities={len(value) - len(kept)}",
                })

    manual = [
        {"turn_index": index, "text": str(turn.get("text") or "")}
        for index, turn in enumerate(turns)
        if turn.get("role") == "patient" and DOCTOR_VOICE.search(str(turn.get("text") or ""))
    ]
    for item in manual:
        ops.append({
            "action": "flag",
            "rule_code": "R11",
            "turn_index": item["turn_index"],
            "before_sha1": sha1(item["text"]),
            "after_sha1": sha1(item["text"]),
            "detail": "doctor_voice_in_patient_turn_queued_for_manual_review",
        })

    kept = [turn for index, turn in enumerate(turns) if index not in doomed]
    return {
        "turns": kept,
        "ops": ops,
        "image_dependent": image_dependent,
        "manual": manual,
        "turns_removed": len(doomed),
        "spliced": bool(splice),
    }


# --------------------------------------------------------------------- S3 flag

def inspect_flags(turns, image_dependent: bool, texts=None, sub=None,
                  labels=None) -> list:
    if texts is None:
        texts = texts_of(turns)
    if sub is None:
        sub = [substantive_len(text) for text in texts]
    roles = [str(t.get("role") or "") for t in turns]
    doctor_indexes = [i for i, role in enumerate(roles) if role == "doctor"]
    patient_indexes = [i for i, role in enumerate(roles) if role == "patient"]
    flags: list = []

    conclusion = B.conclusion_of(turns) if turns else None
    if len(doctor_indexes) < MIN_DOCTOR_TURNS and conclusion is not None:
        flags.append("F01")
    if len(doctor_indexes) > MAX_DOCTOR_TURNS:
        flags.append("F02")
    if image_dependent:
        flags.append("F03")

    short = [i for i in doctor_indexes if sub[i] < SHORT_TURN_CHARS]
    if doctor_indexes and (len(short) == len(doctor_indexes)
                           or len(short) >= SHORT_DOCTOR_TURNS):
        flags.append("F04")

    # "no closing conclusion" reads as: the consultation's last doctor turn is
    # not the conclusion, so the conclusion lands mid-dialogue.
    if conclusion is not None and doctor_indexes and conclusion != doctor_indexes[-1]:
        flags.append("F05")

    # F14: the case has no conclusion turn at all — there is no reward reference
    # and no supervised closing, but the consultation still teaches what the
    # patient reports and which questions were asked.
    if turns and conclusion is None:
        flags.append("F14")

    spoken = sum(1 for i in patient_indexes if sub[i] >= 2)
    if spoken == 1:
        flags.append("F06")

    openings = sum(1 for i in patient_indexes if SESSION_OPENING.match(texts[i]))
    if openings >= 2:
        flags.append("F07")

    run, previous = 1, None
    longest = 1
    for i in doctor_indexes:
        key = re.sub(r"[\s，。、；：！？,.!?;:]+", "", texts[i])
        run = run + 1 if key and key == previous else 1
        previous = key
        longest = max(longest, run)
    if longest >= 2:
        flags.append("F08")

    cjk, latin, _ = _ratio(" ".join(PLACEHOLDER.sub(" ", text) for text in texts))
    if latin >= 8 and cjk >= 10 and latin / (cjk + latin) >= 0.15:
        flags.append("F10")

    if len(turns) < 3:
        flags.append("F11")

    if labels and B.is_department_name(labels.get("topic")):
        flags.append("F12")

    return sorted(set(flags))


def quality_of(flags, image_dependent: bool, doctor_turns: int,
               conclusion_available: bool = True) -> dict:
    rules = sorted(set(flags))
    return {
        "passed": not (set(rules) & BLOCKING_FLAGS),
        "rules": rules,
        "image_dependent": bool(image_dependent),
        "doctor_turns_ok": MIN_DOCTOR_TURNS <= doctor_turns <= MAX_DOCTOR_TURNS,
        "conclusion_available": bool(conclusion_available),
    }


def case_id_of(row: dict) -> str:
    return B.opaque("c", case_key(row))


def case_key(row: dict) -> str:
    return f"{row.get('source')}:{row.get('id')}"


def split_of(row: dict) -> str:
    """The case-level split both training exports use.

    The cleaner pins v3_split on every row from its group pass, so every case
    sharing an opening question lands in the same split; a corpus without the
    pin falls back to the per-case hash split, which is what v2 used.
    """
    pinned = row.get("v3_split")
    if pinned in ("train", "dev", "test"):
        return pinned
    return B.split_of(case_key(row))


def analyse(row: dict, dup_of=None) -> dict:
    """One case measured and repaired in a single pass.

    Both consumers call this: the census reads the counters, the cleaner
    writes the ledger. Drops are evaluated on the original case and
    short-circuit the repair, which is the order the cleaner applies them —
    there is no point repairing a case that leaves the corpus.
    """
    turns = turns_of(row)
    texts = texts_of(turns)
    sub = [substantive_len(text) for text in texts]
    drops = inspect_drops(row, dup_of=dup_of, texts=texts, sub=sub)
    if drops:
        return {"drops": drops, "post_drops": {}, "turns": turns, "ops": [],
                "flags": [], "image_dependent": False, "manual": [],
                "quality": {}, "texts": texts, "sub": sub, "spliced": False}

    fixed = repair(row)
    fixed_texts = texts_of(fixed["turns"])
    fixed_sub = [substantive_len(text) for text in fixed_texts]
    post = structural_drops(fixed["turns"], fixed_texts, fixed_sub)
    if fixed["spliced"]:
        first = next((i for i, t in enumerate(fixed["turns"])
                      if str(t.get("role") or "") == "patient"), None)
        if first is None or fixed_sub[first] < SPLICE_MIN_KEEP:
            post["D06"] = f"post_r14_len={fixed_sub[first] if first is not None else 0}"
    labels = row.get("labels") or {}
    flags = inspect_flags(fixed["turns"], fixed["image_dependent"],
                          fixed_texts, fixed_sub, labels)
    doctor_turns = sum(1 for t in fixed["turns"] if t.get("role") == "doctor")
    return {
        "drops": {},
        "post_drops": post,
        "turns": fixed["turns"],
        "ops": fixed["ops"],
        "flags": flags,
        "image_dependent": fixed["image_dependent"],
        "manual": fixed["manual"],
        "quality": quality_of(flags, fixed["image_dependent"], doctor_turns,
                             B.conclusion_of(fixed["turns"]) is not None),
        "texts": fixed_texts,
        "sub": fixed_sub,
        "spliced": fixed["spliced"],
    }
