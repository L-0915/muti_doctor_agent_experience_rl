# Multi-turn Medical Consultation RL Data (HEAPO episode-v3)

Training data for research on reinforcement learning of a doctor agent that decides, at every turn, whether to **ask** (WEN), **talk** (TALK), or **conclude** (DIG) in a multi-turn medical consultation.

Chinese 中文说明见文末。

## What is inside

| File | Rows | Description |
| --- | ---: | --- |
| `train.parquet` / `dev.parquet` / `test.parquet` | 110,840 / 13,988 / 13,747 | RL episodes (veRL-style), one consultation per row |
| `sft10k/train.jsonl` / `dev.jsonl` / `test.jsonl` | 10,000 / 1,250 / 1,250 | Cold-start SFT (dialogue level), stratified sample |
| `train_sample_tiny.parquet` | 20 | 20 RL episodes for a quick smoke test |
| `sft10k/train_sample_tiny.jsonl` | 20 | 20 SFT records for a quick smoke test |
| `spotcheck20.csv` | 20 | Human-readable spot-check list (metadata only) |
| `manifest.json`, `SHA256SUMS` | — | Counts and per-file SHA-256 |

Full SFT sets (dialogue level 129,148 rows, turn level 376,338 rows) are not published here; they are regenerable from the corpus with the scripts below.

## Schema: RL episode (parquet)

```
prompt          [{role: user, content: <patient's first message>}]
ground_truth    {reference_conclusion, diagnosis, presenting_problem, department, department_path}
extra_info      {sample_id, case_id,
                 patient_state {full_dialogue, symptoms},
                 profile {dialogue_turns, doctor_turns, wen, dig, talk},
                 quality {rules, image_dependent, doctor_turns_ok, pp_is_department, conclusion_available},
                 turn_budget}
data_source     "heapo-episode-v1"
index           int64
```

`prompt` is the only text the policy may see. `patient_state.full_dialogue` is the patient simulator's knowledge base and must never be placed in the prompt. `extra_info.quality.conclusion_available=false` marks 1,965 episodes whose doctor turns contain no conclusion-like utterance; they are usable for state/action learning but must not receive an outcome reward.

## Schema: SFT cold start (jsonl)

One record per dialogue: `messages` (user/assistant alternation), `sample_id`, `case_id`, `source`, `split`, `n_doctor_turns`, `quality{passed, rules}`. Every assistant message is a JSON string `{"action":"WEN|TALK|DIG","message":"..."}`. There is no system message: the system prompt lives in `sys_prompt.py` and is injected at training time.

## Protocol

`WEN` ask one specific question; `TALK` respond without deciding (reassure, acknowledge, advise); `DIG` conclude with next steps. About half of a real doctor's turns are `TALK`, which is why the action space has three classes instead of two.

## Splits and integrity

Splits are case-level: first-question fingerprint grouping, then turn-sequence dedup, then group-level hash split. Guarantees checked by `check_integrity_v3.py` (51 checks, 49 pass): no first question crosses splits, no approximate duplicate crosses splits, SFT and RL agree on splits, no conclusion text leaks into prompts, no conclusion is promotional text or a registration form or the patient's own words. Two known residuals are documented in the cleaning report.

## Quick start

```python
import json
import pyarrow.parquet as pq

rows = pq.read_table("train_sample_tiny.parquet").to_pylist()
row = rows[0]
print(row["prompt"])            # what the policy sees
print(row["ground_truth"]["reference_conclusion"])

for line in open("sft10k/train_sample_tiny.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    print(rec["messages"][-1]["content"])
```

## Scripts

`clean_cases.py` (per-source normalisation) → `clean_corpus_v3.py` + `corpus_v3_rules.py` (33 cleaning rules, per-change ledger) → `build_training_data.py` (RL parquet) / `export_sft.py` (SFT jsonl) → `sample_sft_stratified.py` (cold-start sample) → `check_integrity_v3.py`. `audit_corpus_v3.py` and `audit_training_files.py` are read-only auditors.

## Limitations

* Not a clinical gold standard. Only IMCS-21 provides case-level diagnosis labels; the other five corpora do not. Do not report a single "diagnosis accuracy" across sources.
* No pre-visit EHR: the policy sees only what the patient disclosed.
* Source distribution is skewed (KaMed ≈ 51% of the corpus); stratify or reweight when comparing sources.
* Chinese only. English translation and external English check sets are future work.

## 中文说明

本仓库提供多轮医疗问诊强化学习数据（HEAPO episode-v3）：RL episode 共 138,575 条（train 110,840 / dev 13,988 / test 13,747），冷启动 SFT 共 12,500 条（10,000 / 1,250 / 1,250），另有 20 条快速冒烟样本与 20 条人工抽检清单。动作空间为 WEN（问）/ TALK（聊）/ DIG（诊）三动作，系统提示词见 `sys_prompt.py`。数据按病例级切分，已完成 51 项完整性检查。本数据仅供研究使用，不构成临床金标准；五个来源没有病例级诊断标签，报告结果时须按来源分别说明。

## License / contact

Research use. The underlying consultations come from public academic corpora (IMCS-21, ReMeDi, MedDG, CHIP-MDCFNPC, MedDialog, KaMed); rights remain with their original owners. Direct identifiers were scanned and no matches were found; the scan is a filter, not an anonymity certification. Issues and questions: open an issue in this repository.