"""Export the cold-start SFT set from the cleaned corpus.

SFT here teaches the format: the three-action protocol exactly as real
consultations read — WEN (问, ask), TALK (闲, converse: reassure, acknowledge,
advise without deciding), DIG (诊, conclude). Roughly half of a real doctor's
turns are TALK, so the format the model learns includes it; a later LLM pass
will attach <think> reasoning to these records, and RL will treat TALK as a
non-terminal conversational turn.

Two modes.

turn (default): each WEN or DIG decision becomes one record holding the
visible history up to that point and the observed action as the target.
History shows every earlier turn, TALK included, in the JSON action format.

dialogue: one record per consultation — the whole sequence as it unfolded,
every doctor turn supervised, TALK included.

Every assistant message, in history and target alike, is serialised in the
JSON action format, because that is what the model will see as its own past
output at inference.

SFT and RL share the case-level hash split (build_training_data.split_of), so a
consultation never crosses between the two views' train/dev/test.

Roles are written per the trainer's template: ms-swift and LLaMA-Factory expect
user/assistant (default); --roles patient-doctor keeps the consultation's own
vocabulary for a custom template.

Output: data/sft_rl/{train,dev,test}.jsonl, plus _sft_report.json

Usage:
    python scripts/export_sft.py --mode dialogue   # SFT cold start
    python scripts/export_sft.py --mode turn
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_data as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed_2"
DEFAULT_OUTPUT = ROOT / "data/sft_rl"

ROLE_MAP = {
    "user-assistant": {"patient": "user", "doctor": "assistant"},
    "patient-doctor": {"patient": "patient", "doctor": "doctor"},
}


def action_json(action: str, message: str) -> str:
    return json.dumps({"action": action, "message": message},
                      ensure_ascii=False, separators=(",", ":"))


def render_history(history: list[dict], actions: dict[int, str],
                   role_map: dict[str, str]) -> list[dict]:
    """History as the model would see its own past: every doctor turn,
    TALK included, in the JSON action format."""
    messages: list[dict] = []
    for i, turn in enumerate(history):
        if turn["role"] == "doctor":
            messages.append({"role": role_map["doctor"],
                             "content": action_json(actions[i], turn["text"])})
        else:
            messages.append({"role": role_map["patient"], "content": turn["text"]})
    return messages


def select(decisions: list[dict], limit: int) -> list[dict]:
    """Spread the picks over the consultation — how it opens, how it closes,
    and points in between. Neighbouring decisions share nearly all their
    history, so an uncapped export repeats one context many times."""
    if limit <= 0 or len(decisions) <= limit:
        return decisions
    if limit == 1:
        return [decisions[0]]
    step = (len(decisions) - 1) / (limit - 1)
    picked = [decisions[round(k * step)] for k in range(limit)]
    closing = decisions[-1]
    if closing not in picked:
        picked[-1] = closing
    return picked


def stamp(record: dict, row_id: str, split: str, marker: dict,
          source: str, doctor_turns: int) -> dict:
    """Case identity, source, split and quality markers ride along with the
    messages so a trainer can deduplicate, filter and stratify without a side
    table. No system message is added: scripts/sys_prompt.py injects that at
    train time."""
    record["case_id"] = B.opaque("c", row_id)
    record["source"] = source
    record["split"] = split
    record["n_doctor_turns"] = doctor_turns
    record["quality"] = {
        "passed": marker.get("passed", True) is not False,
        "rules": sorted(str(code) for code in (marker.get("rules") or [])),
    }
    return record


def to_record(row_id: str, turn_id: int, history: list[dict],
              action: str, message: str, actions: dict[int, str],
              role_map: dict[str, str]) -> dict:
    messages = render_history(history, actions, role_map)
    if not messages:
        raise ValueError(f"empty history: {row_id}#{turn_id}")
    if messages[-1]["role"] != role_map["patient"]:
        raise ValueError(
            f"history must end with a patient turn: {row_id}#{turn_id}")
    return {
        "messages": [
            *messages,
            {"role": role_map["doctor"], "content": action_json(action, message)},
        ],
        "sample_id": B.opaque("s", f"{row_id}#{turn_id}"),
    }


def to_dialogue_record(row_id: str, turns: list[dict],
                       actions: dict[int, str],
                       role_map: dict[str, str]) -> dict | None:
    """One record = one whole consultation, supervising every WEN and DIG in
    order. Returns None when nothing supervised remains after the TALK drop or
    the dialogue does not read as a patient→doctor exchange. The record may end
    on a patient turn: 39.7% of consultations close on a pleasantry that the
    TALK drop removes, and the sequence still ends where the last decision
    was."""
    if not turns or turns[0]["role"] != "patient":
        return None
    messages = render_history(turns, actions, role_map)
    roles = {m["role"] for m in messages}
    if roles != {role_map["patient"], role_map["doctor"]}:
        return None
    return {"messages": messages, "sample_id": B.opaque("d", row_id)}


def rl_case_sets(rl_dir: Path) -> dict:
    """case_id per split as the RL parquet files carry it."""
    import pyarrow.parquet as pq

    out: dict = {}
    for name in ("train", "dev", "test"):
        path = rl_dir / f"{name}.parquet"
        if not path.exists():
            continue
        try:
            table = pq.read_table(path, columns=["extra_info.case_id"])
            # pyarrow returns the selected leaf as its own top-level column in
            # some versions and as a narrowed struct in others; accept both.
            if "case_id" in table.schema.names:
                ids = table.column("case_id")
            else:
                ids = table.column("extra_info").combine_chunks().field("case_id")
            out[name] = set(ids.to_pylist())
        except Exception as exc:                      # noqa: BLE001
            print(f"（读取 {path.name} 的 case_id 失败：{exc}）")
    return out


def case_set_relation(splits: dict, rl_dir) -> dict | None:
    """How the SFT case set relates to the RL episode set.

    The two views read the same corpus, but they filter differently: RL needs a
    conclusion inside the turn ceiling, SFT needs a supervised non-TALK
    decision on top of that. This reports the difference instead of assuming
    one contains the other.
    """
    if rl_dir is None:
        return None
    rl_sets = rl_case_sets(Path(rl_dir))
    if not rl_sets:
        return None
    sft_sets = {name: {record["case_id"] for record in rows}
                for name, rows in splits.items()}
    rl_all = set().union(*rl_sets.values())
    sft_all = set().union(*sft_sets.values())
    return {
        "rl_dir": str(rl_dir),
        "rl_cases": len(rl_all),
        "sft_cases": len(sft_all),
        "intersection": len(sft_all & rl_all),
        "sft_minus_rl": len(sft_all - rl_all),
        "rl_minus_sft": len(rl_all - sft_all),
        "sft_subset_of_rl": not (sft_all - rl_all),
        "per_split": {
            name: {
                "rl": len(rl_sets.get(name, set())),
                "sft": len(sft_sets[name]),
                "sft_minus_rl": len(sft_sets[name] - rl_sets.get(name, set())),
                "rl_minus_sft": len(rl_sets.get(name, set()) - sft_sets[name]),
            } for name in ("train", "dev", "test")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-dialogue", type=int, default=4)
    parser.add_argument(
        "--roles", choices=["user-assistant", "patient-doctor"],
        default="user-assistant",
        help="how to name the two parties; ms-swift and LLaMA-Factory "
             "templates expect user-assistant, which is the default")
    parser.add_argument("--max-doctor-turns", type=int,
                        default=B.DEFAULT_MAX_DOCTOR_TURNS,
                        help="same ceiling the RL build applies, so the SFT "
                             "records are always a subset of the episodes")
    parser.add_argument("--rl-dir", type=Path, default=None,
                        help="directory holding the RL parquet files; used to "
                             "report how the SFT case set relates to it")
    parser.add_argument(
        "--mode", choices=["turn", "dialogue"], default="turn",
        help="turn: one record per decision point; "
             "dialogue: one record per whole consultation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    role_map = ROLE_MAP[args.roles]

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    actions_count: collections.Counter = collections.Counter()
    talk_total = 0
    dropped_no_history = 0
    blocked_by_quality = 0
    skipped_no_conclusion = 0
    skipped_long = 0
    dialogues = 0

    # A leading underscore marks bookkeeping written next to the corpus, never
    # a source.
    for path in sorted(p for p in args.input.glob("*.jsonl")
                       if not p.name.startswith("_")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                dialogues += 1
                row_id = f"{row['source']}:{row['id']}"
                marker = row.get("quality") or {}
                # A case the cleaner marked as unreliable teaches the format
                # with a demonstration nobody would follow.
                if marker.get("passed") is False:
                    blocked_by_quality += 1
                    continue
                turns = row["turns"]
                doctor_indexes = [i for i, t in enumerate(turns) if t["role"] == "doctor"]
                if not doctor_indexes:
                    continue
                # The same gates the RL build applies: without them the SFT
                # records would not be a subset of the episodes, and the two
                # views would teach different case sets.
                if len(doctor_indexes) > args.max_doctor_turns:
                    skipped_long += 1
                    continue
                conclusion = B.conclusion_of(turns)
                if conclusion is None:
                    skipped_no_conclusion += 1
                    continue
                actions = {i: B.action_of(turns[i], i == conclusion)
                           for i in doctor_indexes}
                talk_total += sum(1 for a in actions.values() if a == "TALK")
                if not any(a != "TALK" for a in actions.values()):
                    continue

                split = B.split_of(row)
                if args.mode == "dialogue":
                    record = to_dialogue_record(row_id, turns, actions, role_map)
                    if record is not None:
                        splits[split].append(stamp(record, row_id, split, marker,
                                                   str(row["source"]), len(doctor_indexes)))
                        actions_count.update(actions.values())
                else:
                    decisions = []
                    for i in doctor_indexes:
                        if actions[i] == "TALK":
                            continue          # a conversational turn is not a decision
                        history = turns[:i]
                        # Cold start needs a history that is non-empty and ends
                        # with the patient; a decision without one has nothing
                        # to condition on.
                        if not history or history[-1]["role"] != "patient":
                            dropped_no_history += 1
                            continue
                        decisions.append({"turn_id": i, "action": actions[i],
                                          "message": turns[i]["text"]})
                    for decision in select(decisions, args.max_per_dialogue):
                        splits[split].append(stamp(to_record(
                            row_id, decision["turn_id"],
                            turns[:decision["turn_id"]], decision["action"],
                            decision["message"], actions, role_map),
                            row_id, split, marker, str(row["source"]),
                            len(doctor_indexes)))
                        actions_count[decision["action"]] += 1

    total = sum(len(v) for v in splits.values())
    print(f"{dialogues:,} 条问诊 → {total:,} 条 SFT 记录（mode={args.mode}，"
          f"含 TALK 轮 {talk_total:,}）")
    for name in ("train", "dev", "test"):
        out = args.output / f"{name}.jsonl"
        rows = sorted(splits[name], key=lambda r: r["sample_id"])
        with out.open("w", encoding="utf-8", newline="\n") as stream:
            for record in rows:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"写出 {out.name:<14}{len(rows):>9,} 行  {out.stat().st_size / 1e6:>7.1f} MB")

    if skipped_no_conclusion:
        print(f"（无实质结论，跳过 {skipped_no_conclusion:,} 条对话）")
    if skipped_long:
        print(f"（医生轮超过 {args.max_doctor_turns}，跳过 {skipped_long:,} 条对话）")
    print(f"\n动作分布 {dict(actions_count)}")
    if dropped_no_history:
        print(f"（因缺少患者历史而跳过 {dropped_no_history:,} 个决策点）")
    if blocked_by_quality:
        print(f"（质量门阻断，跳过 {blocked_by_quality:,} 条对话）")
    relation = case_set_relation(splits, args.rl_dir)
    if relation:
        # ASCII only: a GBK console cannot encode ∩ or −, and losing the report
        # write to a print is not worth the typography.
        print(f"\nSFT&RL {relation['intersection']:,}；"
              f"SFT-RL {relation['sft_minus_rl']:,}；RL-SFT {relation['rl_minus_sft']:,}")
    report = {
        "case_set_relation": relation,
        "skipped_no_conclusion": skipped_no_conclusion,
        "skipped_doctor_turns_over_limit": skipped_long,
        "source": str(args.input),
        "mode": args.mode,
        "roles": args.roles,
        "max_per_dialogue": args.max_per_dialogue,
        "dialogues": dialogues,
        "records": {k: len(v) for k, v in splits.items()},
        "records_total": total,
        "talk_turns_supervised_or_rendered": talk_total,
        "decisions_dropped_no_history": dropped_no_history,
        "dialogues_blocked_by_quality": blocked_by_quality,
        "by_action": dict(actions_count),
    }
    (args.output / "_sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告: {args.output / '_sft_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
