from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


CANDIDATES_PATH = Path(
    "data/processed/splits/teacher_sft_candidates.jsonl"
)
TRAJECTORIES_PATH = Path(
    "data/processed/teacher/raw_teacher_trajectories.jsonl"
)
EMPTY_SKIPPED_PATH = Path(
    "data/processed/teacher/raw_teacher_empty_skipped_ids.txt"
)
REPORT_PATH = Path(
    "data/processed/teacher/teacher_audit_report.json"
)

ALLOWED_EVENT_TYPES = {"think", "search", "information", "answer"}
EXPECTED_TERMINATIONS = {"answer", "max_turns", "invalid_action"}


def load_jsonl(path: Path, required_fields: set[str]) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            missing = required_fields - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields in {path} at line {line_number}: "
                    f"{sorted(missing)}"
                )
            records.append(record)
    return records


def load_id_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as input_file:
        return [line.strip() for line in input_file if line.strip()]


def duplicate_ids(ids: list[str]) -> list[str]:
    counts = Counter(ids)
    return sorted(uid for uid, count in counts.items() if count > 1)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
    tokens = [
        token
        for token in re.split(r"\s+", text.strip())
        if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def is_exact_match(prediction: str, gold_answers: list[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(
        normalized_prediction == normalize_answer(gold)
        for gold in gold_answers
    )


def validate_trajectory(record: dict[str, Any]) -> tuple[bool, list[str]]:
    events = record["events"]
    reasons = set()

    if not isinstance(events, list):
        return False, ["events_not_list"]

    event_types = []
    answer_indices = []
    search_count = 0
    information_count = 0

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            reasons.add("event_not_object")
            event_types.append(None)
            continue
        if "type" not in event or "content" not in event:
            reasons.add("missing_event_field")
        event_type = event.get("type")
        event_types.append(event_type)
        if event_type not in ALLOWED_EVENT_TYPES:
            reasons.add("unknown_event_type")
        if event_type == "search":
            search_count += 1
            if (
                index + 1 >= len(events)
                or not isinstance(events[index + 1], dict)
                or events[index + 1].get("type") != "information"
            ):
                reasons.add("search_not_followed_by_information")
        elif event_type == "information":
            information_count += 1
            if index == 0 or event_types[index - 1] != "search":
                reasons.add("orphan_information")
        elif event_type == "answer":
            answer_indices.append(index)

    if search_count != information_count:
        reasons.add("search_information_count_mismatch")
    if len(answer_indices) > 1:
        reasons.add("multiple_answers")
    if answer_indices and answer_indices[-1] != len(events) - 1:
        reasons.add("answer_not_last")
    if (
        record["runtime"]["termination_reason"] == "answer"
        and len(answer_indices) != 1
    ):
        reasons.add("answer_termination_without_single_answer")

    sorted_reasons = sorted(reasons)
    return not sorted_reasons, sorted_reasons


def count_searches(record: dict[str, Any]) -> int:
    events = record["events"]
    if not isinstance(events, list):
        return 0
    return sum(
        isinstance(event, dict) and event.get("type") == "search"
        for event in events
    )


def extract_final_answer(record: dict[str, Any]) -> str | None:
    answer_events = [
        event
        for event in record["events"]
        if isinstance(event, dict) and event.get("type") == "answer"
    ]
    if len(answer_events) == 1 and record["events"][-1] is answer_events[0]:
        return answer_events[0]["content"]
    return None


def source_table(counter_by_source: dict[str, Counter]) -> dict[str, dict]:
    keys = set()
    for counter in counter_by_source.values():
        keys.update(counter)
    if "overall" in counter_by_source:
        keys.update(EXPECTED_TERMINATIONS)
    return {
        key: {
            "nq": counter_by_source["nq"].get(key, 0),
            "hotpotqa": counter_by_source["hotpotqa"].get(key, 0),
            "total": counter_by_source["overall"].get(key, 0),
        }
        for key in sorted(keys)
    }


def distribution_table(records: list[dict[str, Any]]) -> dict[str, dict]:
    counters = {
        "overall": Counter(),
        "nq": Counter(),
        "hotpotqa": Counter(),
    }
    for record in records:
        count = count_searches(record)
        counters["overall"][count] += 1
        counters[record["source"]][count] += 1
    keys = set(range(6))
    keys.update(counters["overall"])
    return {
        str(key): {
            "nq": counters["nq"].get(key, 0),
            "hotpotqa": counters["hotpotqa"].get(key, 0),
            "total": counters["overall"].get(key, 0),
        }
        for key in sorted(keys)
    }


def em_summary(rows: list[dict[str, Any]]) -> dict[str, dict]:
    result = {}
    for source in ["overall", "nq", "hotpotqa"]:
        selected = (
            rows
            if source == "overall"
            else [row for row in rows if row["source"] == source]
        )
        em_count = sum(row["exact_match"] for row in selected)
        result[source] = {
            "count": len(selected),
            "em_count": em_count,
            "em_rate": em_count / len(selected) if selected else None,
        }
    return result


def build_report() -> dict[str, Any]:
    candidates = load_jsonl(
        CANDIDATES_PATH,
        {"uid", "source", "question", "gold_answers"},
    )
    trajectories = load_jsonl(
        TRAJECTORIES_PATH,
        {"id", "source", "question", "answers", "events", "runtime"},
    )
    skipped_ids_list = load_id_file(EMPTY_SKIPPED_PATH)

    for record in trajectories:
        if "termination_reason" not in record["runtime"]:
            raise ValueError(
                f"Missing runtime.termination_reason for {record['id']}"
            )

    candidate_ids_list = [record["uid"] for record in candidates]
    trajectory_ids_list = [record["id"] for record in trajectories]
    candidate_ids = set(candidate_ids_list)
    trajectory_ids = set(trajectory_ids_list)
    skipped_ids = set(skipped_ids_list)
    candidate_source = {
        record["uid"]: record["source"] for record in candidates
    }

    candidate_duplicates = duplicate_ids(candidate_ids_list)
    trajectory_duplicates = duplicate_ids(trajectory_ids_list)
    skipped_duplicates = duplicate_ids(skipped_ids_list)
    overlap = sorted(trajectory_ids & skipped_ids)
    unknown_trajectories = sorted(trajectory_ids - candidate_ids)
    unknown_skipped = sorted(skipped_ids - candidate_ids)
    missing = sorted(candidate_ids - trajectory_ids - skipped_ids)

    coverage = {
        "candidates": len(candidates),
        "candidate_sources": dict(
            sorted(Counter(record["source"] for record in candidates).items())
        ),
        "raw_trajectories": len(trajectories),
        "empty_skipped": len(skipped_ids_list),
        "empty_skipped_sources": dict(
            sorted(
                Counter(
                    candidate_source[uid]
                    for uid in skipped_ids
                    if uid in candidate_source
                ).items()
            )
        ),
        "total_accounted": len(trajectory_ids | skipped_ids),
        "duplicate_candidate_ids": len(candidate_duplicates),
        "duplicate_candidate_id_values": candidate_duplicates,
        "duplicate_trajectory_ids": len(trajectory_duplicates),
        "duplicate_trajectory_id_values": trajectory_duplicates,
        "duplicate_skipped_ids": len(skipped_duplicates),
        "duplicate_skipped_id_values": skipped_duplicates,
        "trajectory_skipped_overlap": len(overlap),
        "trajectory_skipped_overlap_ids": overlap,
        "unknown_trajectory_ids": len(unknown_trajectories),
        "unknown_trajectory_id_values": unknown_trajectories,
        "unknown_skipped_ids": len(unknown_skipped),
        "unknown_skipped_id_values": unknown_skipped,
        "missing_candidate_ids": len(missing),
        "missing_candidate_id_values": missing,
        "complete_coverage": (
            trajectory_ids | skipped_ids == candidate_ids
            and not overlap
            and not candidate_duplicates
            and not trajectory_duplicates
            and not skipped_duplicates
        ),
    }

    termination_counters = {
        "overall": Counter(),
        "nq": Counter(),
        "hotpotqa": Counter(),
    }
    structure_reason_counts = Counter()
    structure_invalid_examples = []
    valid_records = []
    answer_em_rows = []
    mismatch_examples = []

    for record in trajectories:
        termination = record["runtime"]["termination_reason"]
        termination_counters["overall"][termination] += 1
        termination_counters[record["source"]][termination] += 1

        is_valid, reasons = validate_trajectory(record)
        record["_canonical_valid"] = is_valid
        record["_search_count"] = count_searches(record)
        if is_valid:
            valid_records.append(record)
        else:
            structure_reason_counts.update(reasons)
            if len(structure_invalid_examples) < 20:
                structure_invalid_examples.append(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "termination_reason": termination,
                        "reasons": reasons,
                    }
                )

        if is_valid and termination == "answer":
            prediction = extract_final_answer(record)
            exact_match = is_exact_match(prediction, record["answers"])
            row = {
                "id": record["id"],
                "source": record["source"],
                "search_count": record["_search_count"],
                "prediction": prediction,
                "gold_answers": record["answers"],
                "exact_match": exact_match,
            }
            answer_em_rows.append(row)
            if not exact_match and len(mismatch_examples) < 20:
                mismatch_examples.append(row.copy())

    answer_trajectories = [
        record
        for record in trajectories
        if record["runtime"]["termination_reason"] == "answer"
    ]
    zero_search_rows = [
        row for row in answer_em_rows if row["search_count"] == 0
    ]
    searched_rows = [
        row for row in answer_em_rows if row["search_count"] >= 1
    ]
    candidate_a = [row for row in answer_em_rows if row["exact_match"]]
    candidate_b = [
        row
        for row in candidate_a
        if row["search_count"] >= 1
    ]

    report = {
        "coverage": coverage,
        "termination": source_table(termination_counters),
        "canonical_structure": {
            "valid": len(valid_records),
            "invalid": len(trajectories) - len(valid_records),
            "invalid_reason_frequency": dict(
                sorted(structure_reason_counts.items())
            ),
            "invalid_examples": structure_invalid_examples,
        },
        "search_distribution": {
            "all_raw_trajectories": distribution_table(trajectories),
            "answer_trajectories": distribution_table(answer_trajectories),
        },
        "exact_match": {
            "all_valid_answer_trajectories": em_summary(answer_em_rows),
            "zero_search_answers": em_summary(zero_search_rows),
            "one_or_more_search_answers": em_summary(searched_rows),
        },
        "filter_simulation": {
            "raw_trajectories": len(trajectories),
            "canonical_structure_valid": len(valid_records),
            "valid_and_termination_answer": len(answer_em_rows),
            "valid_answer_normalized_em": len(candidate_a),
            "valid_answer_em_search_at_least_one": len(candidate_b),
            "correct_removed_only_for_zero_search": (
                len(candidate_a) - len(candidate_b)
            ),
            "candidate_a": {
                "nq": sum(row["source"] == "nq" for row in candidate_a),
                "hotpotqa": sum(
                    row["source"] == "hotpotqa" for row in candidate_a
                ),
                "total": len(candidate_a),
            },
            "candidate_b": {
                "nq": sum(row["source"] == "nq" for row in candidate_b),
                "hotpotqa": sum(
                    row["source"] == "hotpotqa" for row in candidate_b
                ),
                "total": len(candidate_b),
            },
        },
        "em_mismatch_examples": mismatch_examples,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("Teacher Raw Trajectory Audit")
    print("=" * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = build_report()
    print_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print()
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
