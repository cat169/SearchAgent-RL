from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from audit_teacher_trajectories import (
    count_searches,
    extract_final_answer,
    is_exact_match,
    load_jsonl,
    normalize_answer,
    validate_trajectory,
)


INPUT_PATH = Path(
    "data/processed/teacher/raw_teacher_trajectories.jsonl"
)
CLEAN_PATH = Path(
    "data/processed/teacher/clean_teacher_trajectories.jsonl"
)
REPORT_PATH = Path(
    "data/processed/teacher/teacher_filter_report.json"
)

MATCH_REASONS = (
    "normalized_em",
    "date_equivalent",
    "number_equivalent",
    "multi_answer_equivalent",
)
DATE_FORMATS = (
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y-%m-%d",
)
SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
MULTI_ANSWER_CONNECTORS = {"and", "&"}


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_date(text: str) -> date | None:
    value = collapse_whitespace(text)
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            pass
    return None


def date_equivalent(prediction: str, gold: str) -> bool:
    prediction_date = parse_date(prediction)
    gold_date = parse_date(gold)
    return (
        prediction_date is not None
        and gold_date is not None
        and prediction_date == gold_date
    )


def parse_simple_integer(text: str) -> int | None:
    value = collapse_whitespace(text).casefold().replace("-", " ")
    if re.fullmatch(r"\d+", value):
        return int(value)

    tokens = value.split()
    if len(tokens) == 1:
        return SMALL_NUMBERS.get(tokens[0], TENS.get(tokens[0]))
    if (
        len(tokens) == 2
        and tokens[0] in TENS
        and tokens[1] in SMALL_NUMBERS
        and 0 < SMALL_NUMBERS[tokens[1]] < 10
    ):
        return TENS[tokens[0]] + SMALL_NUMBERS[tokens[1]]
    return None


def number_equivalent(prediction: str, gold: str) -> bool:
    prediction_number = parse_simple_integer(prediction)
    gold_number = parse_simple_integer(gold)
    return (
        prediction_number is not None
        and gold_number is not None
        and prediction_number == gold_number
    )


def answer_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_answer(text).split()
        if token not in MULTI_ANSWER_CONNECTORS
    ]


def multi_answer_equivalent(
    prediction: str,
    gold_answers: list[str],
) -> bool:
    # Treat golds as independent list items only when none overlap. This avoids
    # accepting common NQ alias lists as complete multi-answer responses.
    if len(gold_answers) < 2:
        return False

    normalized_golds = [normalize_answer(gold) for gold in gold_answers]
    if any(not gold for gold in normalized_golds):
        return False
    if len(set(normalized_golds)) != len(normalized_golds):
        return False

    gold_tokens = [answer_tokens(gold) for gold in gold_answers]
    if any(not tokens for tokens in gold_tokens):
        return False
    gold_token_sets = [set(tokens) for tokens in gold_tokens]
    for left_index, left_tokens in enumerate(gold_token_sets):
        for right_tokens in gold_token_sets[left_index + 1 :]:
            if left_tokens & right_tokens:
                return False

    combined_gold = Counter(
        token for tokens in gold_tokens for token in tokens
    )
    prediction_counter = Counter(answer_tokens(prediction))
    return bool(prediction_counter) and prediction_counter == combined_gold


def high_precision_answer_match(
    prediction: str,
    gold_answers: list[str],
) -> str | None:
    if is_exact_match(prediction, gold_answers):
        return "normalized_em"
    if any(date_equivalent(prediction, gold) for gold in gold_answers):
        return "date_equivalent"
    if any(number_equivalent(prediction, gold) for gold in gold_answers):
        return "number_equivalent"
    if multi_answer_equivalent(prediction, gold_answers):
        return "multi_answer_equivalent"
    return None


def filter_trajectories() -> tuple[list[dict], dict]:
    records = load_jsonl(
        INPUT_PATH,
        {"id", "source", "question", "answers", "events", "runtime"},
    )
    accepted_records = []
    accepted_by_source = Counter()
    accepted_by_reason = Counter()
    accepted_by_source_and_reason = {
        "nq": Counter(),
        "hotpotqa": Counter(),
    }
    accepted_search_distribution = Counter()
    rejected = Counter()

    for record in records:
        is_valid, _ = validate_trajectory(record)
        if not is_valid:
            rejected["canonical_invalid"] += 1
            continue

        if record["runtime"].get("termination_reason") != "answer":
            rejected["non_answer"] += 1
            continue

        prediction = extract_final_answer(record)
        match_reason = (
            high_precision_answer_match(prediction, record["answers"])
            if prediction is not None
            else None
        )
        if match_reason is None:
            rejected["correctness_mismatch"] += 1
            continue

        accepted_records.append(record)
        source = record["source"]
        accepted_by_source[source] += 1
        accepted_by_reason[match_reason] += 1
        accepted_by_source_and_reason[source][match_reason] += 1
        accepted_search_distribution[count_searches(record)] += 1

    rejected_total = sum(rejected.values())
    report = {
        "input": {"raw_trajectories": len(records)},
        "accepted": {
            "total": len(accepted_records),
            "by_source": {
                source: accepted_by_source[source]
                for source in ("nq", "hotpotqa")
            },
            "by_match_reason": {
                reason: accepted_by_reason[reason]
                for reason in MATCH_REASONS
            },
        },
        "rejected": {
            "total": rejected_total,
            "canonical_invalid": rejected["canonical_invalid"],
            "non_answer": rejected["non_answer"],
            "correctness_mismatch": rejected["correctness_mismatch"],
        },
        "search_distribution": {
            "accepted": {
                str(search_count): accepted_search_distribution[search_count]
                for search_count in sorted(accepted_search_distribution)
            }
        },
        "accepted_by_source_and_match_reason": {
            source: {
                reason: accepted_by_source_and_reason[source][reason]
                for reason in MATCH_REASONS
            }
            for source in ("nq", "hotpotqa")
        },
    }

    if len(accepted_records) + rejected_total != len(records):
        raise ValueError("Accepted and rejected counts do not cover all input")
    accepted_ids = [record["id"] for record in accepted_records]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("Accepted trajectory IDs are not unique")

    return accepted_records, report


def write_outputs(records: list[dict], report: dict) -> None:
    with CLEAN_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with REPORT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    with CLEAN_PATH.open("r", encoding="utf-8") as input_file:
        output_lines = sum(1 for line in input_file if line.strip())
    if output_lines != report["accepted"]["total"]:
        raise ValueError("Clean output line count does not match accepted total")


def print_summary(report: dict) -> None:
    print("Teacher Trajectory Filter")
    print("=" * 50)
    print()
    print(f"Raw trajectories       : {report['input']['raw_trajectories']}")
    print()
    print(f"Accepted                : {report['accepted']['total']}")
    for reason in MATCH_REASONS:
        label = "multi_answer_equiv" if reason == "multi_answer_equivalent" else reason
        print(f"  {label:<22}: {report['accepted']['by_match_reason'][reason]}")
    print()
    print(f"Rejected                : {report['rejected']['total']}")
    for reason in ("canonical_invalid", "non_answer", "correctness_mismatch"):
        print(f"  {reason:<22}: {report['rejected'][reason]}")
    print()
    print("Accepted by source:")
    print(f"  {'NQ':<22}: {report['accepted']['by_source']['nq']}")
    print(f"  {'HotpotQA':<22}: {report['accepted']['by_source']['hotpotqa']}")
    print()
    print("Accepted search distribution:")
    for search_count, count in report["search_distribution"]["accepted"].items():
        print(f"  {search_count:<22}: {count}")
    print()
    print("Outputs:")
    print(f"  {CLEAN_PATH}")
    print(f"  {REPORT_PATH}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    accepted_records, report = filter_trajectories()
    write_outputs(accepted_records, report)
    print_summary(report)


if __name__ == "__main__":
    main()
