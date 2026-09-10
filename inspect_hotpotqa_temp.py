"""Targeted read-only checks for FlashRAG HotpotQA."""

from __future__ import annotations

import heapq
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from datasets import load_dataset


DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_CONFIG = "hotpotqa"
TOP_K = 10


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalized_answers(answers: list[str]) -> tuple[str, ...]:
    return tuple(normalize_whitespace(answer).casefold() for answer in answers)


def keep_top_k(
    heap: list[tuple[tuple[int, int], dict[str, Any]]],
    record: dict[str, Any],
    serial: int,
) -> None:
    # Larger lengths rank first; for ties, earlier dataset rows rank first.
    ranking_key = (record["length"], -serial)
    item = (ranking_key, record)
    if len(heap) < TOP_K:
        heapq.heappush(heap, item)
    elif ranking_key > heap[0][0]:
        heapq.heapreplace(heap, item)


def print_longest(title: str, heap: list[tuple[tuple[int, int], dict[str, Any]]]) -> None:
    print(f"\n=== {title} ===")
    records = [item[1] for item in heap]
    records.sort(key=lambda record: (-record["length"], record["serial"]))
    for rank, record in enumerate(records, start=1):
        print(f"\n[{rank}]")
        print(f"split: {record['split']}")
        print(f"row_index: {record['row_index']}")
        print(f"id: {record['id']}")
        print(f"length: {record['length']}")
        print(f"question: {record['question']!r}")
        print(f"golden_answers: {record['golden_answers']!r}")
        print(f"metadata.type: {record['metadata_type']!r}")
        print(f"metadata.level: {record['metadata_level']!r}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    train_question_counts: Counter[str] = Counter()
    for sample in dataset["train"]:
        train_question_counts[normalize_whitespace(sample["question"])] += 1

    duplicate_questions = {
        question for question, count in train_question_counts.items() if count > 1
    }
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    longest_questions: list[tuple[tuple[int, int], dict[str, Any]]] = []
    longest_answers: list[tuple[tuple[int, int], dict[str, Any]]] = []
    serial = 0

    for split, split_dataset in dataset.items():
        for row_index, sample in enumerate(split_dataset):
            question = sample["question"]
            answers = sample["golden_answers"]
            metadata = sample["metadata"]
            normalized_question = normalize_whitespace(question)

            if split == "train" and normalized_question in duplicate_questions:
                duplicate_groups[normalized_question].append(
                    {
                        "row_index": row_index,
                        "id": sample["id"],
                        "question": question,
                        "golden_answers": answers,
                        "metadata_type": metadata.get("type"),
                        "metadata_level": metadata.get("level"),
                        "supporting_fact_titles": metadata["supporting_facts"]["title"],
                    }
                )

            common_record = {
                "serial": serial,
                "split": split,
                "row_index": row_index,
                "id": sample["id"],
                "question": question,
                "golden_answers": answers,
                "metadata_type": metadata.get("type"),
                "metadata_level": metadata.get("level"),
            }
            question_record = dict(common_record, length=len(question))
            answer_record = dict(
                common_record,
                length=max((len(answer) for answer in answers), default=0),
            )
            keep_top_k(longest_questions, question_record, serial)
            keep_top_k(longest_answers, answer_record, serial)
            serial += 1

    consistent_groups = 0
    inconsistent_groups = 0
    print("=== Train duplicate normalized-question groups ===")
    for group_number, normalized_question in enumerate(
        sorted(duplicate_groups), start=1
    ):
        records = duplicate_groups[normalized_question]
        answer_forms = {
            normalized_answers(record["golden_answers"]) for record in records
        }
        answers_consistent = len(answer_forms) == 1
        if answers_consistent:
            consistent_groups += 1
        else:
            inconsistent_groups += 1

        print(f"\n--- Group {group_number} ---")
        print(f"normalized_question: {normalized_question!r}")
        print(f"answers_consistent: {answers_consistent}")
        for record in records:
            print("  record:")
            print(f"    row_index: {record['row_index']}")
            print(f"    id: {record['id']}")
            print(f"    original_question: {record['question']!r}")
            print(f"    golden_answers: {record['golden_answers']!r}")
            print(f"    metadata.type: {record['metadata_type']!r}")
            print(f"    metadata.level: {record['metadata_level']!r}")
            print(
                "    supporting_facts.title: "
                f"{record['supporting_fact_titles']!r}"
            )

    duplicate_row_count = sum(len(records) for records in duplicate_groups.values())
    print("\n=== Duplicate-question answer consistency summary ===")
    print(f"duplicate_question_groups: {len(duplicate_groups)}")
    print(f"duplicate_question_rows: {duplicate_row_count}")
    print(f"answer_consistent_groups: {consistent_groups}")
    print(f"answer_inconsistent_groups: {inconsistent_groups}")

    print_longest("Longest 10 questions", longest_questions)
    print_longest("Longest 10 answers", longest_answers)


if __name__ == "__main__":
    main()
