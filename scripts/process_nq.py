"""Clean FlashRAG NQ and write one JSONL file for each official split."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_CONFIG = "nq"
REQUIRED_SPLITS = ("train", "dev", "test")
REQUIRED_FIELDS = {"id", "question", "golden_answers"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/nq"),
        help="Output directory (default: data/processed/nq).",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Process at most the first N samples of each split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing split files to be replaced.",
    )
    args = parser.parse_args()
    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        parser.error("--max_samples_per_split must be a positive integer")
    return args


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_answers(
    answers: Any,
    *,
    split: str,
    row_index: int,
    sample_id: Any,
) -> tuple[list[str], int, int]:
    context = f"split={split}, row={row_index}, id={sample_id!r}"
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"Invalid golden_answers ({context}): expected a non-empty list")

    cleaned_answers: list[str] = []
    seen_answers: set[str] = set()
    normalized_count = 0
    duplicate_count = 0

    for answer_index, answer in enumerate(answers):
        if not isinstance(answer, str) or not answer:
            raise ValueError(
                f"Invalid golden_answers[{answer_index}] ({context}): "
                "expected a non-empty string"
            )

        cleaned_answer = normalize_whitespace(answer)
        if not cleaned_answer:
            raise ValueError(
                f"Invalid golden_answers[{answer_index}] ({context}): "
                "answer is empty after whitespace normalization"
            )
        if cleaned_answer != answer:
            normalized_count += 1

        deduplication_key = cleaned_answer.casefold()
        if deduplication_key in seen_answers:
            duplicate_count += 1
            continue

        seen_answers.add(deduplication_key)
        cleaned_answers.append(cleaned_answer)

    if not cleaned_answers:
        raise ValueError(
            f"Invalid golden_answers ({context}): no answers remain after cleaning"
        )
    return cleaned_answers, normalized_count, duplicate_count


def validate_source_schema(dataset: Any) -> None:
    missing_splits = [split for split in REQUIRED_SPLITS if split not in dataset]
    if missing_splits:
        raise ValueError(
            f"FlashRAG NQ is missing required splits: {', '.join(missing_splits)}"
        )

    for split in REQUIRED_SPLITS:
        fields = set(dataset[split].column_names)
        missing_fields = REQUIRED_FIELDS - fields
        if missing_fields:
            raise ValueError(
                f"FlashRAG NQ split={split} is missing fields "
                f"{sorted(missing_fields)}; actual fields={sorted(fields)}"
            )


def transform_sample(
    sample: dict[str, Any], split: str, row_index: int
) -> tuple[dict[str, Any], int, int, int]:
    sample_id = sample.get("id")
    context = f"split={split}, row={row_index}, id={sample_id!r}"

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Invalid id ({context}): expected a non-empty string")

    question = sample.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError(f"Invalid question ({context}): expected a non-empty string")
    cleaned_question = normalize_whitespace(question)
    if not cleaned_question:
        raise ValueError(
            f"Invalid question ({context}): empty after whitespace normalization"
        )

    cleaned_answers, normalized_answers, duplicate_answers = clean_answers(
        sample.get("golden_answers"),
        split=split,
        row_index=row_index,
        sample_id=sample_id,
    )
    record = {
        "uid": f"nq_{sample_id}",
        "source": "nq",
        "source_split": split,
        "question": cleaned_question,
        "gold_answers": cleaned_answers,
        "metadata": {"original_id": sample_id},
    }
    return record, int(cleaned_question != question), normalized_answers, duplicate_answers


def process_split(
    split_dataset: Any,
    split: str,
    output_path: Path,
    max_samples: int | None,
    seen_uids: set[str],
) -> dict[str, int]:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    statistics = {
        "read": 0,
        "written": 0,
        "normalized_questions": 0,
        "normalized_answers": 0,
        "duplicate_answers_removed": 0,
    }

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_index, sample in enumerate(split_dataset):
                if max_samples is not None and row_index >= max_samples:
                    break

                statistics["read"] += 1
                (
                    record,
                    normalized_questions,
                    normalized_answers,
                    duplicate_answers,
                ) = transform_sample(sample, split, row_index)

                uid = record["uid"]
                if uid in seen_uids:
                    raise ValueError(
                        f"Duplicate uid (split={split}, row={row_index}, "
                        f"id={sample.get('id')!r}): {uid}"
                    )
                seen_uids.add(uid)

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                statistics["written"] += 1
                statistics["normalized_questions"] += normalized_questions
                statistics["normalized_answers"] += normalized_answers
                statistics["duplicate_answers_removed"] += duplicate_answers

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return statistics


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    validate_source_schema(dataset)

    output_paths = {
        split: args.output_dir / f"{split}.jsonl" for split in REQUIRED_SPLITS
    }
    existing_paths = [path for path in output_paths.values() if path.exists()]
    if existing_paths and not args.overwrite:
        paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Output file(s) already exist: {paths}. Pass --overwrite to replace them."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen_uids: set[str] = set()
    all_statistics: dict[str, dict[str, int]] = {}

    for split in REQUIRED_SPLITS:
        statistics = process_split(
            dataset[split],
            split,
            output_paths[split],
            args.max_samples_per_split,
            seen_uids,
        )
        all_statistics[split] = statistics
        print(
            f"{split}: read={statistics['read']}, written={statistics['written']}, "
            f"normalized_questions={statistics['normalized_questions']}, "
            f"normalized_answers={statistics['normalized_answers']}, "
            f"duplicate_answers_removed={statistics['duplicate_answers_removed']}"
        )

    totals = {
        key: sum(statistics[key] for statistics in all_statistics.values())
        for key in next(iter(all_statistics.values()))
    }
    print(
        f"overall: read={totals['read']}, written={totals['written']}, "
        f"normalized_questions={totals['normalized_questions']}, "
        f"normalized_answers={totals['normalized_answers']}, "
        f"duplicate_answers_removed={totals['duplicate_answers_removed']}"
    )


if __name__ == "__main__":
    main()
