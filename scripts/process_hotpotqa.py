"""Clean FlashRAG HotpotQA and write one JSONL file per source split."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_CONFIG = "hotpotqa"
REQUIRED_SPLITS = ("train", "dev")
REQUIRED_FIELDS = {"id", "question", "golden_answers", "metadata"}
EXCLUDED_SAMPLE_IDS = frozenset(
    {
        "train_27806",
        "train_77497",
        "train_11353",
        "train_3027",
        "train_37183",
        "train_57311",
        "train_76227",
        "train_18953",
        "train_63809",
        "train_83381",
        "train_54069",
        "train_67184",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/hotpotqa"),
        help="Output directory (default: data/processed/hotpotqa).",
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
        help="Allow existing train.jsonl and dev.jsonl to be replaced.",
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
    sample_id: str,
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
                "empty after whitespace normalization"
            )
        if cleaned_answer != answer:
            normalized_count += 1

        answer_key = cleaned_answer.casefold()
        if answer_key in seen_answers:
            duplicate_count += 1
            continue
        seen_answers.add(answer_key)
        cleaned_answers.append(cleaned_answer)

    if not cleaned_answers:
        raise ValueError(f"Invalid golden_answers ({context}): no answers remain")
    return cleaned_answers, normalized_count, duplicate_count


def clean_supporting_titles(
    titles: Any,
    *,
    split: str,
    row_index: int,
    sample_id: str,
) -> tuple[list[str], int, int]:
    context = f"split={split}, row={row_index}, id={sample_id!r}"
    if not isinstance(titles, list) or not titles:
        raise ValueError(
            f"Invalid supporting_facts.title ({context}): expected a non-empty list"
        )

    cleaned_titles: list[str] = []
    seen_titles: set[str] = set()
    normalized_count = 0
    duplicate_count = 0

    for title_index, title in enumerate(titles):
        if not isinstance(title, str) or not title:
            raise ValueError(
                f"Invalid supporting_facts.title[{title_index}] ({context}): "
                "expected a non-empty string"
            )
        cleaned_title = normalize_whitespace(title)
        if not cleaned_title:
            raise ValueError(
                f"Invalid supporting_facts.title[{title_index}] ({context}): "
                "empty after whitespace normalization"
            )
        if cleaned_title != title:
            normalized_count += 1

        title_key = cleaned_title.casefold()
        if title_key in seen_titles:
            duplicate_count += 1
            continue
        seen_titles.add(title_key)
        cleaned_titles.append(cleaned_title)

    if not cleaned_titles:
        raise ValueError(
            f"Invalid supporting_facts.title ({context}): no titles remain"
        )
    return cleaned_titles, normalized_count, duplicate_count


def validate_source_schema(dataset: Any) -> None:
    missing_splits = [split for split in REQUIRED_SPLITS if split not in dataset]
    if missing_splits:
        raise ValueError(
            f"FlashRAG HotpotQA is missing required splits: {', '.join(missing_splits)}"
        )

    for split in REQUIRED_SPLITS:
        fields = set(dataset[split].column_names)
        missing_fields = REQUIRED_FIELDS - fields
        if missing_fields:
            raise ValueError(
                f"FlashRAG HotpotQA split={split} is missing fields "
                f"{sorted(missing_fields)}; actual fields={sorted(fields)}"
            )


def transform_sample(
    sample: dict[str, Any],
    split: str,
    row_index: int,
    normalized_id: str,
) -> tuple[dict[str, Any], int, int, int, int, int]:
    context = f"split={split}, row={row_index}, id={normalized_id!r}"

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
        sample_id=normalized_id,
    )

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata ({context}): expected a dictionary")

    metadata_type = metadata.get("type")
    if not isinstance(metadata_type, str) or not metadata_type:
        raise ValueError(f"Invalid metadata.type ({context}): expected a non-empty string")
    cleaned_type = normalize_whitespace(metadata_type)
    if not cleaned_type:
        raise ValueError(
            f"Invalid metadata.type ({context}): empty after whitespace normalization"
        )

    metadata_level = metadata.get("level")
    if not isinstance(metadata_level, str) or not metadata_level:
        raise ValueError(
            f"Invalid metadata.level ({context}): expected a non-empty string"
        )
    cleaned_level = normalize_whitespace(metadata_level)
    if not cleaned_level:
        raise ValueError(
            f"Invalid metadata.level ({context}): empty after whitespace normalization"
        )

    supporting_facts = metadata.get("supporting_facts")
    if not isinstance(supporting_facts, dict):
        raise ValueError(
            f"Invalid metadata.supporting_facts ({context}): expected a dictionary"
        )
    (
        supporting_titles,
        normalized_titles,
        duplicate_titles,
    ) = clean_supporting_titles(
        supporting_facts.get("title"),
        split=split,
        row_index=row_index,
        sample_id=normalized_id,
    )

    record = {
        "uid": f"hotpotqa_{normalized_id}",
        "source": "hotpotqa",
        "source_split": split,
        "question": cleaned_question,
        "gold_answers": cleaned_answers,
        "metadata": {
            "original_id": normalized_id,
            "type": cleaned_type,
            "level": cleaned_level,
            "supporting_titles": supporting_titles,
        },
    }
    return (
        record,
        int(cleaned_question != question),
        normalized_answers,
        duplicate_answers,
        normalized_titles,
        duplicate_titles,
    )


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
        "excluded_samples": 0,
        "normalized_questions": 0,
        "normalized_answers": 0,
        "duplicate_answers_removed": 0,
        "normalized_supporting_titles": 0,
        "duplicate_supporting_titles_removed": 0,
    }

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_index, sample in enumerate(split_dataset):
                if max_samples is not None and row_index >= max_samples:
                    break
                statistics["read"] += 1

                raw_id = sample.get("id")
                error_context = f"split={split}, row={row_index}, id={raw_id!r}"
                if not isinstance(raw_id, str) or not raw_id:
                    raise ValueError(
                        f"Invalid id ({error_context}): expected a non-empty string"
                    )
                normalized_id = normalize_whitespace(raw_id)
                if not normalized_id:
                    raise ValueError(
                        f"Invalid id ({error_context}): empty after whitespace normalization"
                    )

                if normalized_id in EXCLUDED_SAMPLE_IDS:
                    statistics["excluded_samples"] += 1
                    continue

                (
                    record,
                    normalized_questions,
                    normalized_answers,
                    duplicate_answers,
                    normalized_titles,
                    duplicate_titles,
                ) = transform_sample(sample, split, row_index, normalized_id)

                uid = record["uid"]
                if uid in seen_uids:
                    raise ValueError(
                        f"Duplicate uid (split={split}, row={row_index}, "
                        f"id={normalized_id!r}): {uid}"
                    )
                seen_uids.add(uid)

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                statistics["written"] += 1
                statistics["normalized_questions"] += normalized_questions
                statistics["normalized_answers"] += normalized_answers
                statistics["duplicate_answers_removed"] += duplicate_answers
                statistics["normalized_supporting_titles"] += normalized_titles
                statistics["duplicate_supporting_titles_removed"] += duplicate_titles

        if statistics["written"] != statistics["read"] - statistics["excluded_samples"]:
            raise RuntimeError(
                f"Record-count invariant failed for split={split}: {statistics}"
            )
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
        print(f"{split}: " + ", ".join(f"{key}={value}" for key, value in statistics.items()))

    totals = {
        key: sum(statistics[key] for statistics in all_statistics.values())
        for key in next(iter(all_statistics.values()))
    }
    print(f"overall: " + ", ".join(f"{key}={value}" for key, value in totals.items()))


if __name__ == "__main__":
    main()
