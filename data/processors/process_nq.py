"""Convert FlashRAG NQ into the project's clean QA JSONL format."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_CONFIG = "nq"
REQUIRED_SOURCE_FIELDS = {"id", "question", "golden_answers"}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: str) -> str:
    """Strip a string and collapse all consecutive whitespace to one space."""

    return WHITESPACE_RE.sub(" ", value).strip()


def clean_answers(raw_answers: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    if isinstance(raw_answers, str):
        answers: Iterable[Any] = [raw_answers]
    elif isinstance(raw_answers, (list, tuple)):
        answers = raw_answers
    else:
        return None, "invalid_type"

    cleaned: List[str] = []
    seen = set()
    for answer in answers:
        if not isinstance(answer, str):
            return None, "invalid_type"
        normalized = normalize_whitespace(answer)
        if not normalized:
            continue
        deduplication_key = normalized.casefold()
        if deduplication_key not in seen:
            seen.add(deduplication_key)
            cleaned.append(normalized)

    if not cleaned:
        return None, "empty_gold_answers"
    return cleaned, None


def transform_sample(sample: Dict[str, Any], split: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    question = sample.get("question")
    original_id = sample.get("id")

    if not isinstance(question, str) or not isinstance(original_id, str):
        return None, "invalid_type"

    question = normalize_whitespace(question)
    if not question:
        return None, "empty_question"

    original_id = original_id.strip()
    if not original_id:
        return None, "empty_original_id"

    answers, error = clean_answers(sample.get("golden_answers"))
    if error is not None:
        return None, error

    return {
        "uid": f"nq_{original_id}",
        "source": "nq",
        "question": question,
        "gold_answers": answers,
        "source_split": split,
        "metadata": {"original_id": original_id},
    }, None


def validate_source_schema(dataset: Any) -> None:
    problems = []
    for split, split_dataset in dataset.items():
        fields = set(split_dataset.features.keys())
        missing = REQUIRED_SOURCE_FIELDS - fields
        if missing:
            sample = split_dataset[0] if len(split_dataset) else None
            problems.append(
                f"split={split!r}, fields={sorted(fields)!r}, "
                f"missing={sorted(missing)!r}, sample={sample!r}"
            )

    if problems:
        details = "\n".join(problems)
        raise ValueError(
            "FlashRAG NQ schema differs from the required source schema; "
            f"processing stopped.\n{details}"
        )


def process_dataset(dataset: Any, output_path: Path, max_samples: Optional[int]) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    split_stats = {
        split: {"total": 0, "kept": 0, "dropped": 0, "drop_reasons": Counter()}
        for split in dataset.keys()
    }
    seen_uids = set()
    processed = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            stop = False
            for split, split_dataset in dataset.items():
                for sample in split_dataset:
                    if max_samples is not None and processed >= max_samples:
                        stop = True
                        break

                    processed += 1
                    stats = split_stats[split]
                    stats["total"] += 1
                    transformed, drop_reason = transform_sample(sample, split)
                    if drop_reason is not None:
                        stats["dropped"] += 1
                        stats["drop_reasons"][drop_reason] += 1
                        continue

                    uid = transformed["uid"]
                    if uid in seen_uids:
                        raise ValueError(f"Duplicate uid detected: {uid}")
                    seen_uids.add(uid)

                    output_file.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                    stats["kept"] += 1

                if stop:
                    break

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    overall = {
        key: sum(stats[key] for stats in split_stats.values())
        for key in ("total", "kept", "dropped")
    }
    return {"splits": split_stats, "overall": overall}


def print_statistics(statistics: Dict[str, Any], output_path: Path) -> None:
    print("Processing statistics")
    for split, stats in statistics["splits"].items():
        line = (
            f"  {split}: total={stats['total']}, kept={stats['kept']}, "
            f"dropped={stats['dropped']}"
        )
        if stats["drop_reasons"]:
            reasons = ", ".join(
                f"{reason}={count}" for reason, count in sorted(stats["drop_reasons"].items())
            )
            line += f" ({reasons})"
        print(line)

    overall = statistics["overall"]
    print(
        f"  overall: total={overall['total']}, kept={overall['kept']}, "
        f"dropped={overall['dropped']}"
    )
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/nq_clean.jsonl"),
        help="Destination JSONL path (default: data/processed/nq_clean.jsonl).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug-only global limit on source samples; do not use for experiment splitting.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Missing dependency 'datasets'. Install project requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    validate_source_schema(dataset)
    statistics = process_dataset(dataset, args.output_path, args.max_samples)
    print_statistics(statistics, args.output_path)


if __name__ == "__main__":
    main()
