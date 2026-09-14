"""Convert AetherSearch SFT data to the canonical trajectory JSONL schema."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_NAME = "muradil211/AetherSearch_SFT"
REQUIRED_SPLIT = "train"
REQUIRED_FIELDS = {
    "id",
    "question",
    "trajectory_type",
    "search_count",
    "full_trajectory_text",
}
ALLOWED_TRAJECTORY_TYPES = {"single_search", "multi_search"}
ALLOWED_EVENT_TYPES = {"think", "search", "information", "answer"}
ROLE_PATTERN = re.compile(
    r"<\|im_start\|>([^\r\n<]+)\r?\n?(.*?)<\|im_end\|>", re.DOTALL
)
EVENT_PATTERN = re.compile(
    r"<(think|search|information|answer)>(.*?)</\1>", re.DOTALL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/aethersearch"),
        help="Output directory (default: data/processed/aethersearch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing train.jsonl to be replaced.",
    )
    return parser.parse_args()


def validate_source_schema(dataset: Any) -> None:
    actual_splits = set(dataset.keys())
    if actual_splits != {REQUIRED_SPLIT}:
        raise ValueError(
            "AetherSearch must contain only the train split; "
            f"actual splits={sorted(actual_splits)}"
        )

    fields = set(dataset[REQUIRED_SPLIT].column_names)
    if fields != REQUIRED_FIELDS:
        raise ValueError(
            "AetherSearch source fields differ from the audited schema; "
            f"expected={sorted(REQUIRED_FIELDS)}, actual={sorted(fields)}"
        )


def parse_assistant_events(
    trajectory_text: Any,
    *,
    sample_id: str,
    question: str,
) -> list[dict[str, str]]:
    if not isinstance(trajectory_text, str) or not trajectory_text:
        raise ValueError("full_trajectory_text must be a non-empty string")

    role_matches = list(ROLE_PATTERN.finditer(trajectory_text))
    roles = [match.group(1).strip() for match in role_matches]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(
            "trajectory roles must be exactly system -> user -> assistant; "
            f"actual={roles}"
        )
    if ROLE_PATTERN.sub("", trajectory_text).strip():
        raise ValueError("trajectory contains text outside the three ChatML messages")
    if (
        trajectory_text.count("<|im_start|>") != 3
        or trajectory_text.count("<|im_end|>") != 3
    ):
        raise ValueError("trajectory must contain exactly three ChatML message pairs")

    user_content = role_matches[1].group(2)
    if question not in user_content:
        raise ValueError("top-level question is not present in the user message")

    assistant_content = role_matches[2].group(2)
    events: list[dict[str, str]] = []
    cursor = 0
    for match in EVENT_PATTERN.finditer(assistant_content):
        if assistant_content[cursor : match.start()].strip():
            raise ValueError(
                "assistant content contains text outside supported control blocks"
            )
        event_type = match.group(1)
        content = match.group(2)
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event_type!r}")
        if not content.strip():
            raise ValueError(f"event {event_type!r} has empty content")
        events.append({"type": event_type, "content": content})
        cursor = match.end()

    if assistant_content[cursor:].strip():
        raise ValueError("assistant content has trailing text outside an event block")
    if not events:
        raise ValueError("no trajectory events were extracted")
    return events


def transform_sample(sample: dict[str, Any], row_index: int) -> dict[str, Any]:
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("id must be a non-empty string")

    question = sample.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a non-empty string")

    trajectory_type = sample.get("trajectory_type")
    if trajectory_type not in ALLOWED_TRAJECTORY_TYPES:
        raise ValueError(f"invalid trajectory_type: {trajectory_type!r}")

    search_count = sample.get("search_count")
    if type(search_count) is not int or search_count < 1:
        raise ValueError("search_count must be an integer greater than or equal to 1")

    events = parse_assistant_events(
        sample.get("full_trajectory_text"),
        sample_id=sample_id,
        question=question,
    )
    if any(event["type"] not in ALLOWED_EVENT_TYPES for event in events):
        raise ValueError("events contain an unsupported type")
    if any(not event["content"].strip() for event in events):
        raise ValueError("events contain empty content")

    event_counts = Counter(event["type"] for event in events)
    if event_counts["search"] != search_count:
        raise ValueError(
            f"search event count {event_counts['search']} does not match "
            f"search_count {search_count}"
        )
    if event_counts["information"] != event_counts["search"]:
        raise ValueError(
            "information event count does not match the search event count"
        )
    if event_counts["answer"] != 1:
        raise ValueError(
            f"expected exactly one answer event, found {event_counts['answer']}"
        )
    if events[-1]["type"] != "answer":
        raise ValueError("the answer event must be the final event")
    if trajectory_type == "single_search" and search_count != 1:
        raise ValueError("single_search requires search_count == 1")
    if trajectory_type == "multi_search" and search_count < 2:
        raise ValueError("multi_search requires search_count >= 2")

    answer = events[-1]["content"]
    answers = [answer]
    if answers[0] != events[-1]["content"]:
        raise ValueError("answers[0] does not match the answer event content")

    return {
        "id": f"aethersearch_{sample_id}",
        "source": "aethersearch",
        "question": question,
        "answers": answers,
        "trajectory_type": trajectory_type,
        "search_count": search_count,
        "events": events,
    }


def print_statistics(statistics: dict[str, Any], output_path: Path) -> None:
    print(f"read={statistics['read']}")
    print(f"written={statistics['written']}")
    print(f"failed={statistics['failed']}")
    print(f"single_search={statistics['single_search']}")
    print(f"multi_search={statistics['multi_search']}")
    print(f"search_events={statistics['search_events']}")
    print(f"information_events={statistics['information_events']}")
    print(f"think_events={statistics['think_events']}")
    print(f"answer_events={statistics['answer_events']}")
    print(
        "search_count_distribution="
        + json.dumps(dict(sorted(statistics["search_count_distribution"].items())))
    )
    print(f"output_path={output_path}")


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from datasets import load_dataset

    output_path = args.output_dir / "train.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    dataset = load_dataset(DATASET_NAME)
    validate_source_schema(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    seen_ids: set[str] = set()
    first_record: dict[str, Any] | None = None
    statistics: dict[str, Any] = {
        "read": 0,
        "written": 0,
        "failed": 0,
        "single_search": 0,
        "multi_search": 0,
        "search_events": 0,
        "information_events": 0,
        "think_events": 0,
        "answer_events": 0,
        "search_count_distribution": Counter(),
    }

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_index, sample in enumerate(dataset[REQUIRED_SPLIT]):
                statistics["read"] += 1
                sample_id = sample.get("id")
                try:
                    record = transform_sample(sample, row_index)
                    if record["id"] in seen_ids:
                        raise ValueError(f"duplicate canonical id: {record['id']}")
                    seen_ids.add(record["id"])
                except ValueError as error:
                    statistics["failed"] += 1
                    print(
                        f"FAILED row={row_index}, id={sample_id!r}: {error}",
                        flush=True,
                    )
                    continue

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                statistics["written"] += 1
                statistics[record["trajectory_type"]] += 1
                statistics["search_count_distribution"][record["search_count"]] += 1
                event_counts = Counter(event["type"] for event in record["events"])
                for event_type in ALLOWED_EVENT_TYPES:
                    statistics[f"{event_type}_events"] += event_counts[event_type]
                if first_record is None:
                    first_record = record

        if statistics["failed"]:
            print_statistics(statistics, output_path)
            raise RuntimeError(
                f"Conversion failed for {statistics['failed']} sample(s); "
                "no output file was published"
            )
        if statistics["written"] != statistics["read"]:
            raise RuntimeError("written count does not match read count")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print_statistics(statistics, output_path)
    print("first_converted_record=")
    print(json.dumps(first_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
