from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


INPUT_TEACHER = Path(
    "data/processed/teacher/clean_teacher_trajectories.jsonl"
)
INPUT_AETHERSEARCH = Path(
    "data/processed/aethersearch/train.jsonl"
)
UNIQUE_POOL_PATH = Path("data/processed/unified/unique_qa_pool.jsonl")
OUTPUT_PATH = Path("data/processed/sft/all.jsonl")

ALLOWED_EVENT_TYPES = {"think", "search", "information", "answer"}
OUTPUT_FIELDS = {
    "id",
    "source",
    "question",
    "answers",
    "search_count",
    "events",
}
EXCLUDED_SFT_IDS = {
    "nq_train_37781",
    "hotpotqa_train_52219",
    "nq_train_15129",
    "nq_train_13170",
}
THINK_PROTOCOL_TAGS = (
    "<search>", "</search>",
    "<information>", "</information>",
    "<answer>", "</answer>",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def count_searches(events: list[dict]) -> int:
    return sum(event["type"] == "search" for event in events)


def validate_events(record: dict) -> int:
    events = record["events"]
    if not events:
        raise ValueError(f"Empty events for {record['id']}")

    event_types = {event["type"] for event in events}
    unexpected = event_types - ALLOWED_EVENT_TYPES
    if unexpected:
        raise ValueError(
            f"Unexpected event types for {record['id']}: "
            f"{sorted(unexpected)}"
        )
    if events[-1]["type"] != "answer":
        raise ValueError(f"Final event is not answer for {record['id']}")

    return count_searches(events)


def clean_think_events(events: list[dict]) -> list[dict]:
    cleaned = []
    for event in events:
        if event["type"] == "think":
            content = event["content"]
            for tag in THINK_PROTOCOL_TAGS:
                content = content.replace(tag, "")
            cleaned.append({**event, "content": content})
        else:
            cleaned.append(event)
    return cleaned


def normalize_teacher(record: dict) -> dict:
    search_count = validate_events(record)
    return {
        "id": record["id"],
        "source": record["source"],
        "question": record["question"],
        "answers": record["answers"],
        "search_count": search_count,
        "events": clean_think_events(record["events"]),
    }


def normalize_aethersearch(record: dict) -> dict:
    # 两个来源统一从 events 重新统计 search_count，
    # 避免依赖各自历史字段定义。
    search_count = validate_events(record)
    if record.get("search_count") != search_count:
        raise ValueError(
            f"AetherSearch search_count mismatch for {record['id']}: "
            f"stored={record.get('search_count')}, events={search_count}"
        )
    return {
        "id": record["id"],
        "source": record["source"],
        "question": record["question"],
        "answers": record["answers"],
        "search_count": search_count,
        "events": clean_think_events(record["events"]),
    }


def build_statistics(records: list[dict]) -> tuple[Counter, Counter, dict]:
    by_source = Counter(record["source"] for record in records)
    search_distribution = Counter(
        record["search_count"] for record in records
    )
    source_search = defaultdict(Counter)
    for record in records:
        source_search[record["source"]][record["search_count"]] += 1
    return by_source, search_distribution, source_search


def print_statistics(records: list[dict]) -> None:
    by_source, search_distribution, source_search = build_statistics(records)
    search_counts = sorted(search_distribution)
    sources = ("nq", "hotpotqa", "aethersearch")

    print("SFT Data Build")
    print("=" * 50)
    print()
    print(f"Teacher trajectories      : {len(records) - by_source['aethersearch']}")
    print(f"AetherSearch trajectories : {by_source['aethersearch']}")
    print(f"Total trajectories        : {len(records)}")
    print()
    print("By source:")
    for source in sources:
        print(f"  {source:<12}: {by_source[source]}")
    print()
    print("Search count distribution:")
    for search_count in search_counts:
        print(f"  {search_count} : {search_distribution[search_count]}")
    print()
    print("Source x search_count:")
    print()
    print(f"{'':<14}" + "".join(f"{count:>8}" for count in search_counts))
    for source in sources:
        row = "".join(
            f"{source_search[source][count]:>8}" for count in search_counts
        )
        print(f"{source:<14}{row}")
    print()
    print("Output:")
    print(f"  {OUTPUT_PATH}")


def build_records() -> list[dict]:
    teacher_records = read_jsonl(INPUT_TEACHER)
    keep_ids = {
        record["uid"] for record in read_jsonl(UNIQUE_POOL_PATH)
        if record["source"] == "aethersearch"
    }
    if len(keep_ids) != 1999:
        raise ValueError(f"Expected 1999 AetherSearch keep IDs, got {len(keep_ids)}")
    input_records = read_jsonl(INPUT_AETHERSEARCH)
    if len(input_records) != 2000:
        raise ValueError(f"Expected 2000 AetherSearch trajectories, got {len(input_records)}")
    input_ids = {record["id"] for record in input_records}
    if len(input_ids) != len(input_records):
        raise ValueError("AetherSearch input IDs are not unique")
    aethersearch_records = [
        record for record in input_records if record["id"] in keep_ids
    ]
    if {record["id"] for record in aethersearch_records} != keep_ids:
        raise ValueError("AetherSearch output IDs do not match keep IDs")

    if len(teacher_records) != 2251:
        raise ValueError(
            f"Expected 2251 Teacher trajectories, got {len(teacher_records)}"
        )
    if len(aethersearch_records) != 1999:
        raise ValueError(
            "Expected 1999 AetherSearch trajectories, "
            f"got {len(aethersearch_records)}"
        )

    # 输出保持 Teacher 在前、AetherSearch 在后；
    # 此阶段不打乱数据，采样留给后续 Narrow/Standard 构造。
    output_records = [
        *(normalize_teacher(record) for record in teacher_records),
        *(normalize_aethersearch(record) for record in aethersearch_records),
    ]

    merged_ids = {record["id"] for record in output_records}
    missing_excluded_ids = EXCLUDED_SFT_IDS - merged_ids
    if missing_excluded_ids:
        raise ValueError(
            "Excluded SFT IDs are missing from merged data: "
            f"{sorted(missing_excluded_ids)}"
        )
    output_records = [
        record for record in output_records
        if record["id"] not in EXCLUDED_SFT_IDS
    ]

    if len(output_records) != 4246:
        raise ValueError(
            f"Expected 4246 total trajectories, got {len(output_records)}"
        )
    output_ids = [record["id"] for record in output_records]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("Merged trajectory IDs are not unique")
    if any(set(record) != OUTPUT_FIELDS for record in output_records):
        raise ValueError("Unified record fields do not match output schema")
    if any(
        record["search_count"] != count_searches(record["events"])
        for record in output_records
    ):
        raise ValueError("Output search_count does not match events")

    return output_records


def main() -> None:
    output_records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_statistics(output_records)


if __name__ == "__main__":
    main()
