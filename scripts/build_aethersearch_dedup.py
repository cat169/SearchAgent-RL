from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


UNIQUE_POOL_PATH = Path("data/processed/unified/unique_qa_pool.jsonl")
INPUT_PATH = Path("data/processed/aethersearch/train.jsonl")
OUTPUT_PATH = Path("data/processed/aethersearch/train_dedup.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    unique_pool = load_jsonl(UNIQUE_POOL_PATH)
    keep_ids = {
        record["uid"]
        for record in unique_pool
        if record["source"] == "aethersearch"
    }
    if len(keep_ids) != 1999:
        raise ValueError(
            f"Expected 1999 AetherSearch keep IDs, got {len(keep_ids)}"
        )

    input_records = load_jsonl(INPUT_PATH)
    if len(input_records) != 2000:
        raise ValueError(
            f"Expected 2000 input trajectories, got {len(input_records)}"
        )

    input_ids = [record["id"] for record in input_records]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("Input trajectory IDs are not unique")

    output_records = [
        record for record in input_records if record["id"] in keep_ids
    ]
    output_ids = [record["id"] for record in output_records]

    if len(output_records) != 1999:
        raise ValueError(
            f"Expected 1999 output trajectories, got {len(output_records)}"
        )
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("Output trajectory IDs are not unique")
    if set(output_ids) != keep_ids:
        raise ValueError("Output trajectory IDs do not match keep IDs")

    removed_ids = sorted(set(input_ids) - keep_ids)
    if len(removed_ids) != 1:
        raise ValueError(
            f"Expected one removed trajectory ID, got {len(removed_ids)}"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    search_counts = Counter(
        record["search_count"] for record in output_records
    )

    print("# AetherSearch Dedup Export")
    print()
    print(f"Input trajectories       : {len(input_records)}")
    print(f"Keep IDs from unique pool: {len(keep_ids)}")
    print(f"Output trajectories      : {len(output_records)}")
    print(f"Removed trajectories     : {len(removed_ids)}")
    print()
    print("Removed ID:")
    print(removed_ids[0])
    print()
    print("Search count distribution:")
    for search_count, count in sorted(search_counts.items()):
        print(f"{search_count} : {count}")
    print()
    print("Output:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
