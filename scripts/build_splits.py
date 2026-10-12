from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


SEED = 123
TEACHER_PER_SOURCE = 2000
GRPO_PER_SOURCE = 1000
VAL_PER_SOURCE = 500
TEST_PER_SOURCE = 1000

INPUT_PATH = Path("data/processed/unified/unique_qa_pool.jsonl")
OUTPUT_DIR = Path("data/processed/splits")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_source(records: list[dict]) -> dict[str, list[dict]]:
    records.sort(key=lambda record: record["uid"])
    random.Random(SEED).shuffle(records)

    teacher_end = TEACHER_PER_SOURCE
    grpo_end = teacher_end + GRPO_PER_SOURCE
    val_end = grpo_end + VAL_PER_SOURCE
    test_end = val_end + TEST_PER_SOURCE
    return {
        "teacher_sft_candidates": records[:teacher_end],
        "grpo": records[teacher_end:grpo_end],
        "val": records[grpo_end:val_end],
        "test": records[val_end:test_end],
        "reserve": records[test_end:],
    }


def main() -> None:
    input_records = read_jsonl(INPUT_PATH)
    nq_records = [r for r in input_records if r["source"] == "nq"]
    hotpotqa_records = [
        r for r in input_records if r["source"] == "hotpotqa"
    ]

    if len(nq_records) != 91415:
        raise ValueError(f"Expected 91415 NQ records, got {len(nq_records)}")
    if len(hotpotqa_records) != 97822:
        raise ValueError(
            f"Expected 97822 HotpotQA records, got {len(hotpotqa_records)}"
        )

    nq_splits = split_source(nq_records)
    hotpotqa_splits = split_source(hotpotqa_records)
    split_names = [
        "teacher_sft_candidates",
        "grpo",
        "val",
        "test",
        "reserve",
    ]
    outputs = {
        name: sorted(
            nq_splits[name] + hotpotqa_splits[name],
            key=lambda record: record["uid"],
        )
        for name in split_names
    }

    per_source_counts = {
        "teacher_sft_candidates": TEACHER_PER_SOURCE,
        "grpo": GRPO_PER_SOURCE,
        "val": VAL_PER_SOURCE,
        "test": TEST_PER_SOURCE,
    }
    expected_sources = {
        name: {"nq": count, "hotpotqa": count}
        for name, count in per_source_counts.items()
    }
    allocated = sum(per_source_counts.values())
    expected_sources["reserve"] = {
        "nq": len(nq_records) - allocated,
        "hotpotqa": len(hotpotqa_records) - allocated,
    }
    expected_totals = {
        name: sum(counts.values()) for name, counts in expected_sources.items()
    }

    uid_sets = {}
    source_counts = {}
    for name, records in outputs.items():
        if len(records) != expected_totals[name]:
            raise ValueError(f"Unexpected {name} size: {len(records)}")

        source_counts[name] = Counter(r["source"] for r in records)
        if dict(source_counts[name]) != expected_sources[name]:
            raise ValueError(
                f"Unexpected source counts for {name}: {source_counts[name]}"
            )

        uid_sets[name] = {r["uid"] for r in records}
        if len(uid_sets[name]) != len(records):
            raise ValueError(f"Duplicate uid within {name}")

    total_output_records = sum(len(records) for records in outputs.values())
    uid_union = set().union(*uid_sets.values())
    overlap_count = total_output_records - len(uid_union)
    if overlap_count != 0:
        raise ValueError(f"Output splits overlap by {overlap_count} uids")
    if len(uid_union) != 189237:
        raise ValueError(f"Expected uid union 189237, got {len(uid_union)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in outputs.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("Experiment Split Build Completed")
    print("=" * 60)
    print("Input:")
    print(f"  NQ           : {len(nq_records)}")
    print(f"  HotpotQA     : {len(hotpotqa_records)}")
    print(f"  Total        : {len(nq_records) + len(hotpotqa_records)}")

    display_names = {
        "teacher_sft_candidates": "Teacher-SFT candidates",
        "grpo": "GRPO",
        "val": "Val",
        "test": "Test",
        "reserve": "Reserve",
    }
    for name in split_names:
        print()
        print(f"{display_names[name]}:")
        print(f"  NQ           : {source_counts[name]['nq']}")
        print(f"  HotpotQA     : {source_counts[name]['hotpotqa']}")
        print(f"  Total        : {len(outputs[name])}")

    print()
    print("UID overlap:")
    print(f"  teacher/grpo/val/test/reserve = {overlap_count}")
    print()
    print("Seed:")
    print(f"  {SEED}")


if __name__ == "__main__":
    main()
