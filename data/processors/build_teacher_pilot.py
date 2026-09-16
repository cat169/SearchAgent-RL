from __future__ import annotations

import json
import random
from pathlib import Path


PILOT_PER_SOURCE = 50
SEED = 123
INPUT_PATH = Path(
    "data/processed/splits/teacher_sft_candidates.jsonl"
)
OUTPUT_PATH = Path(
    "data/processed/teacher/pilot_teacher_candidates.jsonl"
)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]

    selected = []
    source_counts = {}

    for source in ["nq", "hotpotqa"]:
        source_records = sorted(
            (record for record in records if record["source"] == source),
            key=lambda record: record["uid"],
        )
        random.Random(SEED).shuffle(source_records)
        pilot_records = source_records[:PILOT_PER_SOURCE]

        if len(pilot_records) != PILOT_PER_SOURCE:
            raise ValueError(f"Not enough {source} records for pilot")

        selected.extend(pilot_records)
        source_counts[source] = len(pilot_records)

    selected.sort(key=lambda record: record["uid"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in selected:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Teacher pilot built")
    print(f"NQ       : {source_counts['nq']}")
    print(f"HotpotQA : {source_counts['hotpotqa']}")
    print(f"Total    : {len(selected)}")
    print(f"Seed     : {SEED}")


if __name__ == "__main__":
    main()
