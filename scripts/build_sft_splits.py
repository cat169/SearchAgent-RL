from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("data/processed/sft/all.jsonl")
NARROW_PATH = Path("data/processed/sft/narrow.jsonl")
STANDARD_PATH = Path("data/processed/sft/standard.jsonl")
SEED = 123

SOURCE_TARGETS = {
    "nq": 746,
    "hotpotqa": 1063,
    "aethersearch": 1024,
}
EXCLUDED_SFT_IDS = {"nq_train_37781"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_narrow(records: list[dict]) -> list[dict]:
    return [record for record in records if record["search_count"] <= 1]


def sample_shallow(
    source_records: list[dict],
    target: int,
    source: str,
) -> set[str]:
    zero_search = [
        record for record in source_records if record["search_count"] == 0
    ]
    one_search = [
        record for record in source_records if record["search_count"] == 1
    ]
    shallow_total = len(zero_search) + len(one_search)

    if source == "aethersearch":
        target_zero = 0
    else:
        target_zero = round(target * len(zero_search) / shallow_total)
    target_one = target - target_zero

    rng = random.Random(SEED)
    selected = rng.sample(zero_search, target_zero)
    selected.extend(rng.sample(one_search, target_one))
    return {record["id"] for record in selected}


def build_standard(records: list[dict]) -> list[dict]:
    # Standard 保留全部多搜索轨迹，
    # 再从浅层轨迹中按 source 补足到与 Narrow 相同的数据量。
    deep_records = [record for record in records if record["search_count"] >= 2]
    selected_ids = {record["id"] for record in deep_records}

    for source, source_target in SOURCE_TARGETS.items():
        source_records = [
            record for record in records if record["source"] == source
        ]
        deep_count = sum(
            record["search_count"] >= 2 for record in source_records
        )
        shallow_target = source_target - deep_count
        selected_ids.update(
            sample_shallow(source_records, shallow_target, source)
        )

    # 先采样 ID，再按 all.jsonl 原顺序输出，
    # 避免随机采样破坏稳定的数据顺序。
    return [record for record in records if record["id"] in selected_ids]


def build_statistics(records: list[dict]) -> tuple[Counter, Counter, dict]:
    by_source = Counter(record["source"] for record in records)
    by_search_count = Counter(record["search_count"] for record in records)
    source_search = defaultdict(Counter)
    for record in records:
        source_search[record["source"]][record["search_count"]] += 1
    return by_source, by_search_count, source_search


def print_split_statistics(name: str, records: list[dict]) -> None:
    by_source, by_search_count, source_search = build_statistics(records)
    search_counts = sorted(by_search_count)

    print(name)
    print("-" * 12)
    print(f"Total                 : {len(records)}")
    print()
    print("By source:")
    for source in SOURCE_TARGETS:
        print(f"  {source:<20}: {by_source[source]}")
    print()
    print("Search count:")
    for search_count in search_counts:
        print(f"  {search_count} : {by_search_count[search_count]}")
    print()
    print("Source x search_count:")
    print()
    print(f"{'':<14}" + "".join(f"{count:>8}" for count in search_counts))
    for source in SOURCE_TARGETS:
        row = "".join(
            f"{source_search[source][count]:>8}" for count in search_counts
        )
        print(f"{source:<14}{row}")
    print()


def main() -> None:
    records = read_jsonl(INPUT_PATH)
    if len(records) != 4250:
        raise ValueError(f"Expected 4250 trajectories, got {len(records)}")

    all_ids = [record["id"] for record in records]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("all.jsonl contains duplicate IDs")

    narrow = build_narrow(records)
    standard = build_standard(records)

    # 保持原始冻结抽样不变，只在最终输出中排除无 think 的轨迹。
    narrow = [record for record in narrow if record["id"] not in EXCLUDED_SFT_IDS]
    standard = [record for record in standard if record["id"] not in EXCLUDED_SFT_IDS]

    if len(narrow) != 2832:
        raise ValueError(f"Expected 2832 Narrow records, got {len(narrow)}")
    if len(standard) != 2832:
        raise ValueError(
            f"Expected 2832 Standard records, got {len(standard)}"
        )

    narrow_ids = {record["id"] for record in narrow}
    standard_ids = {record["id"] for record in standard}
    if len(narrow_ids) != len(narrow):
        raise ValueError("Narrow contains duplicate IDs")
    if len(standard_ids) != len(standard):
        raise ValueError("Standard contains duplicate IDs")
    if any(record["search_count"] > 1 for record in narrow):
        raise ValueError("Narrow contains a multi-search trajectory")

    deep_ids = {
        record["id"] for record in records if record["search_count"] >= 2
    }
    if not deep_ids <= standard_ids:
        raise ValueError("Standard does not contain every deep trajectory")

    narrow_sources = Counter(record["source"] for record in narrow)
    standard_sources = Counter(record["source"] for record in standard)
    if narrow_sources != standard_sources:
        raise ValueError("Narrow and Standard source counts differ")
    final_source_targets = {**SOURCE_TARGETS, "nq": SOURCE_TARGETS["nq"] - 1}
    if dict(standard_sources) != final_source_targets:
        raise ValueError(
            f"Unexpected Standard source counts: {standard_sources}"
        )

    write_jsonl(NARROW_PATH, narrow)
    write_jsonl(STANDARD_PATH, standard)

    overlap_count = len(narrow_ids & standard_ids)
    print("SFT Split Build")
    print("=" * 50)
    print()
    print(f"All trajectories      : {len(records)}")
    print()
    print_split_statistics("Narrow-SFT", narrow)
    print_split_statistics("Standard-SFT", standard)
    print("Overlap:")
    print(f"  narrow ∩ standard   : {overlap_count}")
    print()
    print("Seed:")
    print(f"  {SEED}")
    print()
    print("Outputs:")
    print(f"  {NARROW_PATH}")
    print(f"  {STANDARD_PATH}")


if __name__ == "__main__":
    main()
