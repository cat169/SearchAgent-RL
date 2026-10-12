from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("data/processed/sft/all.jsonl")
OUTPUT_DIR = INPUT_PATH.parent
SEED = 123
SOURCES = ("nq", "hotpotqa", "aethersearch")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sample_ids(groups: dict, target: int, rng: random.Random) -> set[str]:
    # Proportional largest-remainder quotas, with sorted keys breaking ties.
    keys = sorted(groups)
    total = sum(len(groups[key]) for key in keys)
    if target < 0 or target > total:
        raise ValueError(f"Infeasible sampling target: {target}/{total}")
    if total == 0:
        return set()
    quotas = {key: target * len(groups[key]) // total for key in keys}
    ranked = sorted(keys, key=lambda key: (-(target * len(groups[key]) % total), key))
    for key in ranked[:target - sum(quotas.values())]:
        quotas[key] += 1
    return {
        record["id"]
        for key in keys
        for record in rng.sample(groups[key], quotas[key])
    }


def normalized_questions(records: list[dict]) -> set[str]:
    return {re.sub(r"\s+", " ", r["question"]).strip().casefold() for r in records}


def build_splits(records: list[dict]) -> dict[str, list[dict]]:
    if len(records) != 4246 or len({r["id"] for r in records}) != 4246:
        raise ValueError("Expected 4246 records with unique IDs")
    if Counter(r["search_count"] for r in records) != {0: 1244, 1: 1587, 2: 975, 3: 356, 4: 84}:
        raise ValueError("Unexpected input search_count distribution")

    shallow_groups = defaultdict(list)
    deep_groups = defaultdict(list)
    for record in records:
        groups = shallow_groups if record["search_count"] <= 1 else deep_groups
        groups[(record["source"], record["search_count"])].append(record)
    rng = random.Random(SEED)
    validation_ids = sample_ids(shallow_groups, 100, rng)
    validation_ids.update(sample_ids(deep_groups, 50, rng))
    validation = [r for r in records if r["id"] in validation_ids]
    remaining = [r for r in records if r["id"] not in validation_ids]
    narrow = [r for r in remaining if r["search_count"] <= 1]
    deep = [r for r in remaining if r["search_count"] >= 2]
    narrow_sources = Counter(r["source"] for r in narrow)
    deep_sources = Counter(r["source"] for r in deep)
    standard_ids = {r["id"] for r in deep}
    for source in SOURCES:
        target = narrow_sources[source] - deep_sources[source]
        if target < 0:
            raise ValueError(f"Validation allocation leaves infeasible {source} quota")
        groups = defaultdict(list)
        for record in narrow:
            if record["source"] == source:
                groups[record["search_count"]].append(record)
        standard_ids.update(sample_ids(groups, target, rng))
    standard = [r for r in remaining if r["id"] in standard_ids]
    outputs = {"validation": validation, "narrow": narrow, "standard": standard}
    validate_splits(records, outputs)
    return outputs


def validate_splits(records: list[dict], outputs: dict[str, list[dict]]) -> None:
    ids = {}
    for name, rows in outputs.items():
        expected = 150 if name == "validation" else 2731
        ids[name] = {r["id"] for r in rows}
        if len(rows) != expected or len(ids[name]) != len(rows):
            raise ValueError(f"Invalid {name} size or duplicate IDs")
        if rows != [r for r in records if r["id"] in ids[name]]:
            raise ValueError(f"{name} records or original order changed")
    validation = outputs["validation"]
    if sum(r["search_count"] <= 1 for r in validation) != 100:
        raise ValueError("Validation must contain 100 shallow and 50 deep records")
    if {r["search_count"] for r in validation} != set(range(5)):
        raise ValueError("Validation must cover all five search depths")
    remaining_shallow = {
        r["id"] for r in records
        if r["search_count"] <= 1 and r["id"] not in ids["validation"]
    }
    remaining_deep = {
        r["id"] for r in records
        if r["search_count"] >= 2 and r["id"] not in ids["validation"]
    }
    if ids["narrow"] != remaining_shallow:
        raise ValueError("Narrow must contain all remaining shallow records")
    standard_deep = {r["id"] for r in outputs["standard"] if r["search_count"] >= 2}
    if standard_deep != remaining_deep or len(standard_deep) != 1365:
        raise ValueError("Standard must contain all 1365 remaining deep records")
    if Counter(r["source"] for r in outputs["narrow"]) != Counter(r["source"] for r in outputs["standard"]):
        raise ValueError("Train source counts differ")
    for name in ("narrow", "standard"):
        if ids["validation"] & ids[name]:
            raise ValueError(f"Validation ID overlap with {name}")
        if normalized_questions(validation) & normalized_questions(outputs[name]):
            raise ValueError(f"Validation normalized Question overlap with {name}")


def print_statistics(name: str, records: list[dict]) -> None:
    counts = Counter((r["source"], r["search_count"]) for r in records)
    print(f"\n{name}: {len(records)}")
    print(f"{'source':<14}" + "".join(f"{depth:>8}" for depth in range(5)) + f"{'total':>8}")
    for source in SOURCES:
        print(f"{source:<14}" + "".join(f"{counts[source, depth]:>8}" for depth in range(5))
              + f"{sum(counts[source, depth] for depth in range(5)):>8}")


def main() -> None:
    before = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    records = read_jsonl(INPUT_PATH)
    print_statistics("all", records)
    outputs = build_splits(records)
    if outputs != build_splits(records):
        raise ValueError("Fixed-seed split reproduction failed")
    for name, rows in outputs.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in rows:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print_statistics(name, rows)
        print(f"Output: {path}")
    overlap = len({r["id"] for r in outputs["narrow"]} & {r["id"] for r in outputs["standard"]})
    after = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("all.jsonl SHA256 changed")
    print(f"\nNarrow / Standard intersection: {overlap}")
    print("Validation ID / normalized Question overlap: 0")
    print(f"Seed: {SEED}; reproducibility: PASS")
    print(f"all SHA256 before: {before}\nall SHA256 after : {after}")


if __name__ == "__main__":
    main()
