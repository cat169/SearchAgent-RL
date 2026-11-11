"""构建并冻结 NQ + HotpotQA GRPO Train/Validation，不改变 SFT 或 Eval。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.build_sft_dataset import file_sha256, load_jsonl, normalize_question


VERSION = "nq_hotpotqa_grpo_20261111"
SEED = 123
TRAIN_PER_SOURCE = 1000
VALIDATION_PER_SOURCE = 100
OUTPUT_DIR = Path("data/processed/grpo/nq_hotpotqa")
SOURCE_FILES = (
    ("nq", Path("data/processed/nq/train.jsonl")),
    ("hotpotqa", Path("data/processed/hotpotqa/train.jsonl")),
)
PROTECTED_FILES = (
    (Path("data/processed/sft/nq_hotpotqa_v1/train.jsonl"), "id"),
    (Path("data/processed/sft/nq_hotpotqa_v1/validation.jsonl"), "id"),
    (Path("data/processed/evaluation/nq_hotpotqa_v1/eval.jsonl"), "id"),
    (Path("data/processed/nq/dev.jsonl"), "uid"),
    (Path("data/processed/nq/test.jsonl"), "uid"),
    (Path("data/processed/hotpotqa/dev.jsonl"), "uid"),
)


def build_splits(source_records, excluded_ids, excluded_questions):
    outputs = {"train": [], "validation": []}
    seen_questions = set()
    for source, _ in SOURCE_FILES:
        candidates = []
        # 排序后去重，重复问题保留最小 UID；同时避免两个来源的问题重复。
        for record in sorted(source_records[source], key=lambda row: row["uid"]):
            question = normalize_question(record["question"])
            if record["uid"] in excluded_ids or question in excluded_questions:
                continue
            if question in seen_questions:
                continue
            seen_questions.add(question)
            candidates.append(record)

        # 两个来源各自使用 Random(123)，不依赖全局随机状态。
        random.Random(SEED).shuffle(candidates)
        end = TRAIN_PER_SOURCE + VALIDATION_PER_SOURCE
        if len(candidates) < end:
            raise ValueError(f"Not enough eligible {source} questions: {len(candidates)}")
        for split, selected in (
            ("train", candidates[:TRAIN_PER_SOURCE]),
            ("validation", candidates[TRAIN_PER_SOURCE:end]),
        ):
            outputs[split].extend(
                {
                    "id": record["uid"],
                    "source": record["source"],
                    "source_split": record["source_split"],
                    "question": record["question"],
                    "answers": record["gold_answers"],
                }
                for record in selected
            )
    return outputs


def serialize(records):
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


def main() -> None:
    input_paths = [path for _, path in SOURCE_FILES] + [path for path, _ in PROTECTED_FILES]
    input_hashes = {path.as_posix(): file_sha256(path) for path in input_paths}
    source_records = {source: load_jsonl(path) for source, path in SOURCE_FILES}
    excluded_ids = set()
    excluded_questions = set()
    for path, id_field in PROTECTED_FILES:
        records = load_jsonl(path)
        excluded_ids.update(record[id_field] for record in records)
        excluded_questions.update(normalize_question(record["question"]) for record in records)

    outputs = build_splits(source_records, excluded_ids, excluded_questions)
    seen_ids = set()
    seen_questions = set()
    for split, expected_per_source in (
        ("train", TRAIN_PER_SOURCE),
        ("validation", VALIDATION_PER_SOURCE),
    ):
        records = outputs[split]
        counts = Counter(record["source"] for record in records)
        expected = {source: expected_per_source for source, _ in SOURCE_FILES}
        if len(records) != 2 * expected_per_source or dict(counts) != expected:
            raise RuntimeError(f"Unexpected {split} counts: {dict(counts)}")
        ids = {record["id"] for record in records}
        questions = {normalize_question(record["question"]) for record in records}
        if len(ids) != len(records) or len(questions) != len(records):
            raise RuntimeError(f"Duplicate ID or normalized question within {split}")
        if ids & (excluded_ids | seen_ids) or questions & (excluded_questions | seen_questions):
            raise RuntimeError(f"ID or normalized question isolation failed for {split}")
        seen_ids.update(ids)
        seen_questions.update(questions)

    contents = {split: serialize(records) for split, records in outputs.items()}
    # 重新计算整个划分并比较实际 JSONL 字节，验证成员、顺序和序列化均可复现。
    repeated = build_splits(source_records, excluded_ids, excluded_questions)
    if contents != {split: serialize(records) for split, records in repeated.items()}:
        raise RuntimeError("Repeated split construction is not reproducible")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_info = {}
    for split, records in outputs.items():
        path = OUTPUT_DIR / f"{split}.jsonl"
        path.write_bytes(contents[split])
        expected_hash = hashlib.sha256(contents[split]).hexdigest()
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Output hash mismatch: {path}")
        output_info[split] = {
            "path": path.as_posix(),
            "records": len(records),
            "sources": dict(Counter(record["source"] for record in records)),
            "selected_ids": [record["id"] for record in records],
            "sha256": expected_hash,
        }

    if input_hashes != {path.as_posix(): file_sha256(path) for path in input_paths}:
        raise RuntimeError("An input QA, SFT or Evaluation file changed during construction")

    manifest = {
        "version": VERSION,
        "seed": SEED,
        "input_sha256": input_hashes,
        "question_normalization": "Unicode NFKC + casefold + collapse/strip whitespace",
        "selection_rule": (
            "Exclude IDs and normalized questions from all protected files. "
            "Process sources in order [nq, hotpotqa], sort by uid and keep the first "
            "occurrence of each normalized question across sources. "
            "Shuffle each source independently with Random(123); take first 1000 "
            "for train, next 100 for validation. Concatenate sources in that order."
        ),
        "protected_files": [path.as_posix() for path, _ in PROTECTED_FILES],
        "outputs": output_info,
        "verification": {
            "train_validation_id_overlap": 0,
            "train_validation_normalized_question_overlap": 0,
            "protected_id_overlap": 0,
            "protected_normalized_question_overlap": 0,
            "same_seed_reproducible": True,
            "input_hashes_unchanged": True,
        },
    }
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for split, info in output_info.items():
        print(f"{split}: {info['records']} records, sources={info['sources']}")
        print(f"  SHA256: {info['sha256']}")
    print("Isolation, reproducibility and unchanged input hashes: PASS")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
