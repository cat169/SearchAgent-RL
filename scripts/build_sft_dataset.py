"""从 Clean Teacher 轨迹构建 NQ + HotpotQA 正式 SFT 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from scripts.render_sft_messages import build_messages


SEED = 123
VALIDATION_SIZE = 150
DEFAULT_INPUT = Path("data/processed/teacher/clean_teacher_trajectories.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/sft/nq_hotpotqa_v1")
EVALUATION_FILES = (
    Path("data/processed/nq/dev.jsonl"),
    Path("data/processed/nq/test.jsonl"),
    Path("data/processed/hotpotqa/dev.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/validation canonical and Messages JSONL from clean Teacher trajectories."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    """只做确定性的 Unicode、大小写和空白归一化。"""
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def source_search_distribution(records: list[dict]) -> dict[str, dict[str, int]]:
    sources = ("nq", "hotpotqa")
    table = {
        source: {str(depth): 0 for depth in range(5)}
        for source in sources
    }
    for record in records:
        table[record["source"]][str(record["search_count"])] += 1
    for source in sources:
        table[source]["total"] = sum(table[source][str(depth)] for depth in range(5))
    table["total"] = {
        str(depth): sum(table[source][str(depth)] for source in sources)
        for depth in range(5)
    }
    table["total"]["total"] = len(records)
    return table


def validation_quotas(records: list[dict]) -> dict[str, int]:
    """按最终合格候选的来源比例，用最大余数法分配 150 条 Validation。"""
    source_counts = Counter(record["source"] for record in records)
    exact = {
        source: VALIDATION_SIZE * count / len(records)
        for source, count in source_counts.items()
    }
    quotas = {source: math.floor(value) for source, value in exact.items()}
    for source in sorted(source_counts, key=lambda item: (-(exact[item] - quotas[item]), item)):
        if sum(quotas.values()) == VALIDATION_SIZE:
            break
        quotas[source] += 1
    return quotas


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_messages(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(
                json.dumps({"messages": build_messages(record)}, ensure_ascii=False) + "\n"
            )


def main() -> None:
    args = parse_args()
    input_sha_before = file_sha256(args.input)
    clean_records = load_jsonl(args.input)
    clean_ids = [record["id"] for record in clean_records]
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("Clean Teacher 数据存在重复 ID")

    # 评测集只用于问题隔离，不生成新的 Evaluation 文件。
    evaluation_questions: set[str] = set()
    evaluation_file_stats = {}
    for path in EVALUATION_FILES:
        records = load_jsonl(path)
        normalized_questions = {normalize_question(record["question"]) for record in records}
        evaluation_questions.update(normalized_questions)
        evaluation_file_stats[path.as_posix()] = {
            "records": len(records),
            "unique_normalized_questions": len(normalized_questions),
            "sha256": file_sha256(path),
        }

    train_candidates = []
    excluded_non_train = []
    for record in clean_records:
        is_nq_train = record["source"] == "nq" and record["id"].startswith("nq_train_")
        is_hotpotqa_train = (
            record["source"] == "hotpotqa"
            and record["id"].startswith("hotpotqa_train_")
        )
        if is_nq_train or is_hotpotqa_train:
            train_candidates.append(record)
        else:
            excluded_non_train.append(record["id"])

    # 先隔离与 NQ dev/test、HotpotQA dev 完全相同的规范化问题。
    evaluation_overlaps = [
        record for record in train_candidates
        if normalize_question(record["question"]) in evaluation_questions
    ]
    overlap_ids = {record["id"] for record in evaluation_overlaps}
    isolated_candidates = [record for record in train_candidates if record["id"] not in overlap_ids]

    # 候选池内部按 Clean 文件原顺序保留第一条，保证结果稳定、容易追溯。
    seen_questions: dict[str, str] = {}
    deduplicated_candidates = []
    duplicate_questions = []
    for record in isolated_candidates:
        key = normalize_question(record["question"])
        if key in seen_questions:
            duplicate_questions.append({
                "dropped_id": record["id"],
                "kept_id": seen_questions[key],
                "normalized_question": key,
            })
            continue
        seen_questions[key] = record["id"]
        deduplicated_candidates.append(record)

    # Canonical 只增加真实 Search 计数，不改写原始问题、答案和任何 event 内容。
    compatible_records = []
    incompatible_records = []
    for source_record in deduplicated_candidates:
        canonical = {
            "id": source_record["id"],
            "source": source_record["source"],
            "question": source_record["question"],
            "answers": source_record["answers"],
            "search_count": sum(
                event["type"] == "search" for event in source_record["events"]
            ),
            "events": source_record["events"],
        }
        try:
            build_messages(canonical)
        except ValueError as error:
            incompatible_records.append({
                "id": canonical["id"],
                "reason": str(error),
                "event_types": [event["type"] for event in canonical["events"]],
            })
            continue
        compatible_records.append(canonical)

    if len(compatible_records) < VALIDATION_SIZE:
        raise ValueError(
            f"兼容轨迹只有 {len(compatible_records)} 条，无法划分 {VALIDATION_SIZE} 条 Validation"
        )

    quotas = validation_quotas(compatible_records)
    rng = random.Random(SEED)
    validation_ids: set[str] = set()
    for source in ("nq", "hotpotqa"):
        source_ids = sorted(
            record["id"] for record in compatible_records if record["source"] == source
        )
        validation_ids.update(rng.sample(source_ids, quotas[source]))

    # 两份输出继续沿用 Clean Teacher 文件中的相对顺序。
    validation_records = [
        record for record in compatible_records if record["id"] in validation_ids
    ]
    train_records = [
        record for record in compatible_records if record["id"] not in validation_ids
    ]

    train_ids = {record["id"] for record in train_records}
    validation_id_set = {record["id"] for record in validation_records}
    train_questions = {normalize_question(record["question"]) for record in train_records}
    validation_questions = {
        normalize_question(record["question"]) for record in validation_records
    }
    if train_ids & validation_id_set:
        raise RuntimeError("Train 与 Validation 存在 ID 重叠")
    if train_questions & validation_questions:
        raise RuntimeError("Train 与 Validation 存在规范化 Question 重叠")
    if (train_questions | validation_questions) & evaluation_questions:
        raise RuntimeError("SFT 数据与保留 Evaluation 问题发生重叠")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    messages_dir = args.output_dir / "messages"
    messages_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    train_messages_path = messages_dir / "train_messages.jsonl"
    validation_messages_path = messages_dir / "validation_messages.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    write_messages(train_messages_path, train_records)
    write_messages(validation_messages_path, validation_records)

    # 回读所有输出，确认行数、search_count 和 Messages 均与当前 Renderer 一致。
    for canonical_path, messages_path, expected_records in (
        (train_path, train_messages_path, train_records),
        (validation_path, validation_messages_path, validation_records),
    ):
        written_canonical = load_jsonl(canonical_path)
        written_messages = load_jsonl(messages_path)
        if written_canonical != expected_records:
            raise RuntimeError(f"Canonical 回读不一致: {canonical_path}")
        if len(written_messages) != len(expected_records):
            raise RuntimeError(f"Canonical/Messages 行数不一致: {messages_path}")
        for record, message_record in zip(expected_records, written_messages, strict=True):
            actual_search_count = sum(
                event["type"] == "search" for event in record["events"]
            )
            if record["search_count"] != actual_search_count:
                raise RuntimeError(f"search_count 不一致: {record['id']}")
            if message_record != {"messages": build_messages(record)}:
                raise RuntimeError(f"Messages 转换不一致: {record['id']}")

    input_sha_after = file_sha256(args.input)
    if input_sha_before != input_sha_after:
        raise RuntimeError("原始 Clean Teacher 文件在构建期间发生变化")

    manifest = {
        "version": "nq_hotpotqa_v1",
        "seed": SEED,
        "question_normalization": "Unicode NFKC + casefold + collapse/strip whitespace",
        "input": {
            "path": args.input.as_posix(),
            "records": len(clean_records),
            "sha256_before": input_sha_before,
            "sha256_after": input_sha_after,
        },
        "candidate_filter": {
            "train_prefix_candidates": len(train_candidates),
            "train_prefix_by_source": dict(Counter(record["source"] for record in train_candidates)),
            "excluded_non_train": len(excluded_non_train),
            "evaluation_question_overlaps": len(evaluation_overlaps),
            "evaluation_overlap_ids": [record["id"] for record in evaluation_overlaps],
            "internal_duplicate_questions_removed": len(duplicate_questions),
            "internal_duplicate_details": duplicate_questions,
            "renderer_incompatible_removed": len(incompatible_records),
            "renderer_incompatible_details": incompatible_records,
            "final_compatible_candidates": len(compatible_records),
        },
        "evaluation_inputs": evaluation_file_stats,
        "split": {
            "validation_size": VALIDATION_SIZE,
            "validation_source_quotas": quotas,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "train_validation_id_overlap": 0,
            "train_validation_normalized_question_overlap": 0,
            "sft_evaluation_normalized_question_overlap": 0,
            "validation_ids": sorted(validation_ids),
        },
        "distribution": {
            "train": source_search_distribution(train_records),
            "validation": source_search_distribution(validation_records),
        },
        "outputs": {
            path.as_posix(): {
                "records": len(records),
                "sha256": file_sha256(path),
            }
            for path, records in (
                (train_path, train_records),
                (validation_path, validation_records),
                (train_messages_path, train_records),
                (validation_messages_path, validation_records),
            )
        },
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    train_distribution = source_search_distribution(train_records)
    validation_distribution = source_search_distribution(validation_records)
    report = f"""# NQ + HotpotQA SFT v1 构建报告

## 输入与筛选

- Clean Teacher：{len(clean_records)} 条，SHA256 `{input_sha_before}`。
- 原始 train 候选：{len(train_candidates)} 条（NQ {sum(r['source'] == 'nq' for r in train_candidates)}；HotpotQA {sum(r['source'] == 'hotpotqa' for r in train_candidates)}）。
- 排除非原始 train：{len(excluded_non_train)} 条。
- 与保留评测问题规范化重复：{len(evaluation_overlaps)} 条。
- 候选内部规范化重复问题：{len(duplicate_questions)} 条。
- Renderer 不兼容：{len(incompatible_records)} 条。
- 最终合格候选：{len(compatible_records)} 条。

Renderer 不兼容详情：`{json.dumps(incompatible_records, ensure_ascii=False)}`

## 划分

- 随机种子：{SEED}
- Validation 来源配额：`{json.dumps(quotas, ensure_ascii=False)}`
- Train：{len(train_records)} 条
- Validation：{len(validation_records)} 条
- Train/Validation ID 重叠：0
- Train/Validation 规范化问题重叠：0
- SFT 与 NQ dev/test、HotpotQA dev 规范化问题重叠：0

## Train：来源 × Search 次数

{json.dumps(train_distribution, ensure_ascii=False, indent=2)}

## Validation：来源 × Search 次数

{json.dumps(validation_distribution, ensure_ascii=False, indent=2)}

## 转换与完整性

- Canonical 仅保留 id/source/question/answers/search_count/events；events 内容和顺序未改写。
- Messages 全部由 `scripts.render_sft_messages.build_messages()` 生成并逐条回读比对。
- Gold Answers 未放入 Messages。
- 原始 Clean Teacher 构建前后 SHA256 一致。
- 完整文件哈希和 Validation ID 清单见 `split_manifest.json`。
"""
    (args.output_dir / "build_report.md").write_text(report, encoding="utf-8")

    print("# NQ + HotpotQA SFT v1")
    print(f"Clean records          : {len(clean_records)}")
    print(f"Original train         : {len(train_candidates)}")
    print(f"Evaluation overlaps    : {len(evaluation_overlaps)}")
    print(f"Internal duplicates    : {len(duplicate_questions)}")
    print(f"Renderer incompatible  : {len(incompatible_records)}")
    print(f"Train / Validation     : {len(train_records)} / {len(validation_records)}")
    print(f"Output                 : {args.output_dir}")


if __name__ == "__main__":
    main()
