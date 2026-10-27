"""冻结 NQ test + HotpotQA dev 的独立 Agent Evaluation 数据集。"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from scripts.build_sft_dataset import file_sha256, load_jsonl, normalize_question


SEED = 123
SAMPLES_PER_SOURCE = 100
SFT_TRAIN_PATH = Path("data/processed/sft/nq_hotpotqa_v1/train.jsonl")
SFT_VALIDATION_PATH = Path("data/processed/sft/nq_hotpotqa_v1/validation.jsonl")
OUTPUT_DIR = Path("data/processed/evaluation/nq_hotpotqa_v1")
SOURCE_FILES = (
    ("nq", Path("data/processed/nq/test.jsonl")),
    ("hotpotqa", Path("data/processed/hotpotqa/dev.jsonl")),
)


def main() -> None:
    input_paths = [path for _, path in SOURCE_FILES]
    protected_paths = input_paths + [SFT_TRAIN_PATH, SFT_VALIDATION_PATH]
    sha256_before = {path.as_posix(): file_sha256(path) for path in protected_paths}

    sft_train = load_jsonl(SFT_TRAIN_PATH)
    sft_validation = load_jsonl(SFT_VALIDATION_PATH)
    sft_records = sft_train + sft_validation
    sft_ids = {record["id"] for record in sft_records}
    sft_questions = {normalize_question(record["question"]) for record in sft_records}

    rng = random.Random(SEED)
    selected_records = []
    selected_ids: set[str] = set()
    selected_questions: set[str] = set()
    source_statistics = {}

    for expected_source, input_path in SOURCE_FILES:
        candidates = sorted(load_jsonl(input_path), key=lambda record: record["uid"])
        rng.shuffle(candidates)

        selected_for_source = []
        skipped_sft_id = 0
        skipped_sft_question = 0
        skipped_eval_duplicate_id = 0
        skipped_eval_duplicate_question = 0

        # 按固定洗牌顺序遍历。遇到污染或重复候选就跳过，直到当前来源取满 100 条。
        for candidate in candidates:
            uid = candidate["uid"]
            question_key = normalize_question(candidate["question"])
            if uid in sft_ids:
                skipped_sft_id += 1
                continue
            if question_key in sft_questions:
                skipped_sft_question += 1
                continue
            if uid in selected_ids:
                skipped_eval_duplicate_id += 1
                continue
            if question_key in selected_questions:
                skipped_eval_duplicate_question += 1
                continue

            record = {
                "id": uid,
                "source": candidate["source"],
                "source_split": candidate["source_split"],
                "question": candidate["question"],
                "answers": candidate["gold_answers"],
            }
            if record["source"] != expected_source:
                raise ValueError(
                    f"来源不一致: expected={expected_source}, id={uid}, actual={record['source']}"
                )
            selected_for_source.append(record)
            selected_ids.add(uid)
            selected_questions.add(question_key)
            if len(selected_for_source) == SAMPLES_PER_SOURCE:
                break

        if len(selected_for_source) != SAMPLES_PER_SOURCE:
            raise ValueError(
                f"{expected_source} 只有 {len(selected_for_source)} 条合格 Evaluation 候选"
            )
        selected_records.extend(selected_for_source)
        source_statistics[expected_source] = {
            "input_records": len(candidates),
            "selected_records": len(selected_for_source),
            "skipped_sft_id_overlap": skipped_sft_id,
            "skipped_sft_question_overlap": skipped_sft_question,
            "skipped_eval_duplicate_id": skipped_eval_duplicate_id,
            "skipped_eval_duplicate_question": skipped_eval_duplicate_question,
        }

    if len(selected_records) != 2 * SAMPLES_PER_SOURCE:
        raise RuntimeError("Evaluation 总数不是 200")
    if len(selected_ids) != len(selected_records):
        raise RuntimeError("Evaluation 存在重复 ID")
    if len(selected_questions) != len(selected_records):
        raise RuntimeError("Evaluation 存在重复规范化 Question")

    eval_ids = {record["id"] for record in selected_records}
    eval_questions = {
        normalize_question(record["question"]) for record in selected_records
    }
    id_overlap = eval_ids & sft_ids
    question_overlap = eval_questions & sft_questions
    if id_overlap or question_overlap:
        raise RuntimeError(
            f"Evaluation 与 SFT 未完全隔离: ids={len(id_overlap)}, questions={len(question_overlap)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUT_DIR / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in selected_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    written_records = load_jsonl(eval_path)
    if written_records != selected_records:
        raise RuntimeError("eval.jsonl 回读内容与构建结果不一致")

    sha256_after = {path.as_posix(): file_sha256(path) for path in protected_paths}
    if sha256_before != sha256_after:
        raise RuntimeError("构建期间原始 QA 或 SFT 文件发生变化")

    manifest = {
        "version": "nq_hotpotqa_v1",
        "seed": SEED,
        "samples_per_source": SAMPLES_PER_SOURCE,
        "total_records": len(selected_records),
        "source_distribution": dict(Counter(record["source"] for record in selected_records)),
        "source_split_distribution": dict(
            Counter(record["source_split"] for record in selected_records)
        ),
        "sampling_algorithm": (
            "For each source in order [nq, hotpotqa], sort candidates by uid, "
            "shuffle with one Random(123) instance, then take the first 100 candidates "
            "whose ID and normalized question do not overlap SFT or earlier evaluation selections."
        ),
        "question_normalization": "Unicode NFKC + casefold + collapse/strip whitespace",
        "source_statistics": source_statistics,
        "selected_ids": [record["id"] for record in selected_records],
        "input_sha256": {
            path.as_posix(): sha256_before[path.as_posix()] for path in input_paths
        },
        "sft_input_sha256": {
            SFT_TRAIN_PATH.as_posix(): sha256_before[SFT_TRAIN_PATH.as_posix()],
            SFT_VALIDATION_PATH.as_posix(): sha256_before[SFT_VALIDATION_PATH.as_posix()],
        },
        "evaluation_sha256": file_sha256(eval_path),
        "isolation": {
            "evaluation_duplicate_ids": len(selected_records) - len(selected_ids),
            "evaluation_duplicate_normalized_questions": (
                len(selected_records) - len(selected_questions)
            ),
            "sft_id_overlap": len(id_overlap),
            "sft_normalized_question_overlap": len(question_overlap),
        },
    }
    manifest_path = OUTPUT_DIR / "eval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("# Frozen Agent Evaluation v1")
    print(f"Records       : {len(selected_records)}")
    print(f"Sources       : {manifest['source_distribution']}")
    print(f"SFT overlaps  : IDs={len(id_overlap)}, questions={len(question_overlap)}")
    print(f"Evaluation SHA: {manifest['evaluation_sha256']}")
    print(f"Output        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
