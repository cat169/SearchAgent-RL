from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_sft
from scripts.render_sft_messages import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit real TRL SFTTrainer labels on representative trajectories."
    )
    parser.add_argument("--train-messages", type=Path, required=True)
    parser.add_argument("--train-canonical", type=Path, required=True)
    parser.add_argument("--val-messages", type=Path, required=True)
    parser.add_argument("--val-canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=train_sft.MODEL_NAME)
    parser.add_argument(
        "--max-audit-samples", type=int, default=28,
        help="Total representative trajectories across train and validation (20-30 recommended).",
    )
    parser.add_argument(
        "--batch-examples", type=int, default=6,
        help="Number of complete real batches saved with token arrays.",
    )
    args = parser.parse_args()
    if args.max_audit_samples <= 0:
        parser.error("--max-audit-samples must be positive")
    if args.batch_examples <= 0:
        parser.error("--batch-examples must be positive")
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_aligned(messages_path: Path, canonical_path: Path, split: str) -> list[dict]:
    message_records = load_jsonl(messages_path)
    canonical_records = load_jsonl(canonical_path)
    if len(message_records) != len(canonical_records):
        raise ValueError(
            f"Line-count mismatch: {messages_path}={len(message_records)}, "
            f"{canonical_path}={len(canonical_records)}"
        )
    rows = []
    seen_ids = set()
    for line_index, (message_record, record) in enumerate(
        zip(message_records, canonical_records)
    ):
        uid = record["id"]
        if uid in seen_ids:
            raise ValueError(f"Duplicate canonical ID in {canonical_path}: {uid}")
        seen_ids.add(uid)
        if set(message_record) != {"messages"}:
            raise ValueError(
                f"Unexpected Messages fields at {messages_path}:{line_index + 1}: "
                f"{sorted(message_record)}"
            )
        if message_record["messages"] != build_messages(record):
            raise ValueError(
                f"Messages/Canonical mismatch for {uid} at line {line_index + 1}; "
                "the export differs from build_messages(record)"
            )
        rows.append(
            {
                "uid": uid, "split": split, "record": record,
                "messages": message_record["messages"], "line_index": line_index,
            }
        )
    return rows


def trajectory_characters(row: dict) -> int:
    return len(row["record"]["question"]) + sum(
        len(event["content"]) for event in row["record"]["events"]
    )


def representative_rows(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    chosen = []
    chosen_ids = set()

    def add(row: dict) -> None:
        if row["uid"] not in chosen_ids and len(chosen) < count:
            chosen.append(row)
            chosen_ids.add(row["uid"])

    strata = {}
    for row in rows:
        key = (row["record"]["source"], row["record"]["search_count"])
        if key not in strata or trajectory_characters(row) > trajectory_characters(strata[key]):
            strata[key] = row
    for key in sorted(strata):
        add(strata[key])
    for row in sorted(rows, key=trajectory_characters, reverse=True)[:5]:
        add(row)

    aether_thinks = Counter(
        event["content"]
        for row in rows if row["record"]["source"] == "aethersearch"
        for event in row["record"]["events"] if event["type"] == "think"
    )
    if aether_thinks:
        common_think, _ = aether_thinks.most_common(1)[0]
        for row in rows:
            if row["record"]["source"] == "aethersearch" and any(
                event["type"] == "think" and event["content"] == common_think
                for event in row["record"]["events"]
            ):
                add(row)
                break
    for row in rows:
        add(row)
    return chosen


def select_audit_rows(
    train_rows: list[dict], val_rows: list[dict], total: int
) -> tuple[list[dict], list[dict]]:
    validation_count = min(4, len(val_rows), max(1, total // 5))
    train_count = total - validation_count
    if train_count <= 0:
        raise ValueError("Audit size is too small to include training records")
    return (
        representative_rows(train_rows, min(train_count, len(train_rows))),
        representative_rows(val_rows, validation_count),
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def make_trainer(args: argparse.Namespace, train_rows: list[dict], val_rows: list[dict]):
    train_sft.MODEL_NAME = args.model_path
    trainer_args = SimpleNamespace(output_dir=args.output_dir / "runtime", max_steps=1)
    train_dataset = Dataset.from_list(
        [{"messages": row["messages"], "_audit_uid": row["uid"]} for row in train_rows]
    )
    val_dataset = Dataset.from_list(
        [{"messages": row["messages"], "_audit_uid": row["uid"]} for row in val_rows]
    )
    return train_sft.build_trainer(trainer_args, train_dataset, val_dataset)


def effective_chat_template(trainer) -> str:
    template = trainer.chat_template or trainer.processing_class.chat_template
    if not template:
        raise RuntimeError("The formal Trainer has no effective chat template")
    if "{% generation" not in template and "{%- generation" not in template:
        raise RuntimeError(
            "The formal Trainer chat template has no generation markers required "
            "by assistant_only_loss"
        )
    return template


def render_ids(tokenizer, messages: list[dict], template: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=True, return_dict=True,
        add_generation_prompt=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def message_boundaries(
    tokenizer, messages: list[dict], template: str
) -> tuple[list[int], list[tuple[int, int]]]:
    full_ids = render_ids(tokenizer, messages, template)
    boundaries = []
    previous = 0
    for message_index in range(len(messages)):
        prefix = render_ids(tokenizer, messages[: message_index + 1], template)
        if full_ids[: len(prefix)] != prefix:
            raise RuntimeError(
                f"Effective template is not prefix-preserving at message {message_index}"
            )
        boundaries.append((previous, len(prefix)))
        previous = len(prefix)
    return full_ids, boundaries


def retained_slice(trainer, full_length: int) -> tuple[int, int]:
    maximum = trainer.args.max_length
    if maximum is None or full_length <= maximum:
        return 0, full_length
    if trainer.args.truncation_mode == "keep_start":
        return 0, maximum
    if trainer.args.truncation_mode == "keep_end":
        return full_length - maximum, full_length
    raise RuntimeError(f"Unsupported truncation mode: {trainer.args.truncation_mode}")


def locate_processed(
    dataset, expected_rows: list[dict], split: str
) -> tuple[dict[str, dict], list[str]]:
    if "_audit_uid" not in dataset.column_names:
        raise RuntimeError(
            f"{split}: Trainer preprocessing removed _audit_uid; safe alignment is impossible"
        )
    processed = {}
    for index, example in enumerate(dataset):
        uid = example["_audit_uid"]
        if uid in processed:
            raise RuntimeError(f"{split}: duplicate preprocessed UID: {uid}")
        if "input_ids" not in example or "labels" not in example:
            raise RuntimeError(
                f"{split}/{uid}: preprocessing did not retain input_ids and labels"
            )
        processed[uid] = {"index": index, "example": example}
    expected = {row["uid"] for row in expected_rows}
    extra = set(processed) - expected
    if extra:
        raise RuntimeError(f"{split}: unexpected preprocessed UIDs: {sorted(extra)}")
    return processed, sorted(expected - set(processed))


def token_position(original: int, kept_start: int, kept_end: int) -> int | None:
    return original - kept_start if kept_start <= original < kept_end else None


def classify_labels(
    trainer, row: dict, input_ids: list[int], labels: list[int], template: str
) -> tuple[dict[str, int], list[str], bool, int, int]:
    tokenizer = trainer.processing_class
    full_ids, boundaries = message_boundaries(tokenizer, row["messages"], template)
    kept_start, kept_end = retained_slice(trainer, len(full_ids))
    if input_ids != full_ids[kept_start:kept_end]:
        raise RuntimeError(
            f"{row['split']}/{row['uid']}: preprocessed input_ids differ from the "
            "effective template plus formal truncation"
        )
    if len(labels) != len(input_ids):
        raise RuntimeError(f"{row['split']}/{row['uid']}: labels/input_ids length mismatch")
    if any(label != -100 and label != token for token, label in zip(input_ids, labels)):
        raise RuntimeError(f"{row['split']}/{row['uid']}: labels differ from input_ids")

    think_open = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    counts = Counter()
    diagnostics = []
    answer_complete = False
    end_total = 0
    end_supervised = 0

    for message_index, (message, (start, end)) in enumerate(zip(row["messages"], boundaries)):
        retained = [
            (original, token_position(original, kept_start, kept_end))
            for original in range(start, end)
        ]
        retained = [pair for pair in retained if pair[1] is not None]
        supervised = [
            (original, current) for original, current in retained
            if labels[current] != -100
        ]
        if message["role"] != "assistant":
            category = (
                "information"
                if message["role"] == "user"
                and message["content"].startswith("<information>")
                else message["role"]
            )
            counts[category] += len(supervised)
            continue

        segment = full_ids[start:end]
        opens = [start + i for i, token in enumerate(segment) if token == think_open]
        closes = [start + i for i, token in enumerate(segment) if token == think_close]
        ends = [start + i for i, token in enumerate(segment) if token == im_end]
        if len(opens) != 1 or len(closes) != 1:
            diagnostics.append(
                f"message={message_index}: expected one <think>/</think>, "
                f"found {len(opens)}/{len(closes)}; content={message.get('content', '')[:120]!r}"
            )
        if len(ends) != 1:
            diagnostics.append(
                f"message={message_index}: expected one <|im_end|>, found {len(ends)}"
            )
        valid_think = bool(opens and closes and opens[0] < closes[0])
        think_start, think_end = (opens[0], closes[0]) if valid_think else (None, None)
        action_end = ends[0] if ends else end
        action_type = "search" if message["content"].startswith("<search>") else "answer"

        for original, _ in supervised:
            if valid_think and think_start <= original <= think_end:
                counts["think"] += 1
            elif original < action_end and (not valid_think or original > think_end):
                counts[action_type] += 1
            elif original >= action_end:
                counts["assistant_end"] += 1
            else:
                counts["assistant_unclassified"] += 1

        if ends:
            end_total += 1
            current = token_position(ends[0], kept_start, kept_end)
            if current is not None and labels[current] != -100:
                end_supervised += 1
            elif current is not None:
                diagnostics.append(f"message={message_index}: retained <|im_end|> is masked")
        if action_type == "answer":
            answer_complete = start >= kept_start and end <= kept_end

    counts["total_supervised"] = sum(label != -100 for label in labels)
    return dict(counts), diagnostics, answer_complete, end_supervised, end_total


def dataloader_batch(trainer, dataset, processed_index: int) -> dict:
    batch = next(iter(trainer.get_eval_dataloader(dataset.select([processed_index]))))
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(batch):
        raise RuntimeError(f"Real DataLoader batch lacks: {required - set(batch)}")
    attention = batch["attention_mask"][0].detach().cpu().tolist()
    ids = batch["input_ids"][0].detach().cpu().tolist()
    labels = batch["labels"][0].detach().cpu().tolist()
    active = [index for index, bit in enumerate(attention) if bit]
    return {
        "input_ids": [ids[index] for index in active],
        "attention_mask": [attention[index] for index in active],
        "labels": [labels[index] for index in active],
    }


def audit_processed_split(
    trainer, dataset, rows: list[dict], template: str, split: str
) -> tuple[list[dict], list[str], list[dict]]:
    processed, filtered = locate_processed(dataset, rows, split)
    row_by_uid = {row["uid"]: row for row in rows}
    results = []
    batches = []
    for uid in sorted(processed):
        stored = processed[uid]
        example = stored["example"]
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        batch = dataloader_batch(trainer, dataset, stored["index"])
        if batch["input_ids"] != input_ids or batch["labels"] != labels:
            raise RuntimeError(
                f"{split}/{uid}: DataLoader differs from preprocessed input_ids/labels"
            )
        row = row_by_uid[uid]
        segments, diagnostics, answer_complete, end_supervised, end_total = classify_labels(
            trainer, row, input_ids, labels, template
        )
        result = {
            "uid": uid, "split": split, "source": row["record"]["source"],
            "search_count": row["record"]["search_count"],
            "line_index": row["line_index"],
            "trajectory_characters": trajectory_characters(row),
            "tokens_after_truncation": len(input_ids),
            "supervised_tokens": segments["total_supervised"],
            "masked_ratio": labels.count(-100) / len(labels),
            "segments": segments, "answer_complete": answer_complete,
            "assistant_end_supervised": end_supervised,
            "assistant_end_total": end_total, "diagnostics": diagnostics,
        }
        results.append(result)
        batches.append(
            {
                **result, "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"], "labels": batch["labels"],
                "decoded_supervision": trainer.processing_class.decode(
                    [token for token, label in zip(batch["input_ids"], batch["labels"])
                     if label != -100],
                    skip_special_tokens=False,
                ),
            }
        )
    return results, filtered, batches


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_reports(
    args, trainer, template, train_rows, val_rows, train_results, val_results,
    filtered_train, filtered_val, all_batches,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: package_version(name) for name in
                ("torch", "transformers", "trl", "peft", "datasets", "accelerate")}
    environment = [
        "# Environment", "", f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`", f"- CUDA: `{torch.cuda.is_available()}`",
        f"- GPU: `{torch.cuda.get_device_name(0)}`",
        f"- Model/tokenizer: `{args.model_path}`",
        f"- max_length: `{trainer.args.max_length}`",
        f"- truncation_mode: `{trainer.args.truncation_mode}`",
        "- Trainer: `train_sft.build_trainer()`; `trainer.train()` was not called.",
        "", "## Packages", "",
    ]
    environment.extend(f"- {name}: `{version}`" for name, version in versions.items())
    (args.output_dir / "environment.md").write_text("\n".join(environment) + "\n", encoding="utf-8")

    selected = train_rows + val_rows
    chat = [
        "# Effective chat template", "",
        "Uses `trainer.chat_template` when TRL patches the tokenizer template; otherwise "
        "uses `trainer.processing_class.chat_template`.", "",
        f"- SHA256: `{sha256_text(template)}`",
        f"- Generation markers: `{('{% generation' in template) or ('{%- generation' in template)}`",
        f"- Selected trajectories: `{len(selected)}`", "",
        "| ID | Split | Source | Searches | Characters |", "|---|---|---|---:|---:|",
    ]
    for row in selected:
        chat.append(
            f"| {row['uid']} | {row['split']} | {row['record']['source']} | "
            f"{row['record']['search_count']} | {trajectory_characters(row)} |"
        )
    chat.extend(["", "## Template", "", "```jinja", template, "```", ""])
    (args.output_dir / "chat_template_audit.md").write_text("\n".join(chat), encoding="utf-8")

    results = train_results + val_results
    loss = [
        "# Real Trainer labels and DataLoader audit", "",
        "Every retained row uses TRL-preprocessed labels and was checked against a real DataLoader batch.", "",
        "| ID | Split | Source | Searches | Tokens | Supervised | Masked | Think | Search | Answer | Information | EOS supervised | Answer complete | Diagnostics |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        segments = result["segments"]
        loss.append(
            f"| {result['uid']} | {result['split']} | {result['source']} | {result['search_count']} | "
            f"{result['tokens_after_truncation']} | {result['supervised_tokens']} | "
            f"{result['masked_ratio']:.2%} | {segments.get('think', 0)} | "
            f"{segments.get('search', 0)} | {segments.get('answer', 0)} | "
            f"{segments.get('information', 0)} | "
            f"{result['assistant_end_supervised']}/{result['assistant_end_total']} | "
            f"{result['answer_complete']} | {'; '.join(result['diagnostics']) or 'none'} |"
        )
    loss.extend([
        "", f"- Filtered train: `{filtered_train}`; validation: `{filtered_val}`",
        f"- Zero-supervision retained: `{sum(r['supervised_tokens'] == 0 for r in results)}`",
        f"- Incomplete answers: `{sum(not r['answer_complete'] for r in results)}`",
        f"- Supervised Information tokens: `{sum(r['segments'].get('information', 0) for r in results)}`", "",
    ])
    (args.output_dir / "loss_mask_audit.md").write_text("\n".join(loss), encoding="utf-8")

    longest = sorted(results, key=lambda item: item["trajectory_characters"], reverse=True)[:10]
    truncation = [
        "# Truncation audit", "",
        "Uses post-truncation input_ids and labels retained by the formal Trainer.", "",
        f"- Filtered train IDs: `{filtered_train}`",
        f"- Filtered validation IDs: `{filtered_val}`",
        f"- Incomplete final answers: `{[r['uid'] for r in results if not r['answer_complete']]}`",
        f"- Zero-supervision retained IDs: `{[r['uid'] for r in results if r['supervised_tokens'] == 0]}`",
        "", "## Longest selected trajectories", "",
    ]
    for result in longest:
        truncation.append(
            f"- `{result['uid']}`: chars={result['trajectory_characters']}, "
            f"tokens={result['tokens_after_truncation']}, answer_complete={result['answer_complete']}"
        )
    (args.output_dir / "truncation_audit.md").write_text("\n".join(truncation) + "\n", encoding="utf-8")

    example_ids = {
        batch["uid"] for batch in sorted(
            all_batches,
            key=lambda batch: (batch["source"] != "aethersearch", -batch["search_count"],
                               -batch["trajectory_characters"]),
        )[: args.batch_examples]
    }
    saved_batches = [batch for batch in all_batches if batch["uid"] in example_ids]
    (args.output_dir / "real_batch_examples.json").write_text(
        json.dumps(saved_batches, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    aether = [result for result in results if result["source"] == "aethersearch"]
    summary = [
        "# SFT supervision mechanism audit summary", "", "## Evidence levels", "",
        "1. Static: reuses `train_sft.build_trainer()`.",
        "2. Trainer preprocessing: uses the input_ids and labels retained by TRL.",
        "3. Real DataLoader: every retained selected row is compared with an actual batch.",
        "4. Tokenizer rendering only locates boundaries; it never constructs labels.", "",
        "## Results", "",
        f"- Requested/selected: `{len(selected)}`; retained: `{len(results)}`.",
        f"- Trainer-filtered IDs: `{filtered_train + filtered_val}`.",
        f"- Think supervised in every retained AetherSearch sample: "
        f"`{bool(aether) and all(r['segments'].get('think', 0) > 0 for r in aether)}`.",
        f"- Search tokens: `{sum(r['segments'].get('search', 0) for r in results)}`.",
        f"- Answer tokens: `{sum(r['segments'].get('answer', 0) for r in results)}`.",
        f"- Information tokens: `{sum(r['segments'].get('information', 0) for r in results)}`.",
        f"- Assistant end markers: `{sum(r['assistant_end_supervised'] for r in results)}/"
        f"{sum(r['assistant_end_total'] for r in results)}` supervised.",
        f"- Incomplete answers: `{[r['uid'] for r in results if not r['answer_complete']]}`.",
        f"- Boundary diagnostics: `{[{'uid': r['uid'], 'issues': r['diagnostics']} for r in results if r['diagnostics']]}`.",
        "", "This representative audit verifies supervision mechanics, not full-dataset semantic quality.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the real SFTTrainer audit. Run in the remote "
            "searchrl-sft GPU environment; no simulated result was produced."
        )
    for path in (args.train_messages, args.train_canonical, args.val_messages, args.val_canonical):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_all = load_aligned(args.train_messages, args.train_canonical, "train")
    val_all = load_aligned(args.val_messages, args.val_canonical, "validation")
    train_rows, val_rows = select_audit_rows(train_all, val_all, args.max_audit_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(args, train_rows, val_rows)
    template = effective_chat_template(trainer)
    train_results, filtered_train, train_batches = audit_processed_split(
        trainer, trainer.train_dataset, train_rows, template, "train"
    )
    val_results, filtered_val, val_batches = audit_processed_split(
        trainer, trainer.eval_dataset, val_rows, template, "validation"
    )
    write_reports(
        args, trainer, template, train_rows, val_rows, train_results, val_results,
        filtered_train, filtered_val, train_batches + val_batches,
    )
    print(f"Audited {len(train_results) + len(val_results)} real Trainer/DataLoader rows")
    print(f"Reports written to: {args.output_dir}")
    print("trainer.train() was not called; no optimizer update was performed.")


if __name__ == "__main__":
    main()
