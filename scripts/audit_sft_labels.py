from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the real TRL SFTTrainer labels and loss mask without training."
    )
    parser.add_argument("--train-messages", type=Path, required=True)
    parser.add_argument("--train-canonical", type=Path, required=True)
    parser.add_argument("--val-messages", type=Path, required=True)
    parser.add_argument("--val-canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=train_sft.MODEL_NAME)
    parser.add_argument(
        "--max-audit-samples",
        type=int,
        default=0,
        help="Maximum records per split; 0 audits every record.",
    )
    parser.add_argument(
        "--batch-examples",
        type=int,
        default=8,
        help="Maximum representative records inspected through a real DataLoader.",
    )
    args = parser.parse_args()
    if args.max_audit_samples < 0:
        parser.error("--max-audit-samples must be non-negative")
    if args.batch_examples <= 0:
        parser.error("--batch-examples must be positive")
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_aligned(messages_path: Path, canonical_path: Path) -> list[dict]:
    messages = load_jsonl(messages_path)
    canonical = load_jsonl(canonical_path)
    if len(messages) != len(canonical):
        raise ValueError(
            f"Line-count mismatch: {messages_path}={len(messages)}, "
            f"{canonical_path}={len(canonical)}"
        )
    rows = []
    for index, (message_record, record) in enumerate(zip(messages, canonical)):
        if set(message_record) != {"messages"}:
            raise ValueError(f"Unexpected Messages fields at line {index}: {message_record.keys()}")
        rows.append({"record": record, "messages": message_record["messages"]})
    return rows


def select_rows(rows: list[dict], limit: int) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    selected = []
    seen = set()
    for row in rows:
        key = (row["record"]["source"], row["record"]["search_count"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
            if len(selected) == limit:
                return selected
    selected_ids = {row["record"]["id"] for row in selected}
    selected.extend(
        row for row in rows if row["record"]["id"] not in selected_ids
    )
    return selected[:limit]


def representative_indices(rows: list[dict], limit: int) -> list[int]:
    indices = []
    seen = set()
    for index, row in enumerate(rows):
        key = (row["record"]["source"], row["record"]["search_count"])
        if key not in seen:
            indices.append(index)
            seen.add(key)
            if len(indices) == limit:
                return indices
    for index in range(len(rows)):
        if index not in indices:
            indices.append(index)
            if len(indices) == limit:
                break
    return indices


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def token_ids(encoded) -> list[int]:
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def assistant_mask(encoded) -> list[int]:
    mask = encoded.get("assistant_masks")
    if mask is None:
        mask = encoded.get("assistant_mask")
    if mask is None:
        raise RuntimeError("Tokenizer did not return an assistant token mask")
    if mask and isinstance(mask[0], list):
        mask = mask[0]
    return list(mask)


def render(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    ids = token_ids(encoded)
    mask = assistant_mask(encoded)
    if len(ids) != len(mask):
        raise RuntimeError("Tokenizer input_ids and assistant mask lengths differ")
    return ids, mask


def message_boundaries(tokenizer, messages: list[dict]) -> list[tuple[int, int]]:
    boundaries = []
    previous = 0
    full_ids = render(tokenizer, messages)[0]
    for index in range(len(messages)):
        prefix_ids = render(tokenizer, messages[: index + 1])[0]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Chat template does not preserve the prefix ending at message {index}"
            )
        current = len(prefix_ids)
        boundaries.append((previous, current))
        previous = current
    return boundaries


def classify_supervision(
    tokenizer,
    messages: list[dict],
    labels: list[int],
) -> dict[str, int]:
    counts = Counter()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    ids, _ = render(tokenizer, messages)
    length = min(len(ids), len(labels))
    boundaries = message_boundaries(tokenizer, messages)

    for message, (start, end) in zip(messages, boundaries):
        start = min(start, length)
        end = min(end, length)
        supervised = [index for index in range(start, end) if labels[index] != -100]
        if message["role"] != "assistant":
            category = "information" if message["content"].startswith("<information>") else message["role"]
            counts[category] += len(supervised)
            continue
        if not supervised:
            continue
        segment = ids[start:end]
        think_start = start + segment.index(think_open_id)
        try:
            think_end = start + segment.index(think_close_id)
        except ValueError:
            think_end = end
        im_end_positions = [i for i, value in enumerate(segment) if value == im_end_id]
        end_token = start + max(im_end_positions) if im_end_positions else end
        action_type = "search" if message["content"].startswith("<search>") else "answer"
        for index in supervised:
            if think_start <= index <= think_end:
                counts["think"] += 1
            elif index < end_token:
                counts[action_type] += 1
            else:
                counts["assistant_end"] += 1
    return dict(counts)


def repeated_think_supervision(
    tokenizer,
    messages: list[dict],
    labels: list[int],
    repeated_thinks: Counter,
) -> int:
    ids, _ = render(tokenizer, messages)
    length = min(len(ids), len(labels))
    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    total = 0
    for message, (start, end) in zip(messages, message_boundaries(tokenizer, messages)):
        if message["role"] != "assistant" or repeated_thinks[message["reasoning_content"]] <= 1:
            continue
        start = min(start, length)
        end = min(end, length)
        segment = ids[start:end]
        if think_open_id not in segment:
            continue
        think_start = start + segment.index(think_open_id)
        think_end = (
            start + segment.index(think_close_id)
            if think_close_id in segment
            else end
        )
        total += sum(
            labels[index] != -100
            for index in range(think_start, min(think_end + 1, length))
        )
    return total


def answer_complete(tokenizer, messages: list[dict], maximum: int) -> bool:
    boundaries = message_boundaries(tokenizer, messages)
    final_index = max(
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant" and message["content"].startswith("<answer>")
    )
    return boundaries[final_index][1] <= maximum


def audit_split(tokenizer, rows: list[dict], maximum: int) -> tuple[dict, list[dict]]:
    details = []
    source_segments = defaultdict(Counter)
    repeated_thinks = Counter(
        event["content"]
        for row in rows
        if row["record"]["source"] == "aethersearch"
        for event in row["record"]["events"]
        if event["type"] == "think"
    )
    repeated_aether_think_tokens = 0

    for row in rows:
        ids, mask = render(tokenizer, row["messages"])
        truncated_ids = ids[:maximum]
        truncated_mask = mask[:maximum]
        labels = [token if supervised else -100 for token, supervised in zip(ids, mask)]
        truncated_labels = labels[:maximum]
        segments = classify_supervision(tokenizer, row["messages"], labels)
        truncated_segments = classify_supervision(
            tokenizer, row["messages"], truncated_labels
        )
        source_segments[row["record"]["source"]].update(truncated_segments)
        if row["record"]["source"] == "aethersearch":
            repeated_aether_think_tokens += repeated_think_supervision(
                tokenizer,
                row["messages"],
                truncated_labels,
                repeated_thinks,
            )
        details.append(
            {
                "id": row["record"]["id"],
                "source": row["record"]["source"],
                "search_count": row["record"]["search_count"],
                "tokens": len(ids),
                "supervised": sum(mask),
                "truncated_tokens": len(truncated_ids),
                "truncated_supervised": sum(truncated_mask),
                "answer_complete": answer_complete(tokenizer, row["messages"], maximum),
                "zero_supervision": not any(truncated_mask),
                "segments": segments,
                "truncated_segments": truncated_segments,
            }
        )

    lengths = [item["tokens"] for item in details]
    summary = {
        "count": len(details),
        "min": min(lengths),
        "mean": statistics.fmean(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "over_maximum": sum(item["tokens"] > maximum for item in details),
        "incomplete_answer": sum(not item["answer_complete"] for item in details),
        "zero_supervision": sum(item["zero_supervision"] for item in details),
        "supervised_before": sum(item["supervised"] for item in details),
        "supervised_after": sum(item["truncated_supervised"] for item in details),
        "source_segments": {key: dict(value) for key, value in source_segments.items()},
        "repeated_aether_think_tokens": repeated_aether_think_tokens,
    }
    return summary, details


def make_trainer(args: argparse.Namespace, train_rows: list[dict], val_rows: list[dict]):
    train_sft.MODEL_NAME = args.model_path
    trainer_args = SimpleNamespace(output_dir=args.output_dir / "runtime", max_steps=1)
    train_dataset = Dataset.from_list([{"messages": row["messages"]} for row in train_rows])
    val_dataset = Dataset.from_list([{"messages": row["messages"]} for row in val_rows])
    return train_sft.build_trainer(trainer_args, train_dataset, val_dataset)


def audit_real_batches(trainer, rows: list[dict], limit: int) -> list[dict]:
    results = []
    for index in representative_indices(rows, min(limit, len(rows))):
        one_row = trainer.train_dataset.select([index])
        batch = next(iter(trainer.get_eval_dataloader(one_row)))
        ids = batch["input_ids"][0].detach().cpu().tolist()
        attention = batch["attention_mask"][0].detach().cpu().tolist()
        labels = batch["labels"][0].detach().cpu().tolist()
        supervised_ids = [token for token, label in zip(ids, labels) if label != -100]
        if any(label != -100 and label != token for token, label in zip(ids, labels)):
            raise RuntimeError("A supervised label differs from its input token")
        results.append(
            {
                "id": rows[index]["record"]["id"],
                "source": rows[index]["record"]["source"],
                "search_count": rows[index]["record"]["search_count"],
                "input_ids": ids,
                "attention_mask": attention,
                "labels": labels,
                "effective_tokens": len(supervised_ids),
                "masked_ratio": labels.count(-100) / len(labels),
                "decoded_supervision": trainer.processing_class.decode(
                    supervised_ids, skip_special_tokens=False
                ),
                "segments": classify_supervision(
                    trainer.processing_class, rows[index]["messages"], labels
                ),
            }
        )
    return results


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_reports(
    args: argparse.Namespace,
    trainer,
    train_summary: dict,
    train_details: list[dict],
    val_summary: dict,
    val_details: list[dict],
    batches: list[dict],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = trainer.processing_class
    template = tokenizer.chat_template
    versions = {
        name: package_version(name)
        for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    }
    environment = [
        "# SFT label audit environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        f"- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}`",
        f"- Model/tokenizer source: `{args.model_path}`",
        f"- Formal model constant: `{train_sft.MODEL_NAME}`",
        f"- max_length: `{train_sft.MAX_LENGTH}`",
        "- Trainer construction: `train_sft.build_trainer()`; `trainer.train()` was not called.",
        "",
        "## Packages",
        "",
    ]
    environment.extend(f"- {name}: `{version}`" for name, version in versions.items())
    (args.output_dir / "environment.md").write_text("\n".join(environment) + "\n", encoding="utf-8")

    chat_lines = [
        "# Chat template audit",
        "",
        "This is a Tokenizer runtime result using `trainer.processing_class`.",
        f"- Template SHA256: `{sha256_text(template)}`",
        "- Qwen3 assistant generation regions are taken from the active template, not hand-built masks.",
        "- Representative samples:",
    ]
    for batch in batches:
        chat_lines.append(
            f"  - `{batch['id']}` ({batch['source']}, search_count={batch['search_count']}): "
            f"{batch['segments']}"
        )
    chat_lines.extend(["", "## Active template", "", "```jinja", template, "```", ""])
    (args.output_dir / "chat_template_audit.md").write_text(
        "\n".join(chat_lines), encoding="utf-8"
    )

    loss_lines = [
        "# Loss mask audit",
        "",
        "These results come from real batches returned by `SFTTrainer.get_eval_dataloader()`.",
        "Full arrays are in `real_batches.json`.",
        "",
        "| ID | Source | Searches | Tokens | Effective | -100 ratio | Segments |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for batch in batches:
        loss_lines.append(
            f"| {batch['id']} | {batch['source']} | {batch['search_count']} | "
            f"{len(batch['labels'])} | {batch['effective_tokens']} | "
            f"{batch['masked_ratio']:.4f} | `{batch['segments']}` |"
        )
    loss_lines.extend(["", "## Decoded effective supervision", ""])
    for batch in batches:
        loss_lines.extend(
            [f"### {batch['id']}", "", "```text", batch["decoded_supervision"], "```", ""]
        )
    (args.output_dir / "loss_mask_audit.md").write_text(
        "\n".join(loss_lines), encoding="utf-8"
    )
    (args.output_dir / "real_batches.json").write_text(
        json.dumps(batches, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    truncation_lines = [
        "# Truncation audit",
        "",
        f"Formal max_length: `{train_sft.MAX_LENGTH}`. Statistics use the active Trainer tokenizer/template.",
        "",
        "| Split | Count | Min | Mean | P50 | P90 | P95 | P99 | Max | >4096 | Incomplete answer | Zero supervision | Supervised before | Supervised after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("train", train_summary), ("validation", val_summary)):
        truncation_lines.append(
            f"| {name} | {summary['count']} | {summary['min']} | {summary['mean']:.2f} | "
            f"{summary['p50']} | {summary['p90']} | {summary['p95']} | {summary['p99']} | "
            f"{summary['max']} | {summary['over_maximum']} | {summary['incomplete_answer']} | "
            f"{summary['zero_supervision']} | {summary['supervised_before']} | "
            f"{summary['supervised_after']} |"
        )
    truncation_lines.extend(["", "## Over-length records", ""])
    for split, details in (("train", train_details), ("validation", val_details)):
        over = [item for item in details if item["tokens"] > train_sft.MAX_LENGTH]
        truncation_lines.append(f"- {split}: `{len(over)}`")
        for item in over[:20]:
            truncation_lines.append(
                f"  - `{item['id']}`: {item['tokens']} tokens; "
                f"answer_complete={item['answer_complete']}"
            )
    (args.output_dir / "truncation_audit.md").write_text(
        "\n".join(truncation_lines) + "\n", encoding="utf-8"
    )

    information_supervised = sum(
        batch["segments"].get("information", 0) for batch in batches
    )
    aether_segments = train_summary["source_segments"].get("aethersearch", {})
    aether_supervised = sum(aether_segments.values())
    repeated_think_ratio = (
        train_summary["repeated_aether_think_tokens"] / aether_supervised
        if aether_supervised
        else 0.0
    )
    summary_lines = [
        "# SFT labels and loss-mask audit summary",
        "",
        "## Verification levels",
        "",
        "- Static: `train_sft.py` was imported and its `build_trainer()` was used.",
        "- Tokenizer runtime: the active Trainer tokenizer and chat template produced all length/segment statistics.",
        "- Trainer/DataLoader runtime: representative `input_ids`, `attention_mask`, and `labels` came from a real TRL DataLoader.",
        "- No optimizer step or `trainer.train()` call was performed.",
        "",
        "## Direct answers",
        "",
        f"1. Actual supervised segments in representative batches: `{[batch['segments'] for batch in batches]}`.",
        f"2. Think participates in loss: `{all(batch['segments'].get('think', 0) > 0 for batch in batches)}`.",
        f"3. Search/Answer supervision follows the assistant turns shown above.",
        f"4. Information tokens supervised in representative batches: `{information_supervised}`.",
        f"5. Incomplete answers after truncation: train=`{train_summary['incomplete_answer']}`, validation=`{val_summary['incomplete_answer']}`.",
        f"6. Zero-supervision records: train=`{train_summary['zero_supervision']}`, validation=`{val_summary['zero_supervision']}`.",
        "",
        "## Source-level effective-token composition",
        "",
        f"- Train: `{train_summary['source_segments']}`",
        f"- Validation: `{val_summary['source_segments']}`",
        f"- Repeated AetherSearch Think tokens within effective train supervision: `{train_summary['repeated_aether_think_tokens']}`.",
        f"- Repeated AetherSearch Think share of all effective AetherSearch tokens: `{repeated_think_ratio:.4%}`.",
        "",
        "The audit establishes data/label readiness only; it does not validate semantic trajectory quality or training outcome.",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the real SFTTrainer audit; run this script in the "
            "remote searchrl-sft GPU environment. No simulated CPU result was produced."
        )
    for path in (
        args.train_messages,
        args.train_canonical,
        args.val_messages,
        args.val_canonical,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = select_rows(
        load_aligned(args.train_messages, args.train_canonical),
        args.max_audit_samples,
    )
    val_rows = select_rows(
        load_aligned(args.val_messages, args.val_canonical),
        args.max_audit_samples,
    )
    trainer = make_trainer(args, train_rows, val_rows)

    train_summary, train_details = audit_split(
        trainer.processing_class, train_rows, train_sft.MAX_LENGTH
    )
    val_summary, val_details = audit_split(
        trainer.processing_class, val_rows, train_sft.MAX_LENGTH
    )
    batches = audit_real_batches(trainer, train_rows, args.batch_examples)
    write_reports(
        args,
        trainer,
        train_summary,
        train_details,
        val_summary,
        val_details,
        batches,
    )
    print(f"Reports written to: {args.output_dir}")
    print("trainer.train() was not called; no optimizer update was performed.")


if __name__ == "__main__":
    main()
