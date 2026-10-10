from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_LENGTH = 4096
SEED = 123


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-4B-Base with BF16 LoRA on Messages JSONL."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")

    return args


def load_train_dataset(data_path: Path, limit: int | None):
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def check_output_path(
    output_dir: Path,
    resume_from_checkpoint: Path | None,
) -> None:
    if resume_from_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new directory or --resume-from-checkpoint."
            )
        return

    if not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {resume_from_checkpoint}"
        )
    if resume_from_checkpoint.resolve().parent != output_dir.resolve():
        raise ValueError(
            "The checkpoint must belong to the specified output directory"
        )


def main() -> None:
    args = parse_args()
    check_output_path(args.output_dir, args.resume_from_checkpoint)

    train_dataset = load_train_dataset(args.data_path, args.limit)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        max_length=MAX_LENGTH,
        assistant_only_loss=True,
        packing=False,
        eos_token="<|im_end|>",
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs=2,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        optim="adamw_torch",
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainable, total = trainer.model.get_nb_trainable_parameters()
    print("SFT training configuration")
    print(f"  model                 : {MODEL_NAME}")
    print(f"  data                   : {args.data_path}")
    print(f"  samples                : {len(train_dataset)}")
    print(f"  max_length             : {MAX_LENGTH}")
    print("  batch / accumulation   : 1 / 8")
    print("  LoRA                   : r=16, alpha=32, dropout=0.05")
    print("  LoRA target_modules    : all-linear")
    print(f"  trainable parameters   : {trainable:,}")
    print(f"  total parameters       : {total:,}")
    print(f"  trainable ratio        : {100 * trainable / total:.4f}%")
    print(f"  output                 : {args.output_dir}")

    checkpoint = (
        str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(args.output_dir))
    trainer.processing_class.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
