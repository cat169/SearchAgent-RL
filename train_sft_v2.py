from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_PATH = Path("/root/models/Qwen3-4B-Base")
MAX_LENGTH = 4096
SEED = 123
SPECIAL_TOKEN_IDS = {
    "<think>": 151667,
    "</think>": 151668,
    "<|im_end|>": 151645,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Qwen3-4B-Base with BF16 LoRA and selectively train the "
            "Agent boundary tokens."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Initialize the tokenizer, model, and adapter and run checks only.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.check_only and args.resume_from_checkpoint is not None:
        parser.error("--check-only cannot be combined with --resume-from-checkpoint")

    return args


def load_datasets(args: argparse.Namespace):
    datasets = load_dataset(
        "json",
        data_files={
            "train": str(args.train_path),
            "validation": str(args.val_path),
        },
    )
    train_dataset = datasets["train"]
    if args.limit is not None:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(args.limit, len(train_dataset)))
        )
    return train_dataset, datasets["validation"]


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


def load_and_check_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        encoded_ids = tokenizer.encode(token, add_special_tokens=False)
        if actual_id != expected_id or encoded_ids != [expected_id]:
            raise RuntimeError(
                f"Unexpected tokenizer mapping for {token!r}: "
                f"convert_tokens_to_ids={actual_id}, encode={encoded_ids}, "
                f"expected={[expected_id]}"
            )
    return tokenizer


def build_trainer(args: argparse.Namespace, train_dataset, val_dataset, tokenizer):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        trainable_token_indices={
            "embed_tokens": list(SPECIAL_TOKEN_IDS.values()),
        },
        ensure_weight_tying=True,
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
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs=2,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        optim="adamw_torch",
        seed=SEED,
        data_seed=SEED,
        report_to="tensorboard",
        logging_dir=str(args.output_dir / "runs"),
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        push_to_hub=False,
    )
    return SFTTrainer(
        model=str(args.model_path),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )


def check_adapter_initialization(model) -> None:
    if not model.config.tie_word_embeddings:
        raise RuntimeError("Base model does not have tied input/output embeddings")

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_adapter = getattr(input_embeddings, "token_adapter", None)
    output_adapter = getattr(output_embeddings, "token_adapter", None)
    if input_adapter is None or output_adapter is None:
        raise RuntimeError(
            "PEFT did not wrap both input embeddings and the tied output layer "
            "with trainable-token adapters"
        )

    adapter_name = model.active_adapter
    if isinstance(adapter_name, (list, tuple)):
        if len(adapter_name) != 1:
            raise RuntimeError(f"Expected one active adapter, got {adapter_name}")
        adapter_name = adapter_name[0]

    if adapter_name not in input_adapter.trainable_tokens_delta:
        raise RuntimeError(
            f"Missing trainable_tokens_delta for active adapter {adapter_name!r}"
        )
    token_delta = input_adapter.trainable_tokens_delta[adapter_name]
    if not token_delta.requires_grad:
        raise RuntimeError("trainable_tokens_delta does not require gradients")
    if token_delta.shape[0] != len(SPECIAL_TOKEN_IDS):
        raise RuntimeError(
            "Unexpected trainable token row count: "
            f"{token_delta.shape[0]} instead of {len(SPECIAL_TOKEN_IDS)}"
        )

    configured_ids = input_adapter.token_indices[adapter_name]
    if list(configured_ids) != list(SPECIAL_TOKEN_IDS.values()):
        raise RuntimeError(
            f"PEFT trainable token IDs are {configured_ids}, "
            f"expected {list(SPECIAL_TOKEN_IDS.values())}"
        )

    if getattr(output_adapter, "tied_adapter", None) is not input_adapter:
        raise RuntimeError(
            "The output trainable-token adapter is not tied to the input adapter"
        )

    input_base_weight = input_adapter.base_layer.weight
    output_base_weight = output_adapter.base_layer.weight
    if input_base_weight.data_ptr() != output_base_weight.data_ptr():
        raise RuntimeError("Base input embedding and output head weights are not shared")
    if input_base_weight.requires_grad or output_base_weight.requires_grad:
        raise RuntimeError("The full base embedding/output weight is unexpectedly trainable")

    token_ids = torch.tensor(
        list(SPECIAL_TOKEN_IDS.values()),
        device=input_base_weight.device,
    )
    with torch.inference_mode():
        effective_input_rows = input_embeddings(token_ids).float()
        effective_output_rows = output_embeddings.weight[token_ids].float()
    if not torch.equal(effective_input_rows, effective_output_rows):
        raise RuntimeError(
            "Effective input and output rows differ for the selected special tokens"
        )

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    lora_parameters = [
        (name, parameter)
        for name, parameter in trainable_parameters
        if "lora_" in name
    ]
    token_parameters = [
        (name, parameter)
        for name, parameter in trainable_parameters
        if "trainable_tokens_delta" in name
    ]
    unexpected_parameters = [
        name
        for name, _ in trainable_parameters
        if "lora_" not in name and "trainable_tokens_delta" not in name
    ]
    if not lora_parameters:
        raise RuntimeError("No trainable LoRA parameters were created")
    if not token_parameters:
        raise RuntimeError("No trainable special-token parameters were created")
    if unexpected_parameters:
        raise RuntimeError(
            "Unexpected trainable base parameters: " + ", ".join(unexpected_parameters)
        )

    print("SFT v2 initialization checks passed")
    print(f"  special token IDs       : {SPECIAL_TOKEN_IDS}")
    print(f"  trainable token delta   : {tuple(token_delta.shape)}")
    print(f"  trainable LoRA tensors  : {len(lora_parameters)}")
    print(f"  trainable token tensors : {len(token_parameters)}")
    print("  input/output tying      : structural and effective-row checks passed")
    print("  other base parameters   : frozen")


def main() -> None:
    args = parse_args()
    check_output_path(args.output_dir, args.resume_from_checkpoint)
    tokenizer = load_and_check_tokenizer(args.model_path)
    train_dataset, val_dataset = load_datasets(args)

    print("SFT v2 training configuration")
    print(f"  model path              : {args.model_path}")
    print(f"  train / val paths       : {args.train_path} / {args.val_path}")
    print(f"  train / val samples     : {len(train_dataset)} / {len(val_dataset)}")
    print(f"  max_length              : {MAX_LENGTH}")
    print("  batch / accumulation    : 1 / 8")
    print("  LoRA                    : r=16, alpha=32, dropout=0.05")
    print("  LoRA target_modules     : all-linear")
    print(f"  trainable token IDs     : {list(SPECIAL_TOKEN_IDS.values())}")
    print("  learning rate           : 1e-4")
    print(
        "  epochs / max steps      : "
        f"2 / {args.max_steps if args.max_steps is not None else -1}"
    )
    print(f"  check only              : {args.check_only}")
    print(f"  output                  : {args.output_dir}")

    trainer = build_trainer(args, train_dataset, val_dataset, tokenizer)
    check_adapter_initialization(trainer.model)
    if args.check_only:
        print("Check-only mode completed; trainer.train() was not called.")
        return

    checkpoint = (
        str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_model(str(args.output_dir))
    trainer.processing_class.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
