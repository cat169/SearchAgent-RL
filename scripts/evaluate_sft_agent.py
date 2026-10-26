"""Run Qwen3 LoRA through the real AgentLoop and Local BM25 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

from agent.environment import SearchEnvironment
from agent.loop import AgentLoop
from agent.search_tool import SearchTool
from agent.system_prompt import AGENT_SYSTEM_PROMPT
from retrieval.bm25_retriever import BM25Retriever


DEFAULT_EVAL_PATH = Path(
    "/root/SearchAgent-RL/experiments/nq_hotpotqa_sft_v1/data/eval.jsonl"
)
DEFAULT_BASE_MODEL = Path("/root/models/Qwen3-4B-Base")
DEFAULT_ADAPTER_PATH = Path(
    "/root/SearchAgent-RL/experiments/nq_hotpotqa_sft_v1/runs/sft_full"
)
DEFAULT_INDEX_PATH = Path(
    "/root/blockdata/SearchAgent-RL/retrieval/wiki-18-bm25-index/bm25"
)
DEFAULT_CORPUS_PATH = Path(
    "/root/blockdata/SearchAgent-RL/retrieval/wiki-18-corpus/wiki-18.jsonl"
)
MAX_TURNS = 5
TOP_K = 3
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_INPUT_TOKENS = 4096

STRICT_ACTION_PATTERN = re.compile(
    r"\s*<(search|answer)>(.*?)</\1>\s*",
    re.DOTALL,
)


class InputTooLongError(RuntimeError):
    """Raised instead of silently truncating an over-length Agent history."""


class QwenSFTModel:
    """Small adapter that gives AgentLoop the required generate(messages) method."""

    def __init__(
        self,
        base_model: Path,
        adapter_path: Path,
        max_new_tokens: int,
        max_input_tokens: int,
    ) -> None:
        # Heavy GPU dependencies stay local to this class so metric tests can run on CPU-only machines.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SFT Agent Evaluation")

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.generation_records: list[dict] = []

        # train_sft.py saves processing_class into the adapter directory. Loading it here keeps
        # inference aligned with the tokenizer and chat template saved by the actual SFT run.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        if not self.tokenizer.chat_template:
            raise ValueError(f"Tokenizer has no chat template: {adapter_path}")
        self.chat_template = self.tokenizer.chat_template
        self.chat_template_sha256 = hashlib.sha256(
            self.chat_template.encode("utf-8")
        ).hexdigest()
        self.system_prompt_sha256 = hashlib.sha256(
            AGENT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()

        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_end_token_id is None or self.im_end_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Tokenizer does not define a valid <|im_end|> token")

        base = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        self.model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=False,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self._check_chat_template()

    def _check_chat_template(self) -> None:
        probe_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Question: template compatibility probe\n"},
        ]
        rendered = self.tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.count(AGENT_SYSTEM_PROMPT) != 1:
            raise ValueError(
                "Chat template did not render exactly one copy of AGENT_SYSTEM_PROMPT"
            )
        if "template compatibility probe" not in rendered:
            raise ValueError("Chat template did not preserve the user question")

    def reset_trace(self) -> None:
        self.generation_records.clear()

    def generate(self, messages: list[dict[str, str]]) -> str:
        if any(message.get("role") == "system" for message in messages):
            raise ValueError("AgentLoop messages unexpectedly contain a system message")

        inference_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *messages,
        ]
        encoded = self.tokenizer.apply_chat_template(
            inference_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"]
        input_token_count = input_ids.shape[-1]
        if input_token_count > self.max_input_tokens:
            self.generation_records.append(
                {
                    "input_tokens": input_token_count,
                    "status": "input_too_long",
                    "max_input_tokens": self.max_input_tokens,
                }
            )
            raise InputTooLongError(
                f"Agent input has {input_token_count} tokens, exceeding "
                f"--max-input-tokens={self.max_input_tokens}; history was not truncated"
            )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }
        eos_token_ids = list(
            dict.fromkeys(
                token_id
                for token_id in (self.tokenizer.eos_token_id, self.im_end_token_id)
                if token_id is not None
            )
        )
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=eos_token_ids,
                pad_token_id=pad_token_id,
            )

        new_token_ids = generated[0, input_token_count:]
        token_list = new_token_ids.tolist()
        visible_text = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_with_special_tokens = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        final_token_id = token_list[-1] if token_list else None
        self.generation_records.append(
            {
                "input_tokens": input_token_count,
                "generated_tokens": len(token_list),
                "final_token_id": final_token_id,
                "ended_with_eos": final_token_id in eos_token_ids,
                "reached_max_new_tokens": len(token_list) == self.max_new_tokens,
                "decoded_with_special_tokens": decoded_with_special_tokens,
                "status": "generated",
            }
        )
        return visible_text


def normalize_answer(text: str) -> str:
    """Standard English QA normalization: lowercase, punctuation/articles/extra spaces."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def token_f1(prediction: str, gold_answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    common_count = sum(common.values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_scores(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    if not gold_answers:
        raise ValueError("gold_answers must not be empty")
    return (
        max(exact_match(prediction, answer) for answer in gold_answers),
        max(token_f1(prediction, answer) for answer in gold_answers),
    )


def is_strict_action(text: str) -> bool:
    match = STRICT_ACTION_PATTERN.fullmatch(text)
    return match is not None and bool(match.group(2).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3 LoRA agent with the real AgentLoop and Local BM25."
    )
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS
    )
    args = parser.parse_args()
    for name in ("limit", "max_new_tokens", "max_input_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def load_eval_records(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("Evaluation data is empty")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation data contains duplicate IDs")
    unknown_sources = {record["source"] for record in records} - {"nq", "hotpotqa"}
    if unknown_sources:
        raise ValueError(f"Unsupported evaluation sources: {sorted(unknown_sources)}")
    return records


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {
            "samples": 0,
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "valid_action_rate": 0.0,
            "strict_format_rate": 0.0,
            "answer_completion_rate": 0.0,
            "average_search_count": 0.0,
            "total_turns": 0,
            "valid_actions": 0,
            "strict_actions": 0,
        }
    turns = sum(record["total_turns"] for record in records)
    valid_actions = sum(record["valid_action_count"] for record in records)
    strict_actions = sum(record["strict_action_count"] for record in records)
    return {
        "samples": len(records),
        "answer_em": sum(record["em"] for record in records) / len(records),
        "answer_f1": sum(record["f1"] for record in records) / len(records),
        "valid_action_rate": valid_actions / turns if turns else 0.0,
        "strict_format_rate": strict_actions / turns if turns else 0.0,
        "answer_completion_rate": (
            sum(record["termination_reason"] == "answer" for record in records)
            / len(records)
        ),
        "average_search_count": (
            sum(record["search_count"] for record in records) / len(records)
        ),
        "total_turns": turns,
        "valid_actions": valid_actions,
        "strict_actions": strict_actions,
    }


def write_summary(
    path: Path,
    grouped_metrics: dict[str, dict],
    args: argparse.Namespace,
    model: QwenSFTModel,
) -> None:
    lines = [
        "# SFT Agent Evaluation", "",
        "## Configuration", "",
        f"- Base model: `{args.base_model}`",
        f"- Adapter: `{args.adapter_path}`",
        f"- Evaluation data: `{args.eval_path}`",
        f"- BM25 index: `{args.index_path}`",
        f"- Corpus: `{args.corpus_path}`",
        f"- Top-K: {TOP_K}",
        f"- Max turns: {MAX_TURNS}",
        f"- Max new tokens: {args.max_new_tokens}",
        f"- Max input tokens: {args.max_input_tokens} (no silent truncation)",
        f"- System prompt SHA256: `{model.system_prompt_sha256}`",
        f"- Chat template SHA256: `{model.chat_template_sha256}`", "",
        "## Metrics", "",
        "| Group | Samples | EM | F1 | Valid Action | Strict Format | Answer Completion | Avg Searches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("nq", "hotpotqa", "overall"):
        metric = grouped_metrics[group]
        lines.append(
            f"| {group} | {metric['samples']} | {metric['answer_em']:.4f} | "
            f"{metric['answer_f1']:.4f} | {metric['valid_action_rate']:.4f} | "
            f"{metric['strict_format_rate']:.4f} | "
            f"{metric['answer_completion_rate']:.4f} | "
            f"{metric['average_search_count']:.4f} |"
        )
    lines.extend([
        "", "## Metric definitions", "",
        "- EM/F1 use lowercase, punctuation removal, English article removal and whitespace normalization; multiple gold answers take the best score.",
        "- Valid Action Rate is based on the existing `parse_action()` result for every model turn.",
        "- Strict Format Rate requires the complete visible output to contain only one non-empty `<search>` or `<answer>` action.",
        "- Answer Completion Rate is the fraction of questions terminated by an Answer action.",
        "- Over-length input histories are reported and are never silently truncated.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    records = load_eval_records(args.eval_path, args.limit)
    model = QwenSFTModel(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    retriever = BM25Retriever(
        index_path=str(args.index_path),
        corpus_path=str(args.corpus_path),
        top_k=TOP_K,
    )
    environment = SearchEnvironment(
        SearchTool(retriever=retriever, default_top_k=TOP_K)
    )
    loop = AgentLoop(model=model, environment=environment, max_turns=MAX_TURNS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    completed_records = []
    with results_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for index, record in enumerate(records, 1):
            model.reset_trace()
            # 与正式 SFT Messages 对齐：首轮 user content 就是原始 question，不增加前缀。
            # 输入超过 4096 时 QwenSFTModel 会明确抛错并停止，绝不静默截断历史。
            result = loop.run(record["question"])
            steps = result.steps
            termination_reason = result.termination_reason
            predicted_answer = result.final_answer or ""
            search_count = result.search_count

            turn_records = []
            valid_action_count = 0
            strict_action_count = 0
            for turn_index, step in enumerate(steps):
                generation = model.generation_records[turn_index]
                valid = step.action.type != "invalid"
                strict = is_strict_action(step.model_output)
                valid_action_count += int(valid)
                strict_action_count += int(strict)
                turn_records.append(
                    {
                        "turn": turn_index + 1,
                        "raw_model_output": step.model_output,
                        "decoded_with_special_tokens": generation[
                            "decoded_with_special_tokens"
                        ],
                        "parsed_action": {
                            "type": step.action.type,
                            "content": step.action.content,
                        },
                        "strict_format": strict,
                        "information": step.observation if step.action.type == "search" else "",
                        "input_tokens": generation["input_tokens"],
                        "generated_tokens": generation["generated_tokens"],
                        "ended_with_eos": generation["ended_with_eos"],
                        "reached_max_new_tokens": generation[
                            "reached_max_new_tokens"
                        ],
                    }
                )

            em, f1 = best_qa_scores(predicted_answer, record["answers"])
            output = {
                "id": record["id"],
                "source": record["source"],
                "question": record["question"],
                "gold_answers": record["answers"],
                "predicted_answer": predicted_answer,
                "termination_reason": termination_reason,
                "search_count": search_count,
                "total_turns": len(steps),
                "valid_action_count": valid_action_count,
                "strict_action_count": strict_action_count,
                "turns": turn_records,
                "generation_trace": model.generation_records,
                "em": em,
                "f1": f1,
            }
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
            output_file.flush()
            completed_records.append(output)
            print(
                f"[{index}/{len(records)}] {record['id']}: "
                f"termination={termination_reason}, searches={search_count}, "
                f"EM={em:.0f}, F1={f1:.3f}"
            )

    nq_records = [record for record in completed_records if record["source"] == "nq"]
    hotpotqa_records = [
        record for record in completed_records if record["source"] == "hotpotqa"
    ]
    grouped_metrics = {
        "nq": aggregate(nq_records),
        "hotpotqa": aggregate(hotpotqa_records),
        "overall": aggregate(completed_records),
    }
    write_summary(
        args.output_dir / "summary.md",
        grouped_metrics,
        args,
        model,
    )
    print(f"Results: {results_path}")
    print(f"Summary: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
