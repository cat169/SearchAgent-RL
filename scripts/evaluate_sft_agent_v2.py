"""Evaluate SFT v2 with strict raw protocol parsing and SFT-aligned history."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.protocol import AgentAction
from agent.system_prompt import AGENT_SYSTEM_PROMPT


DEFAULT_EVAL_PATH = Path("data/processed/evaluation/nq_hotpotqa_v1/eval.jsonl")
DEFAULT_BASE_MODEL = Path("/root/models/Qwen3-4B-Base")
DEFAULT_ADAPTER_PATH = Path("/root/SearchAgent-RL/outputs/sft_v2")
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
IM_END_TOKEN = "<|im_end|>"

ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
STRICT_RAW_PATTERN = re.compile(
    r"<think>(.*?)</think>\s*<(search|answer)>(.*?)</\2><\|im_end\|>",
    re.DOTALL,
)
RESERVED_TAG_PATTERN = re.compile(
    r"</?(?:think|search|answer|information)>|<\|(?:im_end|endoftext)\|>"
)


class InputTooLongError(RuntimeError):
    """Raised instead of silently truncating an over-length Agent history."""

    def __init__(self, input_tokens: int, max_input_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens
        super().__init__(
            f"Agent input has {input_tokens} tokens, exceeding "
            f"--max-input-tokens={max_input_tokens}; history was not truncated"
        )


@dataclass(frozen=True)
class RawProtocolResult:
    strict_valid: bool
    valid_action: bool
    ended_with_im_end: bool
    reasoning_content: str
    action: AgentAction
    error: str | None


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    token_ids: list[int]
    input_tokens: int
    final_token_id: int | None
    reached_max_new_tokens: bool
    protocol: RawProtocolResult
    history_prefix_preserved: bool
    completed_turn_matches_template: bool | None = None


@dataclass
class EvaluationStep:
    generation: GenerationResult
    observation: str


@dataclass
class EvaluationResult:
    steps: list[EvaluationStep]
    termination_reason: str
    final_answer: str | None
    input_tokens_at_failure: int | None = None
    max_input_tokens: int | None = None

    @property
    def search_count(self) -> int:
        return sum(step.generation.protocol.action.type == "search" for step in self.steps)


def _action_candidate(raw_text: str) -> tuple[bool, AgentAction]:
    """Find one usable action for diagnostics without authorizing its execution."""
    if "<information>" in raw_text or "</information>" in raw_text:
        return False, AgentAction(type="invalid", content="")
    matches = ACTION_PATTERN.findall(raw_text)
    if len(matches) != 1:
        return False, AgentAction(type="invalid", content="")
    action_type, content = matches[0]
    content = content.strip()
    if not content or RESERVED_TAG_PATTERN.search(content):
        return False, AgentAction(type="invalid", content="")
    return True, AgentAction(type=action_type, content=content)


def parse_raw_generation(
    raw_text: str,
    token_ids: list[int],
    im_end_token_id: int,
) -> RawProtocolResult:
    """Validate the exact SFT v2 raw response, including its terminal token."""
    valid_action, candidate = _action_candidate(raw_text)
    ended_with_im_end = bool(token_ids) and token_ids[-1] == im_end_token_id

    if not ended_with_im_end:
        return RawProtocolResult(
            False, valid_action, False, "", AgentAction("invalid", ""), "missing_im_end"
        )
    if "\ufffd" in raw_text:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "decode_error"
        )
    if raw_text.count("<think>") != 1 or raw_text.count("</think>") != 1:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "invalid_think_count"
        )
    if "<information>" in raw_text or "</information>" in raw_text:
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "forged_information"
        )

    match = STRICT_RAW_PATTERN.fullmatch(raw_text)
    if match is None:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "raw_protocol_mismatch"
        )

    reasoning_content, action_type, action_content = match.groups()
    reasoning_content = reasoning_content.strip()
    action_content = action_content.strip()
    if not reasoning_content:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "empty_think"
        )
    if not action_content:
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "empty_action"
        )
    if RESERVED_TAG_PATTERN.search(reasoning_content):
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "reserved_tag_in_think"
        )
    if RESERVED_TAG_PATTERN.search(action_content):
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "reserved_tag_in_action"
        )

    return RawProtocolResult(
        strict_valid=True,
        valid_action=True,
        ended_with_im_end=True,
        reasoning_content=reasoning_content,
        action=AgentAction(type=action_type, content=action_content),
        error=None,
    )


def assistant_message(protocol: RawProtocolResult) -> dict[str, str]:
    if not protocol.strict_valid:
        raise ValueError("Cannot add an invalid generation to Agent history")
    action = protocol.action
    return {
        "role": "assistant",
        "reasoning_content": protocol.reasoning_content,
        "content": f"<{action.type}>{action.content}</{action.type}>",
    }


def render_history(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def check_history_prefix(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """Ensure inference adds a generation header without rewriting prior messages."""
    training_history = render_history(tokenizer, messages, False)
    inference_prompt = render_history(tokenizer, messages, True)
    if not inference_prompt.startswith(training_history):
        raise RuntimeError(
            "Chat template rewrote completed history when add_generation_prompt=True"
        )
    return training_history, inference_prompt


def completed_turn_matches_template(
    tokenizer,
    messages_before_turn: list[dict],
    inference_prompt: str,
    protocol: RawProtocolResult,
    raw_text: str,
) -> bool:
    completed = [*messages_before_turn, assistant_message(protocol)]
    rendered = render_history(tokenizer, completed, False)
    if not rendered.startswith(inference_prompt):
        return False
    return rendered[len(inference_prompt):].rstrip("\r\n") == raw_text


class QwenSFTV2Model:
    """GPU model adapter that preserves raw generation and structured history."""

    def __init__(
        self,
        base_model: Path,
        adapter_path: Path,
        max_new_tokens: int,
        max_input_tokens: int,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SFT v2 Agent Evaluation")

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), local_files_only=True, trust_remote_code=True
        )
        if not self.tokenizer.chat_template:
            raise ValueError(f"Tokenizer has no chat template: {adapter_path}")
        self.chat_template_sha256 = hashlib.sha256(
            self.tokenizer.chat_template.encode("utf-8")
        ).hexdigest()
        self.system_prompt_sha256 = hashlib.sha256(
            AGENT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
        if (
            self.im_end_token_id is None
            or self.im_end_token_id == self.tokenizer.unk_token_id
        ):
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
            base, str(adapter_path), is_trainable=False
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate_turn(self, messages: list[dict]) -> GenerationResult:
        _, inference_prompt = check_history_prefix(self.tokenizer, messages)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"]
        input_token_count = input_ids.shape[-1]
        if input_token_count > self.max_input_tokens:
            raise InputTooLongError(input_token_count, self.max_input_tokens)

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.im_end_token_id,
                pad_token_id=pad_token_id,
            )

        new_token_ids = generated[0, input_token_count:]
        token_ids = new_token_ids.tolist()
        raw_text = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        protocol = parse_raw_generation(raw_text, token_ids, self.im_end_token_id)
        template_match = None
        if protocol.strict_valid:
            template_match = completed_turn_matches_template(
                self.tokenizer, messages, inference_prompt, protocol, raw_text
            )
        return GenerationResult(
            raw_text=raw_text,
            token_ids=token_ids,
            input_tokens=input_token_count,
            final_token_id=token_ids[-1] if token_ids else None,
            reached_max_new_tokens=len(token_ids) == self.max_new_tokens,
            protocol=protocol,
            history_prefix_preserved=True,
            completed_turn_matches_template=template_match,
        )


def run_agent(model: QwenSFTV2Model, environment, question: str) -> EvaluationResult:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[EvaluationStep] = []

    for _ in range(MAX_TURNS):
        try:
            generation = model.generate_turn(messages)
        except InputTooLongError as error:
            return EvaluationResult(
                steps=steps,
                termination_reason="input_too_long",
                final_answer=None,
                input_tokens_at_failure=error.input_tokens,
                max_input_tokens=error.max_input_tokens,
            )
        protocol = generation.protocol
        if not protocol.strict_valid:
            steps.append(EvaluationStep(generation=generation, observation=""))
            return EvaluationResult(steps, "invalid_action", None)

        messages.append(assistant_message(protocol))
        action = protocol.action
        action_text = f"<{action.type}>{action.content}</{action.type}>"
        step_result = environment.step(action_text)
        if step_result.action.type != action.type or step_result.action.content != action.content:
            raise RuntimeError("SearchEnvironment changed an already validated action")

        observation = step_result.observation
        steps.append(EvaluationStep(generation=generation, observation=observation))
        if action.type == "answer":
            return EvaluationResult(steps, "answer", action.content)

        messages.append({"role": "user", "content": observation.strip()})

    return EvaluationResult(steps, "max_turns", None)


def normalize_answer(text: str) -> str:
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
    common_count = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_scores(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    return (
        max(exact_match(prediction, answer) for answer in gold_answers),
        max(token_f1(prediction, answer) for answer in gold_answers),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the SFT v2 Agent protocol.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and append to an existing results.jsonl prefix.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
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
    return records


def load_resume_results(path: Path, eval_records: list[dict]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume without an existing results file: {path}")

    completed: list[dict] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                raise ValueError(f"Blank line in results file at line {line_number}")
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in results file at line {line_number}: {error}"
                ) from error
            if not isinstance(result, dict):
                raise ValueError(f"Result at line {line_number} is not a JSON object")
            result_id = result.get("id")
            if result_id in seen_ids:
                raise ValueError(f"Duplicate result ID at line {line_number}: {result_id}")
            seen_ids.add(result_id)
            completed.append(result)

    if len(completed) > len(eval_records):
        raise ValueError(
            f"Results contain {len(completed)} rows but Eval contains {len(eval_records)}"
        )
    for index, result in enumerate(completed):
        expected = eval_records[index]
        checks = {
            "id": (result.get("id"), expected["id"]),
            "question": (result.get("question"), expected["question"]),
            "gold_answers": (result.get("gold_answers"), expected["answers"]),
        }
        mismatches = [
            name for name, (actual, wanted) in checks.items() if actual != wanted
        ]
        if mismatches:
            raise ValueError(
                f"Results are not a continuous Eval prefix at line {index + 1} "
                f"({expected['id']}): mismatched {', '.join(mismatches)}"
            )
    return completed


def prepare_results(
    output_dir: Path,
    resume: bool,
    eval_records: list[dict],
) -> tuple[Path, list[dict], str]:
    results_path = output_dir / "results.jsonl"
    if resume:
        completed = load_resume_results(results_path, eval_records)
        return results_path, completed, "a"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return results_path, [], "w"


def aggregate(records: list[dict]) -> dict:
    turns = sum(record["total_turns"] for record in records)
    if not records:
        return {
            "samples": 0,
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "valid_action_rate": 0.0,
            "strict_raw_protocol_rate": 0.0,
            "eos_rate": 0.0,
            "answer_completion_rate": 0.0,
            "average_search_count": 0.0,
            "max_turns_rate": 0.0,
            "input_too_long_count": 0,
            "input_too_long_rate": 0.0,
        }
    return {
        "samples": len(records),
        "answer_em": sum(record["em"] for record in records) / len(records),
        "answer_f1": sum(record["f1"] for record in records) / len(records),
        "valid_action_rate": sum(record["valid_action_count"] for record in records) / turns if turns else 0.0,
        "strict_raw_protocol_rate": sum(record["strict_protocol_count"] for record in records) / turns if turns else 0.0,
        "eos_rate": sum(record["eos_count"] for record in records) / turns if turns else 0.0,
        "answer_completion_rate": sum(record["termination_reason"] == "answer" for record in records) / len(records),
        "average_search_count": sum(record["search_count"] for record in records) / len(records),
        "max_turns_rate": sum(record["termination_reason"] == "max_turns" for record in records) / len(records),
        "input_too_long_count": sum(
            record["termination_reason"] == "input_too_long" for record in records
        ),
        "input_too_long_rate": sum(
            record["termination_reason"] == "input_too_long" for record in records
        ) / len(records),
    }


def write_summary(path: Path, grouped_metrics: dict[str, dict], args, model) -> None:
    lines = [
        "# SFT v2 Agent Evaluation",
        "",
        "## Configuration",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Adapter: `{args.adapter_path}`",
        f"- Evaluation data: `{args.eval_path}`",
        f"- Top-K / max turns: {TOP_K} / {MAX_TURNS}",
        f"- Max new/input tokens: {args.max_new_tokens} / {args.max_input_tokens}",
        f"- System prompt SHA256: `{model.system_prompt_sha256}`",
        f"- Chat template SHA256: `{model.chat_template_sha256}`",
        "",
        "## Metrics",
        "",
        "| Group | N | EM | F1 | Valid Action | Strict Raw Protocol | EOS | Answer Completion | Avg Search | Max Turns | Input Too Long |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("nq", "hotpotqa", "overall"):
        metric = grouped_metrics[group]
        lines.append(
            f"| {group} | {metric['samples']} | {metric['answer_em']:.4f} | "
            f"{metric['answer_f1']:.4f} | {metric['valid_action_rate']:.4f} | "
            f"{metric['strict_raw_protocol_rate']:.4f} | {metric['eos_rate']:.4f} | "
            f"{metric['answer_completion_rate']:.4f} | "
            f"{metric['average_search_count']:.4f} | {metric['max_turns_rate']:.4f} | "
            f"{metric['input_too_long_count']} ({metric['input_too_long_rate']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Valid Action: exactly one non-empty Search or Answer candidate, independent of Think/EOS validity.",
            "- Strict Raw Protocol: one complete Think, then one Action, then `<|im_end|>`, with no extra protocol text.",
            "- EOS Rate: fraction of generated turns whose final token is `<|im_end|>`.",
            "- Answer Completion: fraction of questions terminating with a valid Answer.",
            "- Input Too Long: questions stopped without truncation because the next-turn history exceeded the configured input limit.",
            "- EM/F1 measure final answer correctness and are independent of protocol validity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_record(record: dict, result: EvaluationResult) -> dict:
    turns = []
    for turn_index, step in enumerate(result.steps, 1):
        generation = step.generation
        turns.append(
            {
                "turn": turn_index,
                "raw_generation": generation.raw_text,
                "generated_token_ids": generation.token_ids,
                "input_tokens": generation.input_tokens,
                "generated_tokens": len(generation.token_ids),
                "final_token_id": generation.final_token_id,
                "reached_max_new_tokens": generation.reached_max_new_tokens,
                "protocol": asdict(generation.protocol),
                "history_prefix_preserved": generation.history_prefix_preserved,
                "completed_turn_matches_template": generation.completed_turn_matches_template,
                "information": step.observation,
            }
        )

    prediction = result.final_answer or ""
    if result.termination_reason == "input_too_long":
        em, f1 = 0.0, 0.0
    else:
        em, f1 = best_qa_scores(prediction, record["answers"])
    return {
        "id": record["id"],
        "source": record["source"],
        "question": record["question"],
        "gold_answers": record["answers"],
        "predicted_answer": prediction,
        "termination_reason": result.termination_reason,
        "input_tokens_at_failure": result.input_tokens_at_failure,
        "max_input_tokens": result.max_input_tokens,
        "search_count": result.search_count,
        "total_turns": len(result.steps),
        "valid_action_count": sum(
            step.generation.protocol.valid_action for step in result.steps
        ),
        "strict_protocol_count": sum(
            step.generation.protocol.strict_valid for step in result.steps
        ),
        "eos_count": sum(
            step.generation.protocol.ended_with_im_end for step in result.steps
        ),
        "turns": turns,
        "em": em,
        "f1": f1,
    }


def evaluate_records(
    records: list[dict],
    model,
    environment,
    output_file,
    start_index: int,
    total_records: int,
) -> list[dict]:
    outputs = []
    for offset, record in enumerate(records, 1):
        result = run_agent(model, environment, record["question"])
        output = build_output_record(record, result)
        output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
        output_file.flush()
        outputs.append(output)
        print(
            f"[{start_index + offset}/{total_records}] {record['id']}: "
            f"termination={result.termination_reason}, searches={result.search_count}, "
            f"EM={output['em']:.0f}, F1={output['f1']:.3f}"
        )
    return outputs


def main() -> None:
    args = parse_args()

    records = load_eval_records(args.eval_path, args.limit)
    results_path, completed, output_mode = prepare_results(
        args.output_dir, args.resume, records
    )
    remaining_records = records[len(completed):]

    from agent.environment import SearchEnvironment
    from agent.search_tool import SearchTool
    from retrieval.bm25_retriever import BM25Retriever

    model = QwenSFTV2Model(
        args.base_model, args.adapter_path, args.max_new_tokens, args.max_input_tokens
    )
    retriever = BM25Retriever(
        index_path=str(args.index_path), corpus_path=str(args.corpus_path), top_k=TOP_K
    )
    environment = SearchEnvironment(SearchTool(retriever, default_top_k=TOP_K))

    if remaining_records:
        with results_path.open(
            output_mode, encoding="utf-8", newline="\n"
        ) as output_file:
            completed.extend(
                evaluate_records(
                    remaining_records,
                    model,
                    environment,
                    output_file,
                    start_index=len(completed),
                    total_records=len(records),
                )
            )
    else:
        print(f"All {len(records)} evaluation records are already complete.")

    grouped = {
        "nq": aggregate([record for record in completed if record["source"] == "nq"]),
        "hotpotqa": aggregate([record for record in completed if record["source"] == "hotpotqa"]),
        "overall": aggregate(completed),
    }
    write_summary(args.output_dir / "summary.md", grouped, args, model)
    print(f"Results: {results_path}")
    print(f"Summary: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
