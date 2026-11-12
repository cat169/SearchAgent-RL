"""Inference-only entry point; run from repository root with python -m."""
from __future__ import annotations
import argparse
import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace


def select_questions(records, count, seed):
    rng = random.Random(seed)
    buckets = {}
    seen = set()
    for row in records:
        if row["extra_info"]["split"] != "train":
            raise ValueError("Only train records are permitted")
        key = str(row["extra_info"]["id"])
        if key in seen:
            raise ValueError(f"Duplicate question_id: {key}")
        seen.add(key)
        buckets.setdefault(row["data_source"], []).append(row)
    if len(buckets) != 2:
        raise ValueError("Expected NQ and HotpotQA sources")
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count:
        progress = False
        for source in sorted(buckets):
            if buckets[source] and len(selected) < count:
                selected.append(buckets[source].pop())
                progress = True
        if not progress:
            raise ValueError("Not enough distinct training questions")
    return selected


class VLLMManager:
    """Token-in/token-out adapter for an immutable evaluation checkpoint.

    Version zero identifies this loaded checkpoint, not a training global step.
    No weight updates occur during its lifetime.
    """
    def __init__(self, engine, sampling_cls, seed):
        self.engine, self.sampling_cls, self.seed = engine, sampling_cls, seed
        self.calls = 0
        self.events = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        params = dict(sampling_params)
        logs = params.pop("logprobs", False)
        params["logprobs"] = 0 if logs else None
        params["seed"] = (self.seed + self.calls) % (2**31)
        self.calls += 1
        sampling = self.sampling_cls(**params)
        outputs = await asyncio.to_thread(
            self.engine.generate, [{"prompt_token_ids": list(prompt_ids)}],
            sampling, use_tqdm=False,
        )
        out = outputs[0].outputs[0]
        ids = list(out.token_ids)
        self.events.append({"request_id": request_id, "prompt_ids": list(prompt_ids),
                            "token_ids": ids, "finish_reason": out.finish_reason})
        log_probs = ([entry[token].logprob for token, entry in zip(ids, out.logprobs)]
                     if logs else None)
        return SimpleNamespace(token_ids=ids, log_probs=log_probs,
                               extra_fields={"min_global_steps": 0, "max_global_steps": 0})


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def evaluate(args):
    from datasets import load_dataset
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoConfig
    from vllm import LLM, SamplingParams
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from training.grpo.agent_loop import SearchAgentLoop
    from training.grpo.reward import compute_score
    from evaluation.grpo_pretrain.analyze_rollouts import analyze

    config = OmegaConf.load(args.grpo_config)
    rollout = config.actor_rollout_ref.rollout
    # Same fixed limits as the current launcher and SearchAgentLoop.
    rollout.prompt_length = 2048
    rollout.response_length = 6144
    rollout.max_model_len = 8192
    sampling = dict(temperature=rollout.temperature, top_p=rollout.top_p,
                    top_k=rollout.top_k, repetition_penalty=1.0,
                    logprobs=rollout.calculate_log_probs)
    if sampling["temperature"] <= 0:
        raise ValueError("Independent stochastic rollouts require training temperature > 0")
    agent = OmegaConf.load(args.agent_config)[0]
    for name in ("index_path", "corpus_path"):
        path = Path(agent[name])
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"{name} must be an existing absolute path: {path}")
    selected = select_questions(load_dataset("parquet", data_files=args.train_parquet,
                                            split="train"),
                                8 if args.mode == "pilot" else 40, args.seed)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "rollouts.jsonl").exists():
        raise FileExistsError("Use a fresh output directory; existing rollouts are preserved")
    dump(dest / "selected_questions.json", selected)
    dump(dest / "run_config.json", {**vars(args), "sampling_params": sampling,
         "grpo_config": OmegaConf.to_container(config, resolve=True),
         "agent_config": OmegaConf.to_container(agent, resolve=True),
         "evaluation_model_version": 0, "expected_rollouts": 4})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    custom = config.actor_rollout_ref.model.get("custom_chat_template")
    if custom:
        tokenizer.chat_template = custom
    engine = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=args.tensor_parallel_size,
                 max_model_len=8192, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
                 enforce_eager=True)
    manager = VLLMManager(engine, SamplingParams, args.seed)
    loop = SearchAgentLoop(trainer_config=DictConfigWrap(config), server_manager=manager,
        tokenizer=tokenizer, processor=None, dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        hf_model_type=AutoConfig.from_pretrained(args.model, local_files_only=True).model_type,
        index_path=str(agent.index_path), corpus_path=str(agent.corpus_path))
    # Observe context merges without replacing veRL tokenization or loop behavior.
    original_merge = loop.ct_merge_context_msg
    observations = []
    async def capture_context(previous_messages, updated_messages, *pos, **kw):
        observations.extend(updated_messages[len(previous_messages):])
        return await original_merge(previous_messages, updated_messages, *pos, **kw)
    loop.ct_merge_context_msg = capture_context
    # Sequential rollouts share one engine and one process-local BM25 instance.
    with (dest / "rollouts.jsonl").open("x", encoding="utf-8") as stream:
        for row in selected:
            for rollout_id in range(4):
                manager.events = []
                observations.clear()
                record = dict(question_id=str(row["extra_info"]["id"]),
                    data_source=row["data_source"], question=row["prompt"][-1]["content"],
                    ground_truth=row["reward_model"]["ground_truth"], rollout_id=rollout_id)
                try:
                    out = await loop.run(dict(sampling), raw_prompt=row["prompt"])
                    info = out.extra_fields
                    record.update(info)
                    record.update(status="ok", trajectory_text=tokenizer.decode(
                        out.prompt_ids + out.response_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False),
                        prompt_token_count=len(out.prompt_ids), response_token_count=len(out.response_ids),
                        response_mask_ones=out.response_mask.count(1),
                        response_mask_zeros=out.response_mask.count(0),
                        prompt_ids=out.prompt_ids, response_ids=out.response_ids,
                        response_mask=out.response_mask)
                    record["reward"] = compute_score(row["data_source"], record["trajectory_text"],
                                                       record["ground_truth"], info)
                except Exception as error:
                    record.update(status="error", error_type=type(error).__name__, error=str(error),
                        termination_reason="runtime_error", final_answer=None,
                        strict_protocol_valid=None, search_count=None, reward=None,
                        prompt_token_count=None, response_token_count=None,
                        response_mask_ones=None, response_mask_zeros=None, trajectory_text=None)
                record["observations"] = list(observations)
                record["generation_events"] = [{**event, "raw_text": tokenizer.decode(
                    event["token_ids"], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)} for event in manager.events]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
    analyze(dest / "rollouts.jsonl", dest, selected_path=dest / "selected_questions.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--grpo-config", required=True, help="Resolved training Hydra YAML (--cfg job --resolve)")
    parser.add_argument("--agent-config", default="training/grpo/search_agent.remote.yaml")
    parser.add_argument("--mode", choices=["pilot", "diagnostic"], default="pilot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
