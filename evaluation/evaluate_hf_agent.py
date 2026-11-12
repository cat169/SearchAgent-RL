"""Unified greedy evaluation for complete HF SFT and GRPO checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from evaluation import evaluate_agent as old


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BudgetModel(old.QwenSFTModel):
    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError("Missing model chat template")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids(old.IM_END_TOKEN)
        if self.im_end_token_id is None or self.im_end_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Missing im_end token")
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_input_tokens = 8192
        self.max_new_tokens = 512
        self.initial_tokens = None
        self.budget_stop = False

    def generate_turn(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        n = len(ids)
        if self.initial_tokens is None:
            self.initial_tokens = n
            if n > 2048:
                raise old.InputTooLongError(n, 2048)
        remaining = min(6144 - (n - self.initial_tokens), 8192 - n)
        if remaining <= 0:
            self.budget_stop = True
            raise old.InputTooLongError(n, 8192)
        self.max_new_tokens = min(512, remaining)
        result = super().generate_turn(messages)
        if not result.protocol.ended_with_im_end and len(result.token_ids) >= remaining:
            self.budget_stop = True
        return result


def grouped(rows):
    result = {}
    for source in ("nq", "hotpotqa", "overall"):
        subset = rows if source == "overall" else [r for r in rows if r["source"] == source]
        m = old.aggregate(subset)
        n = len(subset)
        m["average_assistant_decision_turns"] = sum(r["total_turns"] for r in subset)/n if n else 0
        for reason in ("length_limit", "invalid_action"):
            m[reason + "_rate"] = sum(r["termination_reason"] == reason for r in subset)/n if n else 0
        result[source] = m
    return result


def isolation(eval_path, train_path):
    from datasets import load_dataset
    ev = old.load_eval_records(Path(eval_path), None)
    train = load_dataset("parquet", data_files=str(train_path), split="train")
    ids, questions = set(), set()
    for row in train:
        extra = row.get("extra_info") or {}
        identity = extra.get("id", row.get("id"))
        if identity is not None:
            ids.add(str(identity))
        question = row.get("question")
        if question is None:
            prompt = row.get("prompt") or row.get("raw_prompt")
            if not isinstance(prompt, list):
                raise ValueError("Cannot extract question from training schema")
            question = next(m["content"] for m in reversed(prompt) if m["role"] == "user")
        questions.add(old.normalize_answer(question))
    return {"id_overlap": [r["id"] for r in ev if str(r["id"]) in ids],
            "question_overlap": [r["id"] for r in ev if old.normalize_answer(r["question"]) in questions]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "eval-path", "index-path", "corpus-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=name == "eval-path")
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--check-isolation", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.check_isolation:
        if not args.train_parquet:
            parser.error("--train-parquet required")
        print(json.dumps(isolation(args.eval_path, args.train_parquet), indent=2))
        return
    if any(getattr(args, n) is None for n in ("model_path", "index_path", "corpus_path", "output_dir")):
        parser.error("model, index, corpus and output paths are required for evaluation")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    records = old.load_eval_records(args.eval_path, args.limit)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items() if k != "resume"}
    config["data_sha256"] = digest(args.eval_path)
    config["model_files"] = {str(p.relative_to(args.model_path)): digest(p) for p in args.model_path.glob("*.json")}
    config["model_weights"] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in args.model_path.glob("*.safetensors")}
    config["system_prompt_sha256"] = hashlib.sha256(old.AGENT_SYSTEM_PROMPT.encode()).hexdigest()
    config["code_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config["evaluation_code_sha256"] = digest(__file__)
    config["conditions"] = dict(greedy=True, top_k=3, turns=5, per_turn=512, prompt=2048, response=6144, total=8192)
    cp = args.output_dir / "config.json"
    if args.resume and json.loads(cp.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume configuration mismatch")
    path, completed, mode = old.prepare_results(args.output_dir, args.resume, records)
    if not args.resume:
        cp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    from agent.environment import SearchEnvironment
    from agent.search_tool import SearchTool
    from retrieval.bm25_retriever import BM25Retriever
    model = BudgetModel(str(args.model_path), args.device)
    env = SearchEnvironment(SearchTool(BM25Retriever(str(args.index_path), str(args.corpus_path), top_k=3)))
    with path.open(mode, encoding="utf-8") as stream:
        for record in records[len(completed):]:
            model.initial_tokens = None
            model.budget_stop = False
            result = old.run_agent(model, env, record["question"])
            if model.budget_stop:
                result.termination_reason = "length_limit"
            output = old.build_output_record(record, result)
            if result.termination_reason != "answer":
                output["em"] = output["f1"] = 0.0
            output["initial_prompt_tokens"] = model.initial_tokens
            output["assistant_decision_turns"] = len(result.steps)
            completed.append(output)
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()
    metrics = grouped(completed)
    (args.output_dir / "summary.md").write_text("# Unified HF Agent Evaluation\n\n```json\n" + json.dumps(metrics, indent=2) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
