"""Search rollout for veRL v0.9.1 (1876b06d0a3e4e71e06230be10af14492ca8a75b)."""

from __future__ import annotations

import asyncio
import threading
from time import perf_counter
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from agent.raw_protocol import IM_END_TOKEN, assistant_message, parse_raw_generation


PROMPT_BUDGET = 2048
RESPONSE_BUDGET = 6144
TOTAL_BUDGET = 8192
MAX_NEW_TOKENS = 512
MAX_ASSISTANT_TURNS = 5
TOP_K = 3
DEFAULT_INDEX_PATH = "data/retrieval/wiki-18-bm25-index/bm25"
DEFAULT_CORPUS_PATH = "data/retrieval/wiki-18-corpus/wiki-18.jsonl"

_environments = {}
_search_lock = threading.Lock()


def _step_environment(action_text, index_path, corpus_path):
    # Worker 内复用检索实例，并串行执行同步 JVM 检索；本函数在工作线程调用。
    with _search_lock:
        key = (index_path, corpus_path)
        if key not in _environments:
            from agent.environment import SearchEnvironment
            from agent.search_tool import SearchTool
            from retrieval.bm25_retriever import BM25Retriever

            retriever = BM25Retriever(
                index_path=index_path, corpus_path=corpus_path, top_k=TOP_K
            )
            _environments[key] = SearchEnvironment(
                SearchTool(retriever=retriever, default_top_k=TOP_K)
            )
        return _environments[key].step(action_text)


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        hf_model_type=None,
        index_path=DEFAULT_INDEX_PATH,
        corpus_path=DEFAULT_CORPUS_PATH,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            hf_model_type=hf_model_type,
            **kwargs,
        )
        # 不超过本轮预算，也不超过框架将用于 padding 的实际长度。
        self.prompt_budget = min(PROMPT_BUDGET, self.rollout_config.prompt_length)
        self.response_budget = min(RESPONSE_BUDGET, self.rollout_config.response_length)
        self.total_budget = min(TOTAL_BUDGET, self.rollout_config.max_model_len or TOTAL_BUDGET)
        self.im_end_token_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
        if self.im_end_token_id is None or self.im_end_token_id == tokenizer.unk_token_id:
            raise ValueError("Tokenizer does not define a valid <|im_end|> token")
        self.index_path = index_path
        self.corpus_path = corpus_path

    def _cap_text_prompt_length(self, prompt_ids):
        # 覆盖 veRL 默认的左截断；初始问题超限时明确报错，不丢弃 Prompt。
        if len(prompt_ids) > min(self.prompt_budget, self.total_budget):
            raise ValueError(
                f"Initial prompt has {len(prompt_ids)} tokens, exceeding "
                f"budget={min(self.prompt_budget, self.total_budget)}; not truncated"
            )
        return prompt_ids

    async def run(self, sampling_params, **kwargs):
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.ct_build_initial_tokens(messages)
        runtime_ids = list(prompt_ids)
        response_mask = []
        response_logprobs = [] if self.rollout_config.calculate_log_probs else None
        metrics = AgentLoopMetrics()
        request_id = uuid4().hex
        final_answer = ""
        termination_reason = "max_turns"
        search_count = 0
        assistant_turns = 0
        observation_turns = 0
        strict_protocol_valid = True
        min_global_steps = None
        max_global_steps = None

        for _ in range(MAX_ASSISTANT_TURNS):
            remaining = min(
                self.response_budget - len(response_mask),
                self.total_budget - len(runtime_ids),
            )
            if remaining <= 0:
                termination_reason = "length_limit"
                break

            turn_params = dict(sampling_params)
            max_tokens = min(MAX_NEW_TOKENS, remaining)
            turn_params["max_tokens"] = max_tokens
            # 使用 Token stop，而非文本截取；其他采样参数由 veRL 控制。
            turn_params["stop_token_ids"] = list(
                dict.fromkeys([*(turn_params.get("stop_token_ids") or []), self.im_end_token_id])
            )
            started = perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=runtime_ids,
                sampling_params=turn_params,
            )
            versions = getattr(output, "extra_fields", None) or {}
            low = versions.get("min_global_steps")
            high = versions.get("max_global_steps")
            if low is None or high is None:
                raise ValueError("Generation is missing min_global_steps/max_global_steps metadata")
            min_global_steps = low if min_global_steps is None else min(min_global_steps, low)
            max_global_steps = high if max_global_steps is None else max(max_global_steps, high)
            metrics.generate_sequences += perf_counter() - started
            if len(output.token_ids) > max_tokens:
                raise RuntimeError("Inference server exceeded the requested token budget")
            assistant_turns += 1

            merged, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                runtime_ids,
                output.token_ids,
                response_mask,
                response_logprobs,
                assistant_logprobs=output.log_probs,
            )
            runtime_ids = merged.token_ids
            raw_text = self.tokenizer.decode(
                output.token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            protocol = parse_raw_generation(raw_text, output.token_ids, self.im_end_token_id)
            if not protocol.strict_valid:
                strict_protocol_valid = False
                budget_exhausted = (
                    len(response_mask) >= self.response_budget
                    or len(runtime_ids) >= self.total_budget
                )
                termination_reason = (
                    "length_limit"
                    if budget_exhausted and not protocol.ended_with_im_end
                    else "invalid_action"
                )
                break

            messages.append(assistant_message(protocol))
            if protocol.action.type == "answer":
                final_answer = protocol.action.content
                termination_reason = "answer"
                break

            action_text = f"<search>{protocol.action.content}</search>"
            started = perf_counter()
            step_result = await asyncio.to_thread(
                _step_environment, action_text, self.index_path, self.corpus_path
            )
            metrics.tool_calls += perf_counter() - started
            search_count += 1
            previous_messages = list(messages)
            updated_messages = [
                *messages,
                {"role": "user", "content": step_result.observation.strip()},
            ]
            merged, next_mask, next_logprobs = await self.ct_merge_context_msg(
                previous_messages,
                updated_messages,
                runtime_ids,
                response_mask,
                response_logprobs,
            )
            # 工具观察必须整体容纳；超限时不提交该观察及其 Header，也不裁剪。
            if (
                len(next_mask) > self.response_budget
                or len(merged.token_ids) > self.total_budget
            ):
                termination_reason = "length_limit"
                break
            runtime_ids = merged.token_ids
            response_mask = next_mask
            response_logprobs = next_logprobs
            messages = updated_messages
            observation_turns += 1

        response_ids = runtime_ids[len(prompt_ids):]
        if (
            runtime_ids[:len(prompt_ids)] != prompt_ids
            or len(response_ids) != len(response_mask)
            or len(response_ids) > self.response_budget
            or len(runtime_ids) > self.total_budget
        ):
            raise RuntimeError("Continuous Token output violates prefix, mask or budget alignment")

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            metrics=metrics,
            num_turns=1 + assistant_turns + observation_turns,
            extra_fields={
                "min_global_steps": min_global_steps,
                "max_global_steps": max_global_steps,
                "final_answer": final_answer,
                "termination_reason": termination_reason,
                "search_count": search_count,
                "strict_protocol_valid": strict_protocol_valid,
            },
        )
