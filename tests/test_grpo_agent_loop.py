"""CPU control-flow tests using explicit veRL/CT doubles, not real veRL validation."""

import asyncio
import importlib.util
from pathlib import Path
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


EOS = 151645
SEARCH = "<think>Need evidence.</think><search>target query</search><|im_end|>"
ANSWER = "<think>Evidence is sufficient.</think><answer>Paris</answer><|im_end|>"
PROMPT = [10, 11, 12]
CONTEXT = [900, 901, 902]  # Simulated CT boundary, observation and generation header.


class FakeAgentLoopBase:
    def __init__(self, trainer_config, server_manager, tokenizer, **kwargs):
        self.rollout_config = trainer_config.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.context_ids = list(CONTEXT)
        self.context_messages = []

    async def ct_build_initial_tokens(self, messages):
        self.initial_messages = list(messages)
        return self._cap_text_prompt_length(list(self.tokenizer.prompt_ids))

    async def ct_merge_assistant_token(
        self, runtime_token_ids, assistant_token_ids, response_mask,
        response_logprobs=None, assistant_logprobs=None,
    ):
        logs = None
        if response_logprobs is not None:
            logs = response_logprobs + assistant_logprobs
        return (
            SimpleNamespace(token_ids=runtime_token_ids + assistant_token_ids),
            response_mask + [1] * len(assistant_token_ids),
            logs,
        )

    async def ct_merge_context_msg(
        self, previous_messages, updated_messages, runtime_token_ids,
        response_mask, response_logprobs=None, tools=None,
    ):
        self.context_messages.append((list(previous_messages), list(updated_messages)))
        logs = None
        if response_logprobs is not None:
            logs = response_logprobs + [0.0] * len(self.context_ids)
        return (
            SimpleNamespace(token_ids=runtime_token_ids + self.context_ids),
            response_mask + [0] * len(self.context_ids),
            logs,
        )


def load_loop_with_verl_double():
    api = ModuleType("verl.experimental.agent_loop.agent_loop")
    api.AgentLoopBase = FakeAgentLoopBase
    api.AgentLoopMetrics = lambda: SimpleNamespace(generate_sequences=0.0, tool_calls=0.0)
    api.AgentLoopOutput = lambda **kwargs: SimpleNamespace(**kwargs)
    api.register = lambda name: lambda cls: cls
    modules = {
        name: ModuleType(name)
        for name in ("verl", "verl.experimental", "verl.experimental.agent_loop")
    }
    modules[api.__name__] = api
    path = Path(__file__).resolve().parents[1] / "training/grpo/agent_loop.py"
    spec = importlib.util.spec_from_file_location("grpo_loop_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module


agent_loop = load_loop_with_verl_double()


class FakeTokenizer:
    prompt_ids = PROMPT
    unk_token_id = None

    def __init__(self, texts):
        self.texts = texts

    def convert_tokens_to_ids(self, token):
        return EOS

    def decode(self, tokens, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        return self.texts[tuple(tokens)]

    def encode(self, *args, **kwargs):
        raise AssertionError("Sampled model tokens must not be re-encoded")


class FakeServer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        self.calls.append((request_id, list(prompt_ids), dict(sampling_params)))
        sampling_params.pop("max_tokens")
        tokens = self.responses.pop(0)
        return SimpleNamespace(
            token_ids=tokens,
            log_probs=[-0.5] * len(tokens),
            stop_reason="completed",
            extra_fields={"min_global_steps": 0, "max_global_steps": len(self.calls)},
        )


class SearchAgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, responses, texts):
        server = FakeServer(responses)
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    prompt_length=2048,
                    response_length=6144,
                    max_model_len=8192,
                    calculate_log_probs=True,
                )
            )
        )
        loop = agent_loop.SearchAgentLoop(
            trainer_config=SimpleNamespace(config=config),
            server_manager=server,
            tokenizer=FakeTokenizer(texts),
            processor=None,
            dataset_cls=None,
            data_config=SimpleNamespace(config={}),
        )
        return loop, server

    async def test_search_information_answer_keeps_tokens_masks_and_sampling(self):
        search_ids = [101, 102, EOS]
        answer_ids = [201, 202, EOS]
        loop, server = self.make_loop(
            [search_ids, answer_ids], {tuple(search_ids): SEARCH, tuple(answer_ids): ANSWER}
        )
        messages = [{"role": "system", "content": "original system"},
                    {"role": "user", "content": "original question"}]
        params = {"temperature": 0.8, "top_p": 0.9}
        main_thread = threading.get_ident()
        tool_threads = []

        def search(action, index, corpus):
            self.assertEqual(action, "<search>target query</search>")
            tool_threads.append(threading.get_ident())
            return SimpleNamespace(observation="\n\n<information>Paris evidence</information>\n\n")

        with patch.object(agent_loop, "_step_environment", side_effect=search) as tool:
            result = await loop.run(params, raw_prompt=messages)
        tool.assert_called_once()
        self.assertNotEqual(tool_threads[0], main_thread)
        self.assertEqual(loop.initial_messages, messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(result.prompt_ids, PROMPT)
        self.assertEqual(result.response_ids, search_ids + CONTEXT + answer_ids)
        self.assertEqual(result.response_mask, [1, 1, 1, 0, 0, 0, 1, 1, 1])
        self.assertEqual(result.response_logprobs, [-0.5]*3 + [0.0]*3 + [-0.5]*3)
        self.assertEqual(server.calls[1][1], PROMPT + search_ids + CONTEXT)
        self.assertEqual(server.calls[0][0], server.calls[1][0])
        self.assertEqual(server.calls[0][2]["temperature"], 0.8)
        self.assertEqual(server.calls[0][2]["max_tokens"], 512)
        self.assertEqual(params, {"temperature": 0.8, "top_p": 0.9})
        previous, updated = loop.context_messages[0]
        self.assertEqual(previous[-1]["reasoning_content"], "Need evidence.")
        self.assertEqual(previous[-1]["content"], "<search>target query</search>")
        self.assertEqual(updated[-1], {"role": "user", "content": "<information>Paris evidence</information>"})
        self.assertEqual(result.extra_fields, {
            "min_global_steps": 0, "max_global_steps": 2,
            "final_answer": "Paris", "termination_reason": "answer",
            "search_count": 1, "strict_protocol_valid": True,
        })

    async def test_invalid_action_retains_sampled_tokens_without_search(self):
        ids = [301, EOS]
        loop, _ = self.make_loop([ids], {tuple(ids): "<answer>Paris</answer><|im_end|>"})
        with patch.object(agent_loop, "_step_environment") as tool:
            result = await loop.run({}, raw_prompt=[])
        tool.assert_not_called()
        self.assertEqual(result.response_ids, ids)
        self.assertEqual(result.response_mask, [1, 1])
        self.assertEqual(result.extra_fields["termination_reason"], "invalid_action")
        self.assertFalse(result.extra_fields["strict_protocol_valid"])

    async def test_completed_stop_reason_without_eos_is_invalid(self):
        ids = [401, 402]
        loop, _ = self.make_loop([ids], {tuple(ids): "<think>x</think><answer>Paris</answer>"})
        result = await loop.run({}, raw_prompt=[])
        self.assertEqual(result.extra_fields["termination_reason"], "invalid_action")

    async def test_maximum_five_assistant_generations(self):
        ids = [501, EOS]
        loop, server = self.make_loop([ids]*5, {tuple(ids): SEARCH})
        with patch.object(agent_loop, "_step_environment", return_value=SimpleNamespace(
            observation="<information>evidence</information>"
        )) as tool:
            result = await loop.run({}, raw_prompt=[])
        self.assertEqual(len(server.calls), 5)
        self.assertEqual(tool.call_count, 5)
        self.assertEqual(result.extra_fields["termination_reason"], "max_turns")
        self.assertEqual(result.extra_fields["search_count"], 5)
        self.assertEqual(len(result.response_ids), len(result.response_mask))

    async def test_complete_observation_over_budget_is_not_committed(self):
        ids = [601, EOS]
        loop, server = self.make_loop([ids], {tuple(ids): SEARCH})
        loop.response_budget = 4
        with patch.object(agent_loop, "_step_environment", return_value=SimpleNamespace(
            observation="<information>evidence</information>"
        )):
            result = await loop.run({}, raw_prompt=[])
        self.assertEqual(len(server.calls), 1)
        self.assertEqual(result.extra_fields["termination_reason"], "length_limit")
        self.assertEqual(result.response_ids, ids)
        self.assertEqual(result.response_mask, [1, 1])
        self.assertEqual(result.response_logprobs, [-0.5, -0.5])

    async def test_remaining_budget_limits_generation_and_missing_eos(self):
        ids = [701, 702]
        loop, server = self.make_loop([ids], {tuple(ids): "<think>unfinished"})
        loop.response_budget = 2
        result = await loop.run({}, raw_prompt=[])
        self.assertEqual(server.calls[0][2]["max_tokens"], 2)
        self.assertEqual(result.response_ids, ids)
        self.assertEqual(result.extra_fields["termination_reason"], "length_limit")
        self.assertEqual(result.extra_fields["final_answer"], "")

    async def test_total_budget_counts_observation_boundaries(self):
        ids = [801, EOS]
        loop, _ = self.make_loop([ids], {tuple(ids): SEARCH})
        loop.total_budget = len(PROMPT) + len(ids) + len(CONTEXT) - 1
        with patch.object(agent_loop, "_step_environment", return_value=SimpleNamespace(
            observation="<information>evidence</information>"
        )):
            result = await loop.run({}, raw_prompt=[])
        self.assertEqual(result.extra_fields["termination_reason"], "length_limit")
        self.assertEqual(result.response_ids, ids)

    async def test_exactly_full_observation_budget_stops_before_next_generation(self):
        ids = [901, EOS]
        loop, server = self.make_loop([ids], {tuple(ids): SEARCH})
        loop.response_budget = len(ids) + len(CONTEXT)
        with patch.object(agent_loop, "_step_environment", return_value=SimpleNamespace(
            observation="<information>evidence</information>"
        )):
            result = await loop.run({}, raw_prompt=[])
        self.assertEqual(len(server.calls), 1)
        self.assertEqual(result.extra_fields["termination_reason"], "length_limit")
        self.assertEqual(result.response_ids, ids + CONTEXT)

    async def test_initial_prompt_is_never_left_truncated(self):
        loop, server = self.make_loop([], {})
        loop.tokenizer.prompt_ids = [10]*2049
        with self.assertRaisesRegex(ValueError, "not truncated"):
            await loop.run({}, raw_prompt=[])
        self.assertEqual(server.calls, [])


class EnvironmentReuseTests(unittest.TestCase):
    def test_environment_and_retriever_are_reused_in_process(self):
        environment_module = ModuleType("agent.environment")
        tool_module = ModuleType("agent.search_tool")
        retriever_module = ModuleType("retrieval.bm25_retriever")
        environment_module.SearchEnvironment = Mock()
        tool_module.SearchTool = Mock()
        retriever_module.BM25Retriever = Mock()
        agent_loop._environments.clear()
        try:
            with patch.dict("sys.modules", {
                "agent.environment": environment_module,
                "agent.search_tool": tool_module,
                "retrieval.bm25_retriever": retriever_module,
            }):
                agent_loop._step_environment("<search>A</search>", "index", "corpus")
                agent_loop._step_environment("<search>B</search>", "index", "corpus")
            retriever_module.BM25Retriever.assert_called_once_with(
                index_path="index", corpus_path="corpus", top_k=3
            )
            tool_module.SearchTool.assert_called_once()
            environment_module.SearchEnvironment.assert_called_once()
            self.assertEqual(environment_module.SearchEnvironment.return_value.step.call_count, 2)
        finally:
            agent_loop._environments.clear()


if __name__ == "__main__":
    unittest.main()
