import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.protocol import AgentAction
from agent.system_prompt import AGENT_SYSTEM_PROMPT
from scripts.evaluate_sft_agent_v2 import (
    GenerationResult,
    IM_END_TOKEN,
    InputTooLongError,
    aggregate,
    assistant_message,
    check_history_prefix,
    completed_turn_matches_template,
    evaluate_records,
    load_resume_results,
    parse_raw_generation,
    prepare_results,
)
from scripts.render_sft_messages import build_messages


IM_END_ID = 151645


class FakeTrainingTokenizer:
    """Minimal prefix-preserving stand-in for the saved Qwen training template."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("These history tests inspect rendered text")
        parts = []
        for message in messages:
            if message["role"] == "assistant":
                parts.append(
                    "<|im_start|>assistant\n"
                    f"<think>{message['reasoning_content']}</think>"
                    f"{message['content']}<|im_end|>\n"
                )
            else:
                parts.append(
                    f"<|im_start|>{message['role']}\n"
                    f"{message['content']}<|im_end|>\n"
                )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def parse(text: str, final_token: int = IM_END_ID):
    return parse_raw_generation(text, [42, final_token], IM_END_ID)


class RawProtocolTests(unittest.TestCase):
    def test_valid_thinking_search_eos(self):
        result = parse("<think>Need evidence.</think><search>target query</search><|im_end|>")
        self.assertTrue(result.strict_valid)
        self.assertEqual(result.action.type, "search")

    def test_valid_thinking_answer_eos(self):
        result = parse("<think>Evidence is sufficient.</think><answer>Paris</answer><|im_end|>")
        self.assertTrue(result.strict_valid)
        self.assertEqual(result.action.content, "Paris")

    def test_missing_thinking(self):
        self.assertFalse(parse("<answer>Paris</answer><|im_end|>").strict_valid)

    def test_unclosed_thinking(self):
        self.assertFalse(
            parse("<think>broken<answer>Paris</answer><|im_end|>").strict_valid
        )

    def test_duplicate_search(self):
        result = parse(
            "<think>x</think><search>A</search><search>B</search><|im_end|>"
        )
        self.assertFalse(result.strict_valid)
        self.assertFalse(result.valid_action)

    def test_search_then_answer(self):
        result = parse(
            "<think>x</think><search>A</search><answer>B</answer><|im_end|>"
        )
        self.assertFalse(result.strict_valid)
        self.assertFalse(result.valid_action)

    def test_missing_eos(self):
        result = parse("<think>x</think><answer>B</answer>", final_token=99)
        self.assertFalse(result.strict_valid)
        self.assertEqual(result.error, "missing_im_end")

    def test_text_after_action(self):
        result = parse("<think>x</think><answer>B</answer>garbage<|im_end|>")
        self.assertFalse(result.strict_valid)
        self.assertTrue(result.valid_action)

    def test_forged_information(self):
        result = parse(
            "<think>x</think><information>fake</information>"
            "<answer>B</answer><|im_end|>"
        )
        self.assertFalse(result.strict_valid)
        self.assertEqual(result.error, "forged_information")

    def test_action_nested_in_thinking(self):
        result = parse(
            "<think>try <search>A</search></think>"
            "<answer>B</answer><|im_end|>"
        )
        self.assertFalse(result.strict_valid)
        self.assertEqual(result.error, "reserved_tag_in_think")


class HistoryRenderingTests(unittest.TestCase):
    def test_multiturn_history_matches_training_messages_and_rendering(self):
        tokenizer = FakeTrainingTokenizer()
        record = {
            "question": "Where was the author born?",
            "events": [
                {"type": "think", "content": "Find the author."},
                {"type": "search", "content": "book author"},
                {"type": "information", "content": "The author is Ada."},
                {"type": "think", "content": "Find Ada's birthplace."},
                {"type": "search", "content": "Ada birthplace"},
                {"type": "information", "content": "Ada was born in London."},
                {"type": "think", "content": "The evidence is enough."},
                {"type": "answer", "content": "London"},
            ],
        }
        training_messages = build_messages(record)

        runtime_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ]
        first = parse(
            "<think>Find the author.</think><search>book author</search><|im_end|>"
        )
        _, inference_prompt = check_history_prefix(tokenizer, runtime_messages)
        self.assertTrue(
            completed_turn_matches_template(
                tokenizer,
                runtime_messages,
                inference_prompt,
                first,
                first_raw := "<think>Find the author.</think><search>book author</search><|im_end|>",
            )
        )
        self.assertTrue(first_raw.endswith(IM_END_TOKEN))
        runtime_messages.append(assistant_message(first))
        runtime_messages.append(
            {"role": "user", "content": "<information>The author is Ada.</information>"}
        )

        second = parse(
            "<think>Find Ada's birthplace.</think>"
            "<search>Ada birthplace</search><|im_end|>"
        )
        runtime_messages.append(assistant_message(second))
        runtime_messages.append(
            {"role": "user", "content": "<information>Ada was born in London.</information>"}
        )
        final = parse(
            "<think>The evidence is enough.</think>"
            "<answer>London</answer><|im_end|>"
        )
        runtime_messages.append(assistant_message(final))

        self.assertEqual(runtime_messages, training_messages)
        self.assertEqual(
            tokenizer.apply_chat_template(runtime_messages, tokenize=False, add_generation_prompt=False),
            tokenizer.apply_chat_template(training_messages, tokenize=False, add_generation_prompt=False),
        )


def generation(raw_text: str) -> GenerationResult:
    protocol = parse_raw_generation(raw_text, [42, IM_END_ID], IM_END_ID)
    return GenerationResult(
        raw_text=raw_text,
        token_ids=[42, IM_END_ID],
        input_tokens=100,
        final_token_id=IM_END_ID,
        reached_max_new_tokens=False,
        protocol=protocol,
        history_prefix_preserved=True,
        completed_turn_matches_template=True,
    )


class ScenarioModel:
    def generate_turn(self, messages):
        question = messages[1]["content"]
        assistant_turns = sum(message["role"] == "assistant" for message in messages)
        if question == "long question":
            if assistant_turns == 0:
                return generation(
                    "<think>Search first.</think><search>long query</search><|im_end|>"
                )
            raise InputTooLongError(4174, 4096)
        return generation(
            "<think>I know it.</think><answer>correct</answer><|im_end|>"
        )


class FakeEnvironment:
    def step(self, action_text):
        if action_text == "<search>long query</search>":
            return SimpleNamespace(
                action=AgentAction("search", "long query"),
                observation="\n\n<information>large result</information>\n\n",
            )
        return SimpleNamespace(action=AgentAction("answer", "correct"), observation="")


def eval_record(index: int, question: str = "question") -> dict:
    return {
        "id": f"sample_{index:03d}",
        "source": "nq",
        "question": question,
        "answers": ["correct"],
    }


def completed_result(record: dict, termination_reason: str = "answer") -> dict:
    return {
        "id": record["id"],
        "source": record["source"],
        "question": record["question"],
        "gold_answers": record["answers"],
        "predicted_answer": "correct" if termination_reason == "answer" else "",
        "termination_reason": termination_reason,
        "search_count": 0,
        "total_turns": 1,
        "valid_action_count": 1,
        "strict_protocol_count": 1,
        "eos_count": 1,
        "turns": [],
        "em": 1.0 if termination_reason == "answer" else 0.0,
        "f1": 1.0 if termination_reason == "answer" else 0.0,
    }


class InputLengthAndResumeTests(unittest.TestCase):
    def test_input_too_long_is_saved_and_next_question_continues(self):
        records = [eval_record(1, "long question"), eval_record(2, "normal question")]
        output_file = io.StringIO()
        outputs = evaluate_records(
            records,
            ScenarioModel(),
            FakeEnvironment(),
            output_file,
            start_index=0,
            total_records=2,
        )

        self.assertEqual(len(outputs), 2)
        long_output = outputs[0]
        self.assertEqual(long_output["termination_reason"], "input_too_long")
        self.assertEqual(long_output["input_tokens_at_failure"], 4174)
        self.assertEqual(long_output["max_input_tokens"], 4096)
        self.assertEqual((long_output["em"], long_output["f1"]), (0.0, 0.0))
        self.assertEqual(long_output["total_turns"], 1)
        self.assertEqual(long_output["turns"][0]["protocol"]["action"]["type"], "search")
        self.assertEqual(outputs[1]["termination_reason"], "answer")

    def test_resume_131_results_leaves_69_records(self):
        records = [eval_record(index) for index in range(200)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                "".join(
                    json.dumps(completed_result(record)) + "\n"
                    for record in records[:131]
                ),
                encoding="utf-8",
            )
            completed = load_resume_results(path, records)
        self.assertEqual(len(completed), 131)
        self.assertEqual(len(records[len(completed):]), 69)

    def test_resume_rejects_duplicate_id(self):
        records = [eval_record(1), eval_record(2)]
        duplicate = completed_result(records[0])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate result ID"):
                load_resume_results(path, records)

    def test_resume_rejects_dataset_order_mismatch(self):
        records = [eval_record(1), eval_record(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                json.dumps(completed_result(records[1])) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not a continuous Eval prefix"):
                load_resume_results(path, records)

    def test_resumed_aggregate_includes_all_200(self):
        records = [eval_record(index) for index in range(200)]
        results = [completed_result(record) for record in records]
        results[131] = completed_result(records[131], "input_too_long")
        metrics = aggregate(results)
        self.assertEqual(metrics["samples"], 200)
        self.assertEqual(metrics["input_too_long_count"], 1)
        self.assertEqual(metrics["input_too_long_rate"], 1 / 200)

    def test_without_resume_refuses_existing_output(self):
        records = [eval_record(1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "results.jsonl").write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_results(output_dir, False, records)


if __name__ == "__main__":
    unittest.main()
