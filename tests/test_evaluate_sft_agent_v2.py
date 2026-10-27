import unittest

from agent.system_prompt import AGENT_SYSTEM_PROMPT
from scripts.evaluate_sft_agent_v2 import (
    IM_END_TOKEN,
    assistant_message,
    check_history_prefix,
    completed_turn_matches_template,
    parse_raw_generation,
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


if __name__ == "__main__":
    unittest.main()
