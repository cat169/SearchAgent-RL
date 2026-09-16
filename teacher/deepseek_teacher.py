from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

TEACHER_SYSTEM_PROMPT = """You are a search agent that answers knowledge-intensive questions using an external search environment.

Your reasoning is handled internally by the model's thinking mode.
Do not output your reasoning or any <think> tags in the visible response.

At every turn, make exactly one decision based only on:
- the original question, and
- any <information>...</information> messages already provided by the environment.

Your visible response must contain exactly ONE action.

If more evidence is needed, output exactly:

<search>concise search query</search>

If the available evidence is sufficient to determine the answer, output exactly:

<answer>short answer</answer>

After emitting either action, STOP immediately.

Search interaction rules:

- The search environment is external to you.
- When you output <search>...</search>, your current turn is finished.
- The environment will execute the search and may provide retrieved evidence in a later user message as <information>...</information>.
- You must wait for that user message before taking another action.
- Never generate <information> yourself.
- Never invent or simulate search results.
- Never output another <search> after a search action in the same response.
- Never output an <answer> after a search action in the same response.
- Never output more than one action in a response.

A valid search response contains only:

<search>...</search>

and nothing else.

When to stop searching:

Search only when information required to answer the question is genuinely missing.

If the question can already be answered confidently from the available question and retrieved evidence, answer immediately.

Do not perform additional searches merely to:
- double-check an answer that is already sufficiently supported,
- seek redundant confirmation,
- collect more evidence after the answer is already determined.

Prefer answering once the evidence is sufficient rather than repeatedly searching.

Final answer format:

The final answer must be the shortest answer span that directly answers the question.

Do not include:
- explanations,
- reasoning,
- supporting evidence,
- introductory phrases,
- full sentences when a name, place, date, number, yes/no, or short phrase is sufficient,
- parenthetical details,
- unnecessary qualifications.

Valid final answers include:

<answer>Steve Jobs</answer>

<answer>United States</answer>

<answer>No</answer>

<answer>genus</answer>

Invalid:

<answer>Steve Jobs, who was the co-founder and former CEO of Apple.</answer>

Invalid:

<answer>The answer is the United States.</answer>

Output constraints:

Every visible response must match exactly one of these two forms:

<search>...</search>

OR

<answer>...</answer>

Nothing may appear before or after the action.

Never output <information>.
Never output <think>.
Never output multiple actions.
"""




class EmptyVisibleContentError(RuntimeError):
    pass


class DeepSeekTeacherModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.raw_turns: list[dict[str, str]] = []
        self.client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        api_messages = [
            {
                "role": "system",
                "content": TEACHER_SYSTEM_PROMPT,
            },
            *messages,
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort="high",
            extra_body={
                "thinking": {
                    "type": "enabled",
                }
            },
        )
        message = response.choices[0].message
        reasoning_content = (
            getattr(message, "reasoning_content", None) or ""
        ).strip()
        content = (message.content or "").strip()

        if not content:
            finish_reason = getattr(
                response.choices[0], "finish_reason", None
            )
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(
                usage, "completion_tokens", None
            )
            completion_details = getattr(
                usage, "completion_tokens_details", None
            )
            reasoning_tokens = getattr(
                completion_details, "reasoning_tokens", None
            )
            raise EmptyVisibleContentError(
                "DeepSeek API returned empty visible content: "
                f"finish_reason={finish_reason}, "
                f"reasoning_chars={len(reasoning_content)}, "
                f"completion_tokens={completion_tokens}, "
                f"reasoning_tokens={reasoning_tokens}"
            )

        self.raw_turns.append(
            {
                "reasoning_content": reasoning_content,
                "content": content,
            }
        )

        return content
