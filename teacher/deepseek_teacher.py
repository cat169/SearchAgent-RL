from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

TEACHER_SYSTEM_PROMPT = """You are a search agent that answers knowledge-intensive questions using an external search environment.

Your reasoning is handled internally by the model's thinking mode.
Do not output your reasoning or any <think> tags in the visible response.

At each turn, inspect the question and all search results currently available in the context.

If the available information is insufficient, output exactly one search action:

<search>your search query</search>

The environment will execute the query and append the retrieved evidence as:

<information>
retrieved evidence
</information>

After receiving new <information>, reconsider the question using the new evidence.
You may perform another search if important information is still missing.
You may search multiple times across different turns.

When the available evidence is sufficient, output exactly one final answer:

<answer>your final answer</answer>

Rules:
- Each visible response must contain exactly one action.
- Output either one <search>...</search> or one <answer>...</answer>.
- Never output both in the same turn.
- Never output multiple searches in one turn.
- Never output <information> yourself.
- Never invent or simulate search results.
- Never output <think> tags.
- Keep search queries concise and targeted.
- Keep the final answer concise and directly responsive to the question.
- After the closing </search> or </answer> tag, output nothing else.

Example of a multi-turn interaction:

Question:
Which country was the author of The Little Prince born in?

First model response:
<search>The Little Prince author</search>

The environment returns:
<information>
The Little Prince was written by Antoine de Saint-Exupéry.
</information>

The next model response:
<search>Antoine de Saint-Exupéry birthplace country</search>

The environment returns:
<information>
Antoine de Saint-Exupéry was born in Lyon, France.
</information>

The final model response:
<answer>France</answer>

Notice:
- The first search only identifies the author.
- The evidence is not yet sufficient to answer the original question.
- A second search is therefore necessary.
- Only after sufficient evidence is retrieved should the final answer be produced.
"""


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
            raise RuntimeError(
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
