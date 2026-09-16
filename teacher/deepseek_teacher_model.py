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

TEACHER_SYSTEM_PROMPT = """You are a search agent that answers questions using an external search environment.

At every turn:

1. First reason inside:
<think>...</think>

2. If more external information is needed, output exactly one:
<search>...</search>

3. The environment will execute the search and provide:
<information>...</information>

4. Never generate <information> yourself.

5. After receiving information, reason again before deciding the next action.

6. When enough evidence is available, output exactly one:
<answer>...</answer>

A turn must end with exactly one action:
either one <search>...</search>
or one <answer>...</answer>.

Do not output multiple searches in the same turn.
Do not output search and answer in the same turn."""


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

    def generate(self, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        content = message.content

        if not content or not content.strip():
            raise RuntimeError("DeepSeek API returned an empty response")

        reasoning_content = getattr(message, "reasoning_content", None) or ""
        model_output = content
        if reasoning_content.strip():
            model_output = (
                f"<think>{reasoning_content.strip()}</think>\n{content}"
            )

        self.raw_turns.append(
            {
                "context": context,
                "reasoning_content": reasoning_content,
                "content": content,
                "model_output": model_output,
            }
        )

        return model_output
