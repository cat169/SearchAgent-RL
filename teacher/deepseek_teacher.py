from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.system_prompt import AGENT_SYSTEM_PROMPT


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

TEACHER_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT


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
