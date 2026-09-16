import re
from dataclasses import dataclass
from typing import Literal


ActionType = Literal["search", "answer", "invalid"]


@dataclass
class AgentAction:
    type: ActionType
    content: str


ACTION_PATTERN = re.compile(
    r"<(search|answer)>(.*?)</\1>",
    re.DOTALL,
)


def parse_action(text: str) -> AgentAction:
    if "<information>" in text or "</information>" in text:
        return AgentAction(
            type="invalid",
            content="",
        )

    matches = ACTION_PATTERN.findall(text)

    if len(matches) != 1:
        return AgentAction(
            type="invalid",
            content="",
        )

    action_type, content = matches[0]
    content = content.strip()

    if not content:
        return AgentAction(
            type="invalid",
            content="",
        )

    return AgentAction(
        type=action_type,
        content=content,
    )
