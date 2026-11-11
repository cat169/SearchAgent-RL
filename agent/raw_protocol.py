"""Strict raw Assistant protocol shared by Evaluation and GRPO."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.protocol import AgentAction


IM_END_TOKEN = "<|im_end|>"


ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


STRICT_RAW_PATTERN = re.compile(
    r"<think>(.*?)</think>\s*<(search|answer)>(.*?)</\2><\|im_end\|>",
    re.DOTALL,
)


RESERVED_TAG_PATTERN = re.compile(
    r"</?(?:think|search|answer|information)>|<\|(?:im_end|endoftext)\|>"
)


@dataclass(frozen=True)
class RawProtocolResult:
    strict_valid: bool
    valid_action: bool
    ended_with_im_end: bool
    reasoning_content: str
    action: AgentAction
    error: str | None


def _action_candidate(raw_text: str) -> tuple[bool, AgentAction]:
    """Find one usable action for diagnostics without authorizing its execution."""
    if "<information>" in raw_text or "</information>" in raw_text:
        return False, AgentAction(type="invalid", content="")
    matches = ACTION_PATTERN.findall(raw_text)
    if len(matches) != 1:
        return False, AgentAction(type="invalid", content="")
    action_type, content = matches[0]
    content = content.strip()
    if not content or RESERVED_TAG_PATTERN.search(content):
        return False, AgentAction(type="invalid", content="")
    return True, AgentAction(type=action_type, content=content)


def parse_raw_generation(
    raw_text: str,
    token_ids: list[int],
    im_end_token_id: int,
) -> RawProtocolResult:
    """Validate the exact SFT raw response, including its terminal token."""
    valid_action, candidate = _action_candidate(raw_text)
    ended_with_im_end = bool(token_ids) and token_ids[-1] == im_end_token_id

    if not ended_with_im_end:
        return RawProtocolResult(
            False, valid_action, False, "", AgentAction("invalid", ""), "missing_im_end"
        )
    if "\ufffd" in raw_text:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "decode_error"
        )
    if raw_text.count("<think>") != 1 or raw_text.count("</think>") != 1:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "invalid_think_count"
        )
    if "<information>" in raw_text or "</information>" in raw_text:
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "forged_information"
        )

    match = STRICT_RAW_PATTERN.fullmatch(raw_text)
    if match is None:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "raw_protocol_mismatch"
        )

    reasoning_content, action_type, action_content = match.groups()
    reasoning_content = reasoning_content.strip()
    action_content = action_content.strip()
    if not reasoning_content:
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "empty_think"
        )
    if not action_content:
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "empty_action"
        )
    if RESERVED_TAG_PATTERN.search(reasoning_content):
        return RawProtocolResult(
            False, valid_action, True, "", AgentAction("invalid", ""), "reserved_tag_in_think"
        )
    if RESERVED_TAG_PATTERN.search(action_content):
        return RawProtocolResult(
            False, False, True, "", AgentAction("invalid", ""), "reserved_tag_in_action"
        )

    return RawProtocolResult(
        strict_valid=True,
        valid_action=True,
        ended_with_im_end=True,
        reasoning_content=reasoning_content,
        action=AgentAction(type=action_type, content=action_content),
        error=None,
    )


def assistant_message(protocol: RawProtocolResult) -> dict[str, str]:
    if not protocol.strict_valid:
        raise ValueError("Cannot add an invalid generation to Agent history")
    action = protocol.action
    return {
        "role": "assistant",
        "reasoning_content": protocol.reasoning_content,
        "content": f"<{action.type}>{action.content}</{action.type}>",
    }
