from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.system_prompt import AGENT_SYSTEM_PROMPT


def build_messages(record: dict) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
    ]
    pending_reasoning = None
    awaiting_information = False
    answered = False

    for event in record["events"]:
        event_type = event["type"]
        content = event["content"]

        if event_type == "think":
            if pending_reasoning is not None or awaiting_information or answered:
                raise ValueError("think is out of order")
            pending_reasoning = content
        elif event_type in {"search", "answer"}:
            if pending_reasoning is None or awaiting_information or answered:
                raise ValueError(f"{event_type} lacks preceding think or is out of order")
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": pending_reasoning,
                    "content": f"<{event_type}>{content}</{event_type}>",
                }
            )
            pending_reasoning = None
            if event_type == "search":
                awaiting_information = True
            else:
                answered = True
        elif event_type == "information":
            if not awaiting_information:
                raise ValueError("information lacks preceding search")
            messages.append(
                {
                    "role": "user",
                    "content": f"<information>{content}</information>",
                }
            )
            awaiting_information = False
        else:
            raise ValueError(f"unsupported event type: {event_type}")

    if not answered or pending_reasoning is not None or awaiting_information:
        raise ValueError("trajectory does not end with think and answer")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical SFT records as conversational messages JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input.open("r", encoding="utf-8") as input_file, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if sum(event["type"] == "search" for event in record["events"]) != record["search_count"]:
                raise ValueError(f"Search count mismatch for {record['id']}")
            output_file.write(
                json.dumps({"messages": build_messages(record)}, ensure_ascii=False) + "\n"
            )
            count += 1

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Records: {count}")


if __name__ == "__main__":
    main()
