from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teacher.deepseek_teacher import TEACHER_SYSTEM_PROMPT


NARROW_PATH = Path("data/processed/sft/narrow.jsonl")
STANDARD_PATH = Path("data/processed/sft/standard.jsonl")
FIXED_IDS = (
    "hotpotqa_dev_2687",
    "hotpotqa_dev_2870",
    "hotpotqa_dev_1794",
)
PROTOCOL_TAGS = (
    "<think>", "</think>",
    "<search>", "</search>",
    "<information>", "</information>",
    "<answer>", "</answer>",
)


def build_messages(record: dict) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
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


def audit_file(path: Path) -> dict:
    records = 0
    success = 0
    sources = Counter()
    search_counts = Counter()
    search_mismatches = 0
    think_tags = Counter()
    failures = []
    fixed_samples = {}

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            sources[record["source"]] += 1
            search_counts[record["search_count"]] += 1
            if sum(
                event["type"] == "search" for event in record["events"]
            ) != record["search_count"]:
                search_mismatches += 1

            think_contents = [
                event["content"] for event in record["events"]
                if event["type"] == "think"
            ]
            for tag in PROTOCOL_TAGS:
                if any(tag in content for content in think_contents):
                    think_tags[tag] += 1

            try:
                messages = build_messages(record)
            except ValueError as exc:
                failures.append(
                    (record["id"], [e["type"] for e in record["events"]], str(exc))
                )
                continue
            success += 1
            if record["id"] in FIXED_IDS:
                fixed_samples[record["id"]] = messages

    return {
        "records": records,
        "success": success,
        "failures": failures,
        "sources": sources,
        "search_counts": search_counts,
        "search_mismatches": search_mismatches,
        "think_tags": think_tags,
        "fixed_samples": fixed_samples,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    audits = {
        "Narrow": audit_file(NARROW_PATH),
        "Standard": audit_file(STANDARD_PATH),
    }
    for name, audit in audits.items():
        print(f"{name} message audit")
        print(f"  records: {audit['records']}")
        print(f"  successful renders: {audit['success']}")
        print(f"  failed renders: {len(audit['failures'])}")
        print(f"  source: {dict(sorted(audit['sources'].items()))}")
        print(f"  search_count: {dict(sorted(audit['search_counts'].items()))}")
        print(f"  search_count mismatches: {audit['search_mismatches']}")
        print("  think protocol-tag sample counts:")
        for tag in PROTOCOL_TAGS:
            print(f"    {tag}: {audit['think_tags'][tag]}")
        for record_id, pattern, reason in audit["failures"]:
            print(f"  FAILED id={record_id} events={pattern} reason={reason}")
        print()

    for record_id in FIXED_IDS:
        print(f"Fixed sample: {record_id}")
        for name, audit in audits.items():
            messages = audit["fixed_samples"].get(record_id)
            if messages is not None:
                print(f"  split: {name}")
                print(json.dumps(messages, ensure_ascii=False, indent=2))
                break
        else:
            print("  not present or render failed")


if __name__ == "__main__":
    main()
