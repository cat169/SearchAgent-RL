from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.environment import SearchEnvironment
from agent.loop import AgentLoop, AgentStep
from agent.search_tool import SearchTool
from retrieval.bm25_retriever import BM25Retriever
from teacher.deepseek_teacher import DeepSeekTeacherModel


PILOT_PER_SOURCE = 50
SEED = 123
MAX_TURNS = 5
TOP_K = 3
TEACHER_CANDIDATES_PATH = Path(
    "data/processed/splits/teacher_sft_candidates.jsonl"
)
PILOT_PATH = Path(
    "data/processed/teacher/pilot_teacher_candidates.jsonl"
)
INDEX_PATH = "data/retrieval/wiki-18-bm25-index/bm25"
CORPUS_PATH = "data/retrieval/wiki-18-corpus/wiki-18.jsonl"
THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL,
)
INFORMATION_PATTERN = re.compile(
    r"\s*<information>(.*?)</information>\s*",
    re.DOTALL,
)


def extract_events(
    raw_turns: list[dict[str, str]],
    steps: list[AgentStep],
) -> list[dict[str, str]]:
    events = []

    for turn, step in zip(raw_turns, steps, strict=True):
        reasoning_content = turn["reasoning_content"].strip()
        if reasoning_content:
            events.append({"type": "think", "content": reasoning_content})
        else:
            think_matches = THINK_PATTERN.findall(turn["content"])
            if think_matches:
                events.append(
                    {"type": "think", "content": think_matches[-1].strip()}
                )

        action = step.action
        if action.type not in {"search", "answer"}:
            continue

        events.append({"type": action.type, "content": action.content})

        if action.type == "search":
            information_match = INFORMATION_PATTERN.fullmatch(
                step.observation
            )
            if information_match is None:
                raise ValueError(
                    "Could not unwrap environment observation"
                )
            events.append(
                {
                    "type": "information",
                    "content": information_match.group(1),
                }
            )

    return events


def build_pilot() -> list[dict]:
    with TEACHER_CANDIDATES_PATH.open(
        "r", encoding="utf-8"
    ) as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]

    pilot = []
    for source in ["nq", "hotpotqa"]:
        source_records = sorted(
            (record for record in records if record["source"] == source),
            key=lambda record: record["uid"],
        )
        random.Random(SEED).shuffle(source_records)
        selected = source_records[:PILOT_PER_SOURCE]
        if len(selected) != PILOT_PER_SOURCE:
            raise ValueError(f"Not enough {source} records for pilot")
        pilot.extend(selected)

    pilot.sort(key=lambda record: record["uid"])
    PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_PATH.open("w", encoding="utf-8") as output_file:
        for record in pilot:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, choices=[5, 100], default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = build_pilot()

    output_path = Path(
        "data/processed/teacher/smoke_teacher_trajectories.jsonl"
        if args.limit == 5
        else "data/processed/teacher/pilot_teacher_trajectories.jsonl"
    )

    nq_records = [record for record in pilot if record["source"] == "nq"]
    hotpotqa_records = [
        record for record in pilot if record["source"] == "hotpotqa"
    ]
    if args.limit == 5:
        selected_records = sorted(
            nq_records[:3] + hotpotqa_records[:2],
            key=lambda record: record["uid"],
        )
    else:
        selected_records = pilot

    model = DeepSeekTeacherModel()
    retriever = BM25Retriever(
        index_path=INDEX_PATH,
        corpus_path=CORPUS_PATH,
        top_k=TOP_K,
    )
    environment = SearchEnvironment(
        SearchTool(retriever=retriever, default_top_k=TOP_K)
    )
    loop = AgentLoop(
        model=model,
        environment=environment,
        max_turns=MAX_TURNS,
    )

    outputs = []
    for record in selected_records:
        model.raw_turns.clear()
        result = loop.run(f"Question: {record['question']}\n")
        events = extract_events(model.raw_turns, result.steps)
        event_search_count = sum(
            event["type"] == "search" for event in events
        )
        information_count = sum(
            event["type"] == "information" for event in events
        )
        if event_search_count != result.search_count:
            raise ValueError(
                f"Search count mismatch for {record['uid']}: "
                f"events={event_search_count}, runtime={result.search_count}"
            )
        if information_count != event_search_count:
            raise ValueError(
                f"Information count mismatch for {record['uid']}"
            )

        output = {
            "id": record["uid"],
            "source": record["source"],
            "split": "teacher_sft",
            "question": record["question"],
            "answers": record["gold_answers"],
            "events": events,
            "raw_model_turns": [
                {
                    "reasoning_content": turn["reasoning_content"],
                    "content": turn["content"],
                }
                for turn in model.raw_turns
            ],
            "runtime": {
                "termination_reason": result.termination_reason,
            },
        }
        outputs.append(output)
        print(
            f"{record['uid']}: done={result.done}, "
            f"turns={result.turns}, searches={result.search_count}, "
            f"termination={result.termination_reason}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for output in outputs:
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
