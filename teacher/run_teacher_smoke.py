from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.environment import SearchEnvironment
from agent.loop import AgentLoop
from agent.protocol import parse_action
from agent.tools.search_tool import SearchTool
from retrieval.bm25_retriever import BM25Retriever
from teacher.deepseek_teacher_model import DeepSeekTeacherModel


SMOKE_COUNT = 5
MAX_TURNS = 5
TOP_K = 3
PILOT_PATH = Path(
    "data/processed/teacher/pilot_teacher_candidates.jsonl"
)
OUTPUT_PATH = Path(
    "data/processed/teacher/smoke_teacher_trajectories.jsonl"
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
    trajectory: str,
) -> list[dict[str, str]]:
    events = []

    for index, turn in enumerate(raw_turns):
        reasoning_content = turn["reasoning_content"].strip()
        if reasoning_content:
            events.append({"type": "think", "content": reasoning_content})
        else:
            think_matches = THINK_PATTERN.findall(turn["content"])
            if think_matches:
                events.append(
                    {"type": "think", "content": think_matches[-1].strip()}
                )

        action = parse_action(turn["model_output"])
        if action.type not in {"search", "answer"}:
            continue

        events.append({"type": action.type, "content": action.content})

        if action.type == "search":
            next_context = (
                raw_turns[index + 1]["context"]
                if index + 1 < len(raw_turns)
                else trajectory
            )
            prefix = turn["context"] + turn["model_output"]
            observation = next_context[len(prefix) :]
            information_match = INFORMATION_PATTERN.fullmatch(observation)
            if information_match is None:
                raise ValueError(
                    f"Could not recover environment observation for turn "
                    f"{index + 1}"
                )
            events.append(
                {
                    "type": "information",
                    "content": information_match.group(1),
                }
            )

    return events


def main() -> None:
    if not PILOT_PATH.exists():
        raise FileNotFoundError(f"Pilot file does not exist: {PILOT_PATH}")

    with PILOT_PATH.open("r", encoding="utf-8") as input_file:
        pilot = [json.loads(line) for line in input_file if line.strip()]

    nq_records = [record for record in pilot if record["source"] == "nq"]
    hotpotqa_records = [
        record for record in pilot if record["source"] == "hotpotqa"
    ]
    if len(nq_records) < 3 or len(hotpotqa_records) < 2:
        raise ValueError("Pilot does not contain 3 NQ and 2 HotpotQA records")

    smoke_records = sorted(
        nq_records[:3] + hotpotqa_records[:2],
        key=lambda record: record["uid"],
    )
    if len(smoke_records) != SMOKE_COUNT:
        raise ValueError(f"Expected {SMOKE_COUNT} smoke records")

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
    for record in smoke_records:
        model.raw_turns.clear()
        result = loop.run(f"Question: {record['question']}\n")
        events = extract_events(model.raw_turns, result.trajectory)
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
                "done": result.done,
                "turns": result.turns,
                "search_count": result.search_count,
                "valid_action_count": result.valid_action_count,
                "invalid_action_count": result.invalid_action_count,
                "termination_reason": result.termination_reason,
                "final_answer": result.final_answer,
            },
        }
        outputs.append(output)
        print(
            f"{record['uid']}: done={result.done}, "
            f"turns={result.turns}, searches={result.search_count}, "
            f"termination={result.termination_reason}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for output in outputs:
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} trajectories to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
