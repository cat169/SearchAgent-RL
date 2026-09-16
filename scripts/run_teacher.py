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
from teacher.deepseek_teacher import (
    DeepSeekTeacherModel,
    EmptyVisibleContentError,
)


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
    parser.add_argument(
        "--limit", type=int, choices=[5, 100, 4000], default=5
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_completed_ids(output_path: Path) -> set[str]:
    completed_ids = set()

    with output_path.open("r", encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue
            completed_id = json.loads(line)["id"]
            if completed_id in completed_ids:
                raise ValueError(
                    f"Duplicate id in resume file: {completed_id}"
                )
            completed_ids.add(completed_id)

    return completed_ids


def load_empty_skipped_ids(skipped_path: Path) -> set[str]:
    skipped_ids = set()

    with skipped_path.open("r", encoding="utf-8") as skipped_file:
        for line in skipped_file:
            skipped_id = line.strip()
            if not skipped_id:
                continue
            if skipped_id in skipped_ids:
                raise ValueError(
                    f"Duplicate id in empty skipped file: {skipped_id}"
                )
            skipped_ids.add(skipped_id)

    return skipped_ids


def main() -> None:
    args = parse_args()

    if args.limit == 4000:
        with TEACHER_CANDIDATES_PATH.open(
            "r", encoding="utf-8"
        ) as input_file:
            selected_records = [
                json.loads(line) for line in input_file if line.strip()
            ]
        nq_count = sum(
            record["source"] == "nq" for record in selected_records
        )
        hotpotqa_count = sum(
            record["source"] == "hotpotqa"
            for record in selected_records
        )
        if (
            len(selected_records) != 4000
            or nq_count != 2000
            or hotpotqa_count != 2000
        ):
            raise ValueError(
                "Expected 4000 Teacher candidates: "
                "NQ=2000 and HotpotQA=2000"
            )
    else:
        pilot = build_pilot()
        nq_records = [
            record for record in pilot if record["source"] == "nq"
        ]
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

    output_paths = {
        5: Path(
            "data/processed/teacher/smoke_teacher_trajectories.jsonl"
        ),
        100: Path(
            "data/processed/teacher/pilot_teacher_trajectories.jsonl"
        ),
        4000: Path(
            "data/processed/teacher/raw_teacher_trajectories.jsonl"
        ),
    }
    skipped_paths = {
        5: Path(
            "data/processed/teacher/"
            "smoke_teacher_empty_skipped_ids.txt"
        ),
        100: Path(
            "data/processed/teacher/"
            "pilot_teacher_empty_skipped_ids.txt"
        ),
        4000: Path(
            "data/processed/teacher/"
            "raw_teacher_empty_skipped_ids.txt"
        ),
    }
    output_path = output_paths[args.limit]
    skipped_path = skipped_paths[args.limit]

    completed_ids = set()
    empty_skipped_ids = set()
    if args.resume and output_path.exists():
        completed_ids = load_completed_ids(output_path)
    if args.resume and skipped_path.exists():
        empty_skipped_ids = load_empty_skipped_ids(skipped_path)
    conflicting_ids = completed_ids & empty_skipped_ids
    if conflicting_ids:
        raise ValueError(
            "Ids exist in both trajectory and empty skipped files: "
            f"{sorted(conflicting_ids)}"
        )
    processed_ids = completed_ids | empty_skipped_ids

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume else "w"
    written_count = 0
    new_empty_skipped_count = 0

    with output_path.open(
        output_mode, encoding="utf-8"
    ) as output_file, skipped_path.open(
        output_mode, encoding="utf-8"
    ) as skipped_file:
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

        for record in selected_records:
            if record["uid"] in processed_ids:
                continue

            model.raw_turns.clear()
            try:
                result = loop.run(f"Question: {record['question']}\n")
            except EmptyVisibleContentError:
                skipped_file.write(record["uid"] + "\n")
                skipped_file.flush()
                new_empty_skipped_count += 1
                print(
                    f"{record['uid']}: skipped, "
                    "reason=empty_visible_content"
                )
                continue

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
                    f"events={event_search_count}, "
                    f"runtime={result.search_count}"
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
            output_file.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )
            output_file.flush()
            written_count += 1

            print(
                f"{record['uid']}: done={result.done}, "
                f"turns={result.turns}, searches={result.search_count}, "
                f"termination={result.termination_reason}"
            )

    print(f"Selected: {len(selected_records)}")
    print(f"Existing trajectories: {len(completed_ids)}")
    print(f"Existing empty skipped: {len(empty_skipped_ids)}")
    print(f"Saved trajectories: {written_count}")
    print(f"Skipped empty responses: {new_empty_skipped_count}")


if __name__ == "__main__":
    main()
