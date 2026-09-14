"""Read-only audit of muradil211/AetherSearch_SFT."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DATASET_NAME = "muradil211/AetherSearch_SFT"
EXPECTED_FIELDS = {
    "id",
    "question",
    "trajectory_type",
    "search_count",
    "full_trajectory_text",
}
LOCAL_DATASETS = {
    "NQ train": Path("data/processed/nq/train.jsonl"),
    "NQ dev": Path("data/processed/nq/dev.jsonl"),
    "NQ test": Path("data/processed/nq/test.jsonl"),
    "HotpotQA train": Path("data/processed/hotpotqa/train.jsonl"),
    "HotpotQA dev": Path("data/processed/hotpotqa/dev.jsonl"),
}
ID_PATTERN = re.compile(r"^\d{6}$")
ROLE_PATTERN = re.compile(
    r"<\|im_start\|>([^\r\n<]+)\r?\n?(.*?)<\|im_end\|>", re.DOTALL
)
ROLE_MARKER_PATTERN = re.compile(r"<\|im_start\|>([^\r\n<]+)")
CONTROL_TAGS = ("think", "search", "information", "answer")
MAX_ANOMALY_EXAMPLES = 10
MAX_MISMATCH_EXAMPLES = 20
MAX_OVERLAP_EXAMPLES = 20


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def tag_pattern(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)


def describe_lengths(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(values)

    def percentile(percent: float) -> int:
        return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]

    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def duplicate_summary(counter: Counter[str]) -> dict[str, int]:
    duplicates = [count for count in counter.values() if count > 1]
    return {
        "duplicate_values": len(duplicates),
        "rows_in_duplicate_groups": sum(duplicates),
        "duplicate_occurrences_beyond_first": sum(count - 1 for count in duplicates),
    }


def compact_snippet(text: Any, limit: int = 500) -> str:
    if not isinstance(text, str):
        return repr(text)
    if len(text) <= limit:
        return repr(text)
    half = limit // 2
    return repr(text[:half] + " ... <snip> ... " + text[-half:])


def add_anomaly(
    counts: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    sample: dict[str, Any],
    row_index: int,
    snippet: Any,
) -> None:
    counts[category] += 1
    if len(examples[category]) < MAX_ANOMALY_EXAMPLES:
        examples[category].append(
            {
                "row_index": row_index,
                "id": sample.get("id"),
                "question": sample.get("question"),
                "trajectory_type": sample.get("trajectory_type"),
                "search_count": sample.get("search_count"),
                "trajectory_snippet": compact_snippet(snippet),
            }
        )


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dataset = load_dataset(DATASET_NAME)
    print("=== BASIC STRUCTURE ===")
    print(dataset)
    print(f"split_names={list(dataset.keys())!r}")
    for split, split_dataset in dataset.items():
        print(f"\n[{split}] rows={len(split_dataset)}")
        print(f"features={split_dataset.features!r}")
        print(f"column_names={split_dataset.column_names!r}")
        first = split_dataset[0] if len(split_dataset) else None
        actual_types = (
            {key: type(value).__name__ for key, value in first.items()}
            if isinstance(first, dict)
            else None
        )
        print(f"first_sample_python_types={actual_types!r}")
        print("first_complete_raw_sample=")
        print_json(first)

    flat_samples: list[tuple[str, int, dict[str, Any]]] = []
    for split, split_dataset in dataset.items():
        for row_index, sample in enumerate(split_dataset):
            flat_samples.append((split, row_index, sample))

    print("\n=== TRAJECTORY TEXT SAMPLES ===")
    if flat_samples:
        first_text = flat_samples[0][2].get("full_trajectory_text")
        print("first_full_trajectory_text=")
        print(first_text)
    for sample_number in (2, 3):
        if len(flat_samples) >= sample_number:
            text = flat_samples[sample_number - 1][2].get("full_trajectory_text")
            print(f"\ntrajectory_{sample_number}_first_500={text[:500]!r}")
            print(f"trajectory_{sample_number}_last_500={text[-500:]!r}")

    quality_by_split: dict[str, dict[str, Any]] = {}
    field_python_types: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    global_ids: Counter[str] = Counter()
    raw_questions: Counter[str] = Counter()
    normalized_questions: Counter[str] = Counter()
    trajectories: Counter[str] = Counter()
    valid_numeric_ids: set[int] = set()
    invalid_id_format_examples: list[dict[str, Any]] = []
    type_distribution: Counter[str] = Counter()
    search_count_distribution: Counter[str] = Counter()
    type_search_cross: Counter[str] = Counter()
    relationship_mismatches: list[dict[str, Any]] = []
    relationship_mismatch_count = 0

    role_marker_distribution: Counter[str] = Counter()
    role_order_distribution: Counter[str] = Counter()
    im_start_count_distribution: Counter[int] = Counter()
    im_end_count_distribution: Counter[int] = Counter()
    tag_presence: Counter[str] = Counter()
    system_prompts: Counter[str] = Counter()
    system_identity_distribution: Counter[str] = Counter()
    system_prompt_checks: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    anomaly_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_lengths: list[int] = []
    query_lengths: list[int] = []
    final_answer_lengths: list[int] = []
    longest_trajectories: list[dict[str, Any]] = []
    empty_query_count = 0
    query_over_500_count = 0
    final_answer_extraction_failures = 0
    parsed_message_trajectories = 0

    aether_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split, split_dataset in dataset.items():
        quality = Counter(
            {
                "total_rows": 0,
                "non_string_id": 0,
                "empty_id": 0,
                "non_string_question": 0,
                "empty_question": 0,
                "non_string_trajectory_type": 0,
                "empty_trajectory_type": 0,
                "non_integer_search_count": 0,
                "non_string_full_trajectory_text": 0,
                "empty_full_trajectory_text": 0,
            }
        )
        split_ids: Counter[str] = Counter()
        split_raw_questions: Counter[str] = Counter()
        split_normalized_questions: Counter[str] = Counter()
        split_trajectories: Counter[str] = Counter()

        for row_index, sample in enumerate(split_dataset):
            quality["total_rows"] += 1
            for field, value in sample.items():
                field_python_types[split][field][type(value).__name__] += 1

            sample_id = sample.get("id")
            question = sample.get("question")
            trajectory_type = sample.get("trajectory_type")
            search_count = sample.get("search_count")
            text = sample.get("full_trajectory_text")

            if not isinstance(sample_id, str):
                quality["non_string_id"] += 1
            elif not sample_id.strip():
                quality["empty_id"] += 1
            else:
                split_ids[sample_id] += 1
                global_ids[sample_id] += 1
                if ID_PATTERN.fullmatch(sample_id):
                    valid_numeric_ids.add(int(sample_id))
                elif len(invalid_id_format_examples) < 20:
                    invalid_id_format_examples.append(
                        {"split": split, "row_index": row_index, "id": sample_id}
                    )

            if not isinstance(question, str):
                quality["non_string_question"] += 1
            elif not question.strip():
                quality["empty_question"] += 1
            else:
                normalized = normalize_question(question)
                split_raw_questions[question] += 1
                split_normalized_questions[normalized] += 1
                raw_questions[question] += 1
                normalized_questions[normalized] += 1
                aether_questions[normalized].append(
                    {"id": sample_id, "question": question, "row_index": row_index}
                )

            if not isinstance(trajectory_type, str):
                quality["non_string_trajectory_type"] += 1
                type_key = f"<{type(trajectory_type).__name__}>"
            elif not trajectory_type.strip():
                quality["empty_trajectory_type"] += 1
                type_key = "<empty>"
            else:
                type_key = trajectory_type
            type_distribution[type_key] += 1

            if type(search_count) is not int:
                quality["non_integer_search_count"] += 1
                count_key = f"<{type(search_count).__name__}> {search_count!r}"
            else:
                count_key = str(search_count)
            search_count_distribution[count_key] += 1
            type_search_cross[f"{type_key} | {count_key}"] += 1

            relationship_reason = None
            if trajectory_type == "single_search" and search_count != 1:
                relationship_reason = "single_search requires search_count == 1"
            elif trajectory_type == "multi_search" and (
                type(search_count) is not int or search_count < 2
            ):
                relationship_reason = "multi_search requires search_count >= 2"
            elif type(search_count) is int and not 1 <= search_count <= 4:
                relationship_reason = "search_count is outside [1, 4]"
            if relationship_reason:
                relationship_mismatch_count += 1
                if len(relationship_mismatches) < MAX_MISMATCH_EXAMPLES:
                    relationship_mismatches.append(
                        {
                            "split": split,
                            "row_index": row_index,
                            "id": sample_id,
                            "trajectory_type": trajectory_type,
                            "search_count": search_count,
                            "reason": relationship_reason,
                        }
                    )

            if not isinstance(text, str):
                quality["non_string_full_trajectory_text"] += 1
                continue
            if not text.strip():
                quality["empty_full_trajectory_text"] += 1
                continue

            split_trajectories[text] += 1
            trajectories[text] += 1
            trajectory_lengths.append(len(text))
            longest_trajectories.append(
                {
                    "id": sample_id,
                    "question": question,
                    "trajectory_type": trajectory_type,
                    "search_count": search_count,
                    "character_length": len(text),
                }
            )

            marker_roles = ROLE_MARKER_PATTERN.findall(text)
            role_marker_distribution.update(role.strip() for role in marker_roles)
            im_start_count = text.count("<|im_start|>")
            im_end_count = text.count("<|im_end|>")
            im_start_count_distribution[im_start_count] += 1
            im_end_count_distribution[im_end_count] += 1

            role_matches = list(ROLE_PATTERN.finditer(text))
            roles = tuple(match.group(1).strip() for match in role_matches)
            role_order_distribution[" -> ".join(roles)] += 1
            known_roles = {"system", "user", "assistant"}
            separators = ROLE_PATTERN.sub("", text)
            message_parse_ok = (
                bool(role_matches)
                and not separators.strip()
                and set(roles) <= known_roles
                and im_start_count == im_end_count == len(role_matches)
            )
            if message_parse_ok:
                parsed_message_trajectories += 1
            else:
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "messages_not_stably_parseable",
                    sample,
                    row_index,
                    text,
                )

            if not text.startswith("<|im_start|>"):
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "illegal_start_marker",
                    sample,
                    row_index,
                    text[:500],
                )
            if not text.endswith("<|im_end|>"):
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "illegal_end_marker",
                    sample,
                    row_index,
                    text[-500:],
                )
            if im_start_count != im_end_count or im_start_count != len(role_matches):
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "im_marker_count_mismatch",
                    sample,
                    row_index,
                    {
                        "im_start": im_start_count,
                        "im_end": im_end_count,
                        "parsed_segments": len(role_matches),
                    },
                )

            segments_by_role: dict[str, list[str]] = defaultdict(list)
            for match in role_matches:
                segments_by_role[match.group(1).strip()].append(match.group(2))
            for required_role in ("system", "user", "assistant"):
                if not segments_by_role[required_role]:
                    add_anomaly(
                        anomaly_counts,
                        anomaly_examples,
                        f"missing_{required_role}_segment",
                        sample,
                        row_index,
                        text,
                    )

            for system_prompt in segments_by_role["system"]:
                prompt = system_prompt.strip()
                system_prompts[prompt] += 1
                lowered = prompt.casefold()
                if "qwen2.5" in lowered:
                    system_identity_distribution["Qwen2.5"] += 1
                elif "qwen3" in lowered:
                    system_identity_distribution["Qwen3"] += 1
                elif "qwen" in lowered:
                    system_identity_distribution["other Qwen"] += 1
                else:
                    system_identity_distribution["no Qwen identity found"] += 1
                if "search" in lowered:
                    system_prompt_checks["contains_search_tool_description"] += 1
                if all(f"<{tag}>" in prompt for tag in ("think", "search", "answer")):
                    system_prompt_checks["contains_think_search_answer_rules"] += 1

            user_text = "\n".join(segments_by_role["user"])
            if isinstance(question, str) and question not in user_text:
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "top_level_question_not_in_user_segment",
                    sample,
                    row_index,
                    user_text,
                )

            assistant_text = "\n".join(segments_by_role["assistant"])
            information_text = assistant_text
            extracted: dict[str, list[str]] = {
                "think": tag_pattern("think").findall(assistant_text),
                "search": tag_pattern("search").findall(assistant_text),
                "answer": tag_pattern("answer").findall(assistant_text),
                "information": tag_pattern("information").findall(information_text),
            }
            for tag in CONTROL_TAGS:
                if extracted[tag]:
                    tag_presence[tag] += 1
                relevant_text = (
                    assistant_text if tag in {"think", "search", "answer"} else information_text
                )
                opening_count = relevant_text.count(f"<{tag}>")
                closing_count = relevant_text.count(f"</{tag}>")
                if opening_count != closing_count:
                    add_anomaly(
                        anomaly_counts,
                        anomaly_examples,
                        f"unpaired_{tag}_tags",
                        sample,
                        row_index,
                        relevant_text,
                    )

            for query in extracted["search"]:
                stripped_query = query.strip()
                query_lengths.append(len(stripped_query))
                if not stripped_query:
                    empty_query_count += 1
                    add_anomaly(
                        anomaly_counts,
                        anomaly_examples,
                        "empty_search_query",
                        sample,
                        row_index,
                        query,
                    )
                if len(stripped_query) > 500:
                    query_over_500_count += 1
            for information in extracted["information"]:
                if not information.strip():
                    add_anomaly(
                        anomaly_counts,
                        anomaly_examples,
                        "empty_information",
                        sample,
                        row_index,
                        information,
                    )
            for thought in extracted["think"]:
                if not thought.strip():
                    add_anomaly(
                        anomaly_counts,
                        anomaly_examples,
                        "empty_think",
                        sample,
                        row_index,
                        thought,
                    )

            answers = [answer.strip() for answer in extracted["answer"]]
            nonempty_answers = [answer for answer in answers if answer]
            if any(not answer for answer in answers):
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "empty_final_answer",
                    sample,
                    row_index,
                    extracted["answer"],
                )
            if len(nonempty_answers) != 1:
                final_answer_extraction_failures += 1
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "final_answer_not_unique",
                    sample,
                    row_index,
                    extracted["answer"],
                )
            else:
                final_answer_lengths.append(len(nonempty_answers[0]))

            if type(search_count) is int and len(extracted["search"]) != search_count:
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "search_block_count_mismatch",
                    sample,
                    row_index,
                    {"extracted": len(extracted["search"]), "declared": search_count},
                )
            if len(extracted["information"]) != len(extracted["search"]):
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "information_search_count_mismatch",
                    sample,
                    row_index,
                    {
                        "search_blocks": len(extracted["search"]),
                        "information_blocks": len(extracted["information"]),
                    },
                )

            answer_close = text.rfind("</answer>")
            if answer_close >= 0 and text.find("<search>", answer_close) >= 0:
                add_anomaly(
                    anomaly_counts,
                    anomaly_examples,
                    "search_after_final_answer",
                    sample,
                    row_index,
                    text[answer_close:],
                )

        quality["duplicate_id"] = duplicate_summary(split_ids)
        quality["duplicate_raw_question"] = duplicate_summary(split_raw_questions)
        quality["duplicate_normalized_question"] = duplicate_summary(
            split_normalized_questions
        )
        quality["duplicate_full_trajectory_text"] = duplicate_summary(
            split_trajectories
        )
        quality_by_split[split] = dict(quality)

    expected_ids = set(range(1, 2001))
    id_audit = {
        "format": r"^\d{6}$",
        "global_unique": all(count == 1 for count in global_ids.values()),
        "duplicate_summary": duplicate_summary(global_ids),
        "valid_formatted_id_count": len(valid_numeric_ids),
        "missing_ids_000001_to_002000": [
            f"{value:06d}" for value in sorted(expected_ids - valid_numeric_ids)
        ],
        "out_of_range_ids": [
            f"{value:06d}" for value in sorted(valid_numeric_ids - expected_ids)
        ],
        "invalid_format_count": sum(
            count for value, count in global_ids.items() if not ID_PATTERN.fullmatch(value)
        ),
        "invalid_format_examples": invalid_id_format_examples,
    }
    relationship_audit = {
        "trajectory_type_distribution": dict(type_distribution),
        "search_count_distribution": dict(search_count_distribution),
        "trajectory_type_search_count_cross_distribution": dict(type_search_cross),
        "relationship_mismatch_count": relationship_mismatch_count,
        "relationship_mismatch_examples_max_20": relationship_mismatches,
        "all_search_counts_in_1_to_4": all(
            type(sample.get("search_count")) is int
            and 1 <= sample["search_count"] <= 4
            for _, _, sample in flat_samples
        ),
    }

    first_text = flat_samples[0][2]["full_trajectory_text"] if flat_samples else ""
    first_role_matches = list(ROLE_PATTERN.finditer(first_text))
    first_assistant = "\n".join(
        match.group(2)
        for match in first_role_matches
        if match.group(1).strip() == "assistant"
    )
    protocol_discovery = {
        "role_markers": dict(role_marker_distribution),
        "role_order_distribution": dict(role_order_distribution),
        "im_start_count_distribution": dict(im_start_count_distribution),
        "im_end_count_distribution": dict(im_end_count_distribution),
        "tag_presence_by_trajectory": dict(tag_presence),
        "search_query_representation": "<search>...</search> inside assistant segment",
        "search_result_representation": "<information>...</information> inside assistant segment",
        "final_answer_representation": "<answer>...</answer> inside assistant segment",
        "first_search_queries": tag_pattern("search").findall(first_assistant),
        "first_information_blocks": tag_pattern("information").findall(
            first_assistant
        ),
        "first_final_answers": tag_pattern("answer").findall(first_assistant),
        "system_prompt_distinct_count": len(system_prompts),
        "system_prompt_counts": [
            {"count": count, "content": prompt}
            for prompt, count in system_prompts.items()
        ],
        "first_system_prompt_full": next(iter(system_prompts), None),
        "system_identity_distribution": dict(system_identity_distribution),
        "system_prompt_checks": dict(system_prompt_checks),
    }

    longest_trajectories.sort(key=lambda item: item["character_length"], reverse=True)
    length_audit = {
        "full_trajectory_text_lengths": describe_lengths(trajectory_lengths),
        "longest_10_trajectories": longest_trajectories[:10],
        "search_query_lengths": describe_lengths(query_lengths),
        "empty_search_query_count": empty_query_count,
        "search_query_over_500_characters_count": query_over_500_count,
        "final_answer_lengths": describe_lengths(final_answer_lengths),
        "unable_to_extract_unique_final_answer_count": final_answer_extraction_failures,
    }

    cross_audit: dict[str, Any] = {}
    for source_name, path in LOCAL_DATASETS.items():
        if not path.exists():
            cross_audit[source_name] = {"status": "skipped", "reason": "file missing"}
            continue

        local_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                record = json.loads(line)
                question = record.get("question")
                if isinstance(question, str):
                    local_questions[normalize_question(question)].append(
                        {
                            "uid": record.get("uid"),
                            "question": question,
                            "gold_answers": record.get("gold_answers"),
                            "line_number": line_number,
                        }
                    )

        overlap_keys = set(aether_questions) & set(local_questions)
        overlap_examples = []
        for key in sorted(overlap_keys):
            for aether in aether_questions[key]:
                for local in local_questions[key]:
                    if len(overlap_examples) >= MAX_OVERLAP_EXAMPLES:
                        break
                    overlap_examples.append(
                        {
                            "aethersearch_id": aether["id"],
                            "aethersearch_question": aether["question"],
                            "source_split": source_name,
                            "source_uid": local["uid"],
                            "source_gold_answers": local["gold_answers"],
                        }
                    )
                if len(overlap_examples) >= MAX_OVERLAP_EXAMPLES:
                    break
            if len(overlap_examples) >= MAX_OVERLAP_EXAMPLES:
                break

        cross_audit[source_name] = {
            "status": "audited",
            "unique_overlapping_questions": len(overlap_keys),
            "aethersearch_samples_involved": sum(
                len(aether_questions[key]) for key in overlap_keys
            ),
            "examples_max_20": overlap_examples,
        }

    print("\n=== FIELD QUALITY AND PYTHON TYPES ===")
    print_json(
        {
            "expected_fields": sorted(EXPECTED_FIELDS),
            "actual_field_python_types": {
                split: {
                    field: dict(counts) for field, counts in fields.items()
                }
                for split, fields in field_python_types.items()
            },
            "quality_by_split": quality_by_split,
            "global_duplicate_raw_questions": duplicate_summary(raw_questions),
            "global_duplicate_normalized_questions": duplicate_summary(
                normalized_questions
            ),
            "global_duplicate_trajectories": duplicate_summary(trajectories),
            "id_audit": id_audit,
        }
    )
    print("\n=== TRAJECTORY TYPE AND SEARCH COUNT ===")
    print_json(relationship_audit)
    print("\n=== DISCOVERED TRAJECTORY PROTOCOL ===")
    print_json(protocol_discovery)
    print("\n=== TRAJECTORY STRUCTURE VALIDATION ===")
    print_json(
        {
            "total_trajectories": len(flat_samples),
            "stably_parsed_as_messages": parsed_message_trajectories,
            "anomaly_counts": dict(anomaly_counts),
            "anomaly_examples_max_10_per_category": anomaly_examples,
        }
    )
    print("\n=== LENGTH AND CONTENT STATISTICS ===")
    print_json(length_audit)
    print("\n=== LOCAL NQ/HOTPOTQA CROSS AUDIT ===")
    print_json(cross_audit)


if __name__ == "__main__":
    main()
