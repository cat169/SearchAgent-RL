"""CPU-only analysis of unpadded real rollout records."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(input_path, output, expected=4, selected_path=None):
    records = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    seen = set()
    for record in records:
        key = (record["question_id"], record["rollout_id"])
        if key in seen:
            raise ValueError(f"Duplicate rollout: {key}")
        seen.add(key)
        groups[record["question_id"]].append(record)
    if selected_path:
        for row in json.loads(Path(selected_path).read_text(encoding="utf-8")):
            groups.setdefault(str(row["extra_info"]["id"]), [])
    valid = [r for r in records if r["status"] == "ok"]
    if any(r["reward"] not in (0, 1) for r in valid):
        raise ValueError("Expected binary rewards")
    stats = []
    categories = Counter()
    for qid, rows in sorted(groups.items()):
        rewards = [r["reward"] for r in rows if r["status"] == "ok"]
        complete = len(rows) == expected and len(rewards) == expected
        category = ("mixed" if 0 < sum(rewards) < expected else
                    "all_one" if sum(rewards) == expected else "all_zero") if complete else "incomplete"
        categories[category] += 1
        stats.append(dict(question_id=qid, record_count=len(rows), valid_count=len(rewards),
                          rewards=json.dumps(rewards), category=category))
    n = len(valid)
    complete_count = sum(categories[k] for k in ("all_zero", "all_one", "mixed"))
    summary = dict(record_count=len(records), valid_record_count=n,
        failed_record_count=len(records)-n, expected_rollouts=expected,
        answer_completion_rate=sum(r["termination_reason"] == "answer" for r in valid)/n if n else None,
        strict_protocol_validity=sum(bool(r["strict_protocol_valid"]) for r in valid)/n if n else None,
        mean_reward=sum(r["reward"] for r in valid)/n if n else None,
        mean_search_count=sum(r["search_count"] for r in valid)/n if n else None,
        termination_reason_distribution=dict(Counter(r["termination_reason"] for r in records)),
        group_count=len(groups), complete_group_count=complete_count,
        incomplete_group_count=categories["incomplete"],
        informative_group_ratio=categories["mixed"]/complete_count if complete_count else None,
        group_counts=dict(categories),
        complete_group_ratios={k: categories[k]/complete_count if complete_count else None
                               for k in ("all_zero", "all_one", "mixed")})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "group_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_id", "record_count", "valid_count", "rewards", "category"])
        writer.writeheader()
        writer.writerows(stats)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-questions", type=Path)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    args = parser.parse_args()
    selected = args.selected_questions or args.input.parent / "selected_questions.json"
    analyze(args.input, args.output or args.input.parent, args.expected_rollouts,
            selected if selected.exists() else None)


if __name__ == "__main__":
    main()
