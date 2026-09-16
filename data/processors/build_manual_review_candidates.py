from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/dedup_audit/near_duplicate_candidates.jsonl"
)
OUTPUT_PATH = Path(
    "data/processed/dedup_audit/manual_review_candidates.jsonl"
)


def source_pair(left_source: str, right_source: str) -> str:
    return "__".join(sorted([left_source, right_source]))


def main() -> None:
    candidates = []

    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            candidate = json.loads(line)
            left = candidate["left"]
            right = candidate["right"]
            jaccard = candidate["jaccard"]

            if left["source"] != right["source"]:
                review_group = "cross_source"
            elif jaccard >= 0.90:
                review_group = "same_source_high_similarity"
            else:
                continue

            candidates.append(
                {
                    "review_group": review_group,
                    "jaccard": jaccard,
                    "left": {
                        "uid": left["id"],
                        "source": left["source"],
                        "question": left["question"],
                        "answers": left["answers"],
                    },
                    "right": {
                        "uid": right["id"],
                        "source": right["source"],
                        "question": right["question"],
                        "answers": right["answers"],
                    },
                    "review": None,
                }
            )

    group_order = {
        "cross_source": 0,
        "same_source_high_similarity": 1,
    }
    candidates.sort(
        key=lambda item: (
            group_order[item["review_group"]],
            -item["jaccard"],
        )
    )

    pair_counts = Counter()
    group_counts = Counter()

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for pair_id, candidate in enumerate(candidates):
            candidate = {"pair_id": pair_id, **candidate}
            output_file.write(
                json.dumps(candidate, ensure_ascii=False) + "\n"
            )

            group_counts[candidate["review_group"]] += 1
            pair_counts[
                source_pair(
                    candidate["left"]["source"],
                    candidate["right"]["source"],
                )
            ] += 1

    null_review_count = sum(
        candidate["review"] is None for candidate in candidates
    )

    print(f"Total manual review candidates: {len(candidates)}")
    print(
        "Cross-source candidates: "
        f"{group_counts['cross_source']}"
    )
    print(
        "Same-source high-similarity candidates: "
        f"{group_counts['same_source_high_similarity']}"
    )
    print()

    for pair, count in sorted(pair_counts.items()):
        print(f"{pair}: {count}")

    print()
    print(f"review == null: {null_review_count}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
