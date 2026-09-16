from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DATASET_DIRS = {
    "nq": Path("data/processed/nq"),
    "hotpotqa": Path("data/processed/hotpotqa"),
    "aethersearch": Path("data/processed/aethersearch"),
}
REVIEW_PATH = Path(
    "data/processed/dedup_audit/manual_review_candidates_reviewed.jsonl"
)
OUTPUT_DIR = Path("data/processed/unified")
POOL_PATH = OUTPUT_DIR / "unique_qa_pool.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "dedup_manifest.jsonl"


def normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[?!.。？！]+$", "", text).strip()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def load_records(source: str, directory: Path) -> list[dict]:
    records = []

    for path in sorted(directory.glob("*.jsonl")):
        for item in read_jsonl(path):
            if source == "aethersearch":
                record = {
                    "uid": item["id"],
                    "source": source,
                    "source_split": item.get("source_split", path.stem),
                    "question": item["question"],
                    "gold_answers": item["answers"],
                    "metadata": {
                        "trajectory_type": item["trajectory_type"],
                        "search_count": item["search_count"],
                    },
                }
            else:
                record = {
                    "uid": item["uid"],
                    "source": item["source"],
                    "source_split": item["source_split"],
                    "question": item["question"],
                    "gold_answers": item["gold_answers"],
                    "metadata": item["metadata"],
                }

            records.append(record)

    return records


def main() -> None:
    records = []
    input_counts = {}

    for source, directory in DATASET_DIRS.items():
        source_records = load_records(source, directory)
        records.extend(source_records)
        input_counts[source] = len(source_records)

    records_by_uid = {record["uid"]: record for record in records}
    if len(records_by_uid) != len(records):
        raise ValueError("Input uids are not globally unique")

    parent = {uid: uid for uid in records_by_uid}

    def find(uid: str) -> str:
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def union(left_uid: str, right_uid: str) -> None:
        left_root = find(left_uid)
        right_root = find(right_uid)
        if left_root != right_root:
            parent[right_root] = left_root

    normalized_groups = defaultdict(list)
    for record in records:
        normalized_groups[normalize_question(record["question"])].append(
            record["uid"]
        )

    exact_groups = [
        uids for uids in normalized_groups.values() if len(uids) > 1
    ]
    evidence_relations = []

    for uids in exact_groups:
        for uid in uids[1:]:
            union(uids[0], uid)
            evidence_relations.append((uids[0], uid, "exact_duplicate"))

    manual_duplicate_pairs = 0
    for reviewed in read_jsonl(REVIEW_PATH):
        left_uid = reviewed["left"]["uid"]
        right_uid = reviewed["right"]["uid"]

        if left_uid not in records_by_uid or right_uid not in records_by_uid:
            raise ValueError(
                f"Reviewed pair references unknown uid: "
                f"{left_uid}, {right_uid}"
            )

        if reviewed["review"] == "duplicate":
            union(left_uid, right_uid)
            evidence_relations.append(
                (left_uid, right_uid, "manual_duplicate")
            )
            manual_duplicate_pairs += 1

    components = defaultdict(list)
    for uid in records_by_uid:
        components[find(uid)].append(uid)

    evidence_by_component = defaultdict(set)
    for left_uid, _, evidence in evidence_relations:
        evidence_by_component[find(left_uid)].add(evidence)

    unique_records = []
    manifests = []
    removed_uids = set()

    for member_uids in components.values():
        aethersearch_uids = [
            uid
            for uid in member_uids
            if records_by_uid[uid]["source"] == "aethersearch"
        ]
        canonical_uid = min(aethersearch_uids or member_uids)
        unique_records.append(records_by_uid[canonical_uid])

        if len(member_uids) > 1:
            removed = sorted(
                uid for uid in member_uids if uid != canonical_uid
            )
            removed_uids.update(removed)
            manifests.append(
                {
                    "canonical_uid": canonical_uid,
                    "canonical_source": records_by_uid[canonical_uid][
                        "source"
                    ],
                    "removed_uids": removed,
                    "members": [
                        {
                            "uid": uid,
                            "source": records_by_uid[uid]["source"],
                            "question": records_by_uid[uid]["question"],
                            "gold_answers": records_by_uid[uid][
                                "gold_answers"
                            ],
                        }
                        for uid in sorted(member_uids)
                    ],
                    "evidence": sorted(
                        evidence_by_component[find(canonical_uid)]
                    ),
                }
            )

    unique_records.sort(key=lambda record: record["uid"])
    manifests.sort(key=lambda manifest: manifest["canonical_uid"])

    unique_uids = {record["uid"] for record in unique_records}
    if len(unique_uids) != len(unique_records):
        raise ValueError("Unique pool contains duplicate uids")
    if removed_uids & unique_uids:
        raise ValueError("Removed uid remains in unique pool")
    if any(m["canonical_uid"] not in unique_uids for m in manifests):
        raise ValueError("Manifest canonical uid is missing from unique pool")
    if len(records) != len(unique_records) + len(removed_uids):
        raise ValueError("Input/unique/removed record counts are inconsistent")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with POOL_PATH.open("w", encoding="utf-8") as output_file:
        for record in unique_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with MANIFEST_PATH.open("w", encoding="utf-8") as output_file:
        for manifest in manifests:
            output_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    remaining_counts = Counter(
        record["source"] for record in unique_records
    )

    print("=" * 60)
    print("Unique QA Pool Build Completed")
    print("=" * 60)
    print("Input records:")
    print(f"  NQ           : {input_counts['nq']}")
    print(f"  HotpotQA     : {input_counts['hotpotqa']}")
    print(f"  AetherSearch : {input_counts['aethersearch']}")
    print(f"  Total        : {len(records)}")
    print()
    print(f"Exact duplicate groups       : {len(exact_groups)}")
    print(f"Manual duplicate pairs used : {manual_duplicate_pairs}")
    print(f"Final duplicate components  : {len(manifests)}")
    print(f"Records removed             : {len(removed_uids)}")
    print(f"Unique QA records           : {len(unique_records)}")
    print()
    print("Remaining by source:")
    print(f"  NQ           : {remaining_counts['nq']}")
    print(f"  HotpotQA     : {remaining_counts['hotpotqa']}")
    print(f"  AetherSearch : {remaining_counts['aethersearch']}")
    print()
    print("Output:")
    print(f"  {POOL_PATH}")
    print(f"  {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
