from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from datasketch import MinHash, MinHashLSH


SOURCE_PRIORITY = {
    "aethersearch": 0,
    "nq": 1,
    "hotpotqa": 2,
}


# ============================================================
# JSON helpers
# ============================================================

def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

            yield item


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Question normalization
# ============================================================

def normalize_question(question: str) -> str:
    """
    Conservative normalization for deduplication only.

    Important:
    - This NEVER overwrites the original question.
    - No stemming.
    - No stopword removal.
    - No synonym replacement.
    - No semantic normalization.
    """

    text = unicodedata.normalize("NFKC", question)
    text = text.casefold()

    # Normalize all whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove only trailing sentence-ending punctuation.
    # Internal punctuation is intentionally preserved.
    text = re.sub(r"[?!.。？！]+$", "", text).strip()

    return text


# ============================================================
# Answer extraction
# ============================================================

def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    result = []
    seen = set()

    for answer in value:
        if not isinstance(answer, str):
            continue

        answer = answer.strip()
        if not answer:
            continue

        key = answer.casefold()

        if key not in seen:
            seen.add(key)
            result.append(answer)

    return result


def extract_answers(item: dict[str, Any]) -> list[str]:
    """
    Extract reference answers from different processed schemas.

    NQ / HotpotQA canonical QA:
        gold_answers

    Generic canonical QA:
        answers

    AetherSearch trajectory fallback:
        final_answer / answer event
    """

    # NQ / HotpotQA formal processed schema
    answers = normalize_answers(item.get("gold_answers"))
    if answers:
        return answers

    # Generic canonical schema fallback
    answers = normalize_answers(item.get("answers"))
    if answers:
        return answers

    # AetherSearch-style derived answer
    answers = normalize_answers(item.get("final_answer"))
    if answers:
        return answers

    # Last fallback: recover final answer from events
    events = item.get("events")

    if isinstance(events, list):
        for event in reversed(events):
            if (
                isinstance(event, dict)
                and event.get("type") == "answer"
            ):
                answers = normalize_answers(
                    event.get("content")
                )

                if answers:
                    return answers

    return []

# ============================================================
# Dataset loading
# ============================================================

def load_dataset_directory(
    directory: Path,
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    jsonl_files = sorted(directory.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No .jsonl files found under: {directory}"
        )

    records = []

    stats = {
        "read": 0,
        "written": 0,
        "skipped_empty_question": 0,
    }

    for path in jsonl_files:
        fallback_split = path.stem

        print(f"[load] {source}: {path}")

        for item in read_jsonl(path):
            stats["read"] += 1

            question = item.get("question")

            if not isinstance(question, str):
                stats["skipped_empty_question"] += 1
                continue

            question = question.strip()

            if not question:
                stats["skipped_empty_question"] += 1
                continue

            raw_id = item.get("uid")

            if raw_id is None:
                raw_id = item.get("id")

            if raw_id is None:
                raw_id = (
                    f"{source}_{fallback_split}_"
                    f"{stats['read'] - 1}"
                )

            original_split = (
                item.get("source_split")
                or item.get("split")
                or item.get("original_split")
                or fallback_split
            )

            normalized = normalize_question(question)

            if not normalized:
                stats["skipped_empty_question"] += 1
                continue

            record = {
                "uid": str(raw_id),
                "id": str(raw_id),
                "source": source,
                "original_split": str(original_split),
                "question": question,
                "normalized_question": normalized,
                "answers": extract_answers(item),
            }

            records.append(record)
            stats["written"] += 1

    return records, stats


# ============================================================
# Exact duplicate analysis
# ============================================================

def build_exact_groups(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups = defaultdict(list)

    for record in records:
        groups[record["normalized_question"]].append(record)

    return dict(groups)


def choose_representative(
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Used only for near-duplicate audit.

    If the exact group already contains AetherSearch, prefer its
    wording as the representative because AetherSearch has an
    existing trajectory.

    This does NOT yet modify or delete formal data.
    """

    return min(
        members,
        key=lambda item: (
            SOURCE_PRIORITY.get(item["source"], 999),
            len(item["question"]),
            item["uid"],
        ),
    )


def merge_answers_for_display(
    members: list[dict[str, Any]],
) -> list[str]:
    """
    Used only for auditing.

    Formal answer merging decisions will be made later.
    """

    result = []
    seen = set()

    for member in members:
        for answer in member["answers"]:
            key = answer.casefold()

            if key not in seen:
                seen.add(key)
                result.append(answer)

    return result


def make_exact_group_output(
    normalized_question: str,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = sorted(
        {member["source"] for member in members}
    )

    return {
        "normalized_question": normalized_question,
        "size": len(members),
        "sources": sources,
        "members": [
            {
                "id": member["id"],
                "source": member["source"],
                "original_split": member["original_split"],
                "question": member["question"],
                "answers": member["answers"],
            }
            for member in members
        ],
    }


def compute_exact_stats(
    records: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    load_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    duplicate_groups = [
        members
        for members in groups.values()
        if len(members) > 1
    ]

    source_record_counts = Counter(
        record["source"]
        for record in records
    )

    within_source_duplicate_groups = Counter()

    cross_source_duplicate_groups = Counter()

    for members in duplicate_groups:
        per_source = Counter(
            member["source"]
            for member in members
        )

        for source, count in per_source.items():
            if count >= 2:
                within_source_duplicate_groups[source] += 1

        sources = sorted(per_source.keys())

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                pair_name = f"{sources[i]}__{sources[j]}"
                cross_source_duplicate_groups[pair_name] += 1

    return {
        "load_stats": load_stats,
        "total_questions": len(records),
        "source_record_counts": dict(source_record_counts),
        "unique_after_exact_normalization": len(groups),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_records_beyond_one": (
            len(records) - len(groups)
        ),
        "within_source_duplicate_groups": dict(
            within_source_duplicate_groups
        ),
        "cross_source_duplicate_groups": dict(
            cross_source_duplicate_groups
        ),
    }


# ============================================================
# Near duplicate analysis
# ============================================================

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def build_shingles(normalized_question: str) -> frozenset[str]:
    """
    Mixed word unigram + bigram shingles.

    Example:
      "who directed inception"

    becomes:
      u:who
      u:directed
      u:inception
      b:who directed
      b:directed inception
    """

    tokens = WORD_PATTERN.findall(normalized_question)

    shingles: set[str] = set()

    for token in tokens:
        shingles.add(f"u:{token}")

    for i in range(len(tokens) - 1):
        shingles.add(
            f"b:{tokens[i]}\u241f{tokens[i + 1]}"
        )

    return frozenset(shingles)


def build_minhash(
    shingles: frozenset[str],
    num_perm: int,
    seed: int,
) -> MinHash:
    minhash = MinHash(
        num_perm=num_perm,
        seed=seed,
    )

    for shingle in shingles:
        minhash.update(
            shingle.encode("utf-8")
        )

    return minhash


def jaccard_similarity(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    intersection = len(left.intersection(right))
    union = len(left.union(right))

    return intersection / union


def get_bin_name(score: float) -> str | None:
    if 0.60 <= score < 0.70:
        return "060_070"

    if 0.70 <= score < 0.80:
        return "070_080"

    if 0.80 <= score < 0.90:
        return "080_090"

    if 0.90 <= score <= 1.00:
        return "090_100"

    return None


def make_source_pair(
    left_source: str,
    right_source: str,
) -> str:
    return "__".join(sorted([left_source, right_source]))


def reservoir_add(
    samples: list[dict[str, Any]],
    item: dict[str, Any],
    seen_count: int,
    max_samples: int,
    rng: random.Random,
) -> None:
    """
    Uniform reservoir sampling without storing every candidate.
    """

    if len(samples) < max_samples:
        samples.append(item)
        return

    replacement_index = rng.randint(
        0,
        seen_count - 1,
    )

    if replacement_index < max_samples:
        samples[replacement_index] = item


def build_near_duplicate_audit(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    num_perm: int,
    minhash_seed: int,
    lsh_threshold: float,
    min_jaccard: float,
    samples_per_bin: int,
    random_seed: int,
) -> dict[str, Any]:
    normalized_questions = list(groups.keys())

    group_views = []

    for normalized_question in normalized_questions:
        members = groups[normalized_question]
        representative = choose_representative(members)

        group_views.append(
            {
                "normalized_question": normalized_question,
                "representative": representative,
                "sources": sorted(
                    {member["source"] for member in members}
                ),
                "exact_group_size": len(members),
                "answers": merge_answers_for_display(members),
            }
        )

    print(
        f"[near] building LSH index for "
        f"{len(group_views):,} exact-unique questions"
    )

    lsh = MinHashLSH(
        threshold=lsh_threshold,
        num_perm=num_perm,
    )

    inserted = 0
    skipped_no_shingles = 0

    # Pass 1: build LSH index.
    for index, view in enumerate(group_views):
        shingles = build_shingles(
            view["normalized_question"]
        )

        if not shingles:
            skipped_no_shingles += 1
            continue

        minhash = build_minhash(
            shingles=shingles,
            num_perm=num_perm,
            seed=minhash_seed,
        )

        lsh.insert(
            f"g{index}",
            minhash,
        )

        inserted += 1

        if inserted % 10000 == 0:
            print(
                f"[near] indexed {inserted:,} questions"
            )

    # Cache shingles during query/Jaccard pass.
    @lru_cache(maxsize=50000)
    def cached_shingles(index: int) -> frozenset[str]:
        return build_shingles(
            group_views[index]["normalized_question"]
        )

    rng = random.Random(random_seed)

    bin_samples: dict[str, list[dict[str, Any]]] = {
        "060_070": [],
        "070_080": [],
        "080_090": [],
        "090_100": [],
    }

    bin_seen_counts = Counter()

    source_pair_samples: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(
        lambda: {
            "060_070": [],
            "070_080": [],
            "080_090": [],
            "090_100": [],
        }
    )

    source_pair_bin_counts: dict[str, Counter[str]] = (
        defaultdict(Counter)
    )
    source_pair_candidates = Counter()

    candidate_pairs_from_lsh = 0
    pairs_at_or_above_min_jaccard = 0

    cross_source_candidates = 0
    within_source_candidates = 0

    candidate_output_path = (
        output_dir / "near_duplicate_candidates.jsonl"
    )

    print("[near] querying LSH candidates")

    with candidate_output_path.open(
        "w",
        encoding="utf-8",
    ) as candidate_file:

        for left_index, left_view in enumerate(group_views):
            left_shingles = cached_shingles(left_index)

            if not left_shingles:
                continue

            left_minhash = build_minhash(
                shingles=left_shingles,
                num_perm=num_perm,
                seed=minhash_seed,
            )

            matches = lsh.query(left_minhash)

            for match_key in matches:
                right_index = int(match_key[1:])

                # Process every unordered pair only once.
                if right_index <= left_index:
                    continue

                candidate_pairs_from_lsh += 1

                right_view = group_views[right_index]
                right_shingles = cached_shingles(right_index)

                score = jaccard_similarity(
                    left_shingles,
                    right_shingles,
                )

                if score < min_jaccard:
                    continue

                pairs_at_or_above_min_jaccard += 1

                left_sources = set(left_view["sources"])
                right_sources = set(right_view["sources"])

                if left_sources != right_sources:
                    # This is intentionally broad:
                    # any source composition difference is useful
                    # for cross-source audit.
                    cross_source_candidates += 1
                else:
                    within_source_candidates += 1

                left_rep = left_view["representative"]
                right_rep = right_view["representative"]

                source_pair = make_source_pair(
                    left_rep["source"],
                    right_rep["source"],
                )
                source_pair_candidates[source_pair] += 1

                candidate = {
                    "jaccard": round(score, 6),

                    "left": {
                        "id": left_rep["id"],
                        "source": left_rep["source"],
                        "original_split": left_rep["original_split"],
                        "question": left_rep["question"],
                        "normalized_question": (
                            left_view["normalized_question"]
                        ),
                        "answers": left_view["answers"],
                        "sources_in_exact_group": (
                            left_view["sources"]
                        ),
                        "exact_group_size": (
                            left_view["exact_group_size"]
                        ),
                    },

                    "right": {
                        "id": right_rep["id"],
                        "source": right_rep["source"],
                        "original_split": right_rep["original_split"],
                        "question": right_rep["question"],
                        "normalized_question": (
                            right_view["normalized_question"]
                        ),
                        "answers": right_view["answers"],
                        "sources_in_exact_group": (
                            right_view["sources"]
                        ),
                        "exact_group_size": (
                            right_view["exact_group_size"]
                        ),
                    },
                }

                candidate_file.write(
                    json.dumps(
                        candidate,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                bin_name = get_bin_name(score)

                if bin_name is not None:
                    bin_seen_counts[bin_name] += 1
                    source_pair_bin_counts[source_pair][bin_name] += 1

                    reservoir_add(
                        samples=bin_samples[bin_name],
                        item=candidate,
                        seen_count=bin_seen_counts[bin_name],
                        max_samples=samples_per_bin,
                        rng=rng,
                    )

                    reservoir_add(
                        samples=(
                            source_pair_samples[source_pair][bin_name]
                        ),
                        item=candidate,
                        seen_count=(
                            source_pair_bin_counts[source_pair][bin_name]
                        ),
                        max_samples=samples_per_bin,
                        rng=rng,
                    )

            if (left_index + 1) % 10000 == 0:
                print(
                    f"[near] queried "
                    f"{left_index + 1:,}/"
                    f"{len(group_views):,}"
                )

    for bin_name, samples in bin_samples.items():
        write_jsonl(
            output_dir / f"audit_{bin_name}.jsonl",
            samples,
        )

    for source_pair in sorted(source_pair_candidates):
        pair_output_dir = (
            output_dir / "by_source_pair" / source_pair
        )

        for bin_name, samples in source_pair_samples[
            source_pair
        ].items():
            write_jsonl(
                pair_output_dir / f"audit_{bin_name}.jsonl",
                samples,
            )

    near_stats = {
        "exact_unique_questions": len(group_views),
        "lsh_inserted_questions": inserted,
        "skipped_no_shingles": skipped_no_shingles,

        "configuration": {
            "shingles": "word_unigram_plus_bigram",
            "num_perm": num_perm,
            "minhash_seed": minhash_seed,
            "lsh_threshold": lsh_threshold,
            "min_jaccard_written": min_jaccard,
            "samples_per_bin": samples_per_bin,
            "random_seed": random_seed,
        },

        "candidate_pairs_from_lsh": candidate_pairs_from_lsh,
        "pairs_at_or_above_min_jaccard": (
            pairs_at_or_above_min_jaccard
        ),

        "within_source_candidates": within_source_candidates,
        "cross_source_candidates": cross_source_candidates,

        "jaccard_bins": dict(bin_seen_counts),
        "source_pair_candidates": dict(
            sorted(source_pair_candidates.items())
        ),
        "jaccard_bins_by_source_pair": {
            source_pair: {
                bin_name: source_pair_bin_counts[
                    source_pair
                ].get(bin_name, 0)
                for bin_name in [
                    "060_070",
                    "070_080",
                    "080_090",
                    "090_100",
                ]
            }
            for source_pair in sorted(source_pair_candidates)
        },
    }

    return near_stats


# ============================================================
# Main
# ============================================================

def parse_audit_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact and near question duplicates across "
            "NQ, HotpotQA, and AetherSearch. "
            "This script does NOT delete or split formal data."
        )
    )

    parser.add_argument(
        "--nq_dir",
        type=Path,
        default=Path("data/processed/nq"),
    )

    parser.add_argument(
        "--hotpotqa_dir",
        type=Path,
        default=Path("data/processed/hotpotqa"),
    )

    parser.add_argument(
        "--aethersearch_dir",
        type=Path,
        default=Path("data/processed/aethersearch"),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/dedup_audit"),
    )

    parser.add_argument(
        "--num_perm",
        type=int,
        default=128,
        help="Number of MinHash permutations.",
    )

    parser.add_argument(
        "--minhash_seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--lsh_threshold",
        type=float,
        default=0.50,
        help=(
            "Loose MinHash-LSH threshold used only to retrieve "
            "candidate pairs."
        ),
    )

    parser.add_argument(
        "--min_jaccard",
        type=float,
        default=0.60,
        help=(
            "Only candidate pairs with true Jaccard >= this "
            "value are written."
        ),
    )

    parser.add_argument(
        "--samples_per_bin",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=123,
    )

    return parser.parse_args(arguments)


def run_audit(args: argparse.Namespace) -> None:
    if args.num_perm < 1:
        raise ValueError("--num_perm must be >= 1")

    if not (0.0 < args.lsh_threshold <= 1.0):
        raise ValueError(
            "--lsh_threshold must be in (0, 1]"
        )

    if not (0.0 <= args.min_jaccard <= 1.0):
        raise ValueError(
            "--min_jaccard must be in [0, 1]"
        )

    if args.samples_per_bin < 1:
        raise ValueError(
            "--samples_per_bin must be >= 1"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    load_stats = {}

    datasets = [
        ("nq", args.nq_dir),
        ("hotpotqa", args.hotpotqa_dir),
        ("aethersearch", args.aethersearch_dir),
    ]

    for source, directory in datasets:
        records, stats = load_dataset_directory(
            directory=directory,
            source=source,
        )

        all_records.extend(records)
        load_stats[source] = stats

        print(
            f"[load] {source}: "
            f"{stats['written']:,} usable questions"
        )

    print()
    print(
        f"[exact] total usable questions: "
        f"{len(all_records):,}"
    )

    exact_groups = build_exact_groups(all_records)

    exact_stats = compute_exact_stats(
        records=all_records,
        groups=exact_groups,
        load_stats=load_stats,
    )

    duplicate_group_rows = []

    for normalized_question, members in exact_groups.items():
        if len(members) <= 1:
            continue

        duplicate_group_rows.append(
            make_exact_group_output(
                normalized_question=normalized_question,
                members=members,
            )
        )

    # Largest groups first for easier inspection.
    duplicate_group_rows.sort(
        key=lambda row: (
            -row["size"],
            row["normalized_question"],
        )
    )

    write_jsonl(
        args.output_dir / "exact_duplicate_groups.jsonl",
        duplicate_group_rows,
    )

    write_json(
        args.output_dir / "exact_duplicate_stats.json",
        exact_stats,
    )

    print(
        f"[exact] unique after normalization: "
        f"{exact_stats['unique_after_exact_normalization']:,}"
    )

    print(
        f"[exact] duplicate groups: "
        f"{exact_stats['exact_duplicate_groups']:,}"
    )

    print(
        f"[exact] duplicate records beyond one: "
        f"{exact_stats['exact_duplicate_records_beyond_one']:,}"
    )

    print()
    print("[near] starting MinHash-LSH audit")

    near_stats = build_near_duplicate_audit(
        groups=exact_groups,
        output_dir=args.output_dir,
        num_perm=args.num_perm,
        minhash_seed=args.minhash_seed,
        lsh_threshold=args.lsh_threshold,
        min_jaccard=args.min_jaccard,
        samples_per_bin=args.samples_per_bin,
        random_seed=args.random_seed,
    )

    write_json(
        args.output_dir / "near_duplicate_stats.json",
        near_stats,
    )

    print()
    print("=" * 60)
    print("Global dedup audit completed")
    print("=" * 60)

    print(
        f"Total questions              : "
        f"{exact_stats['total_questions']:,}"
    )

    print(
        f"Exact-unique questions       : "
        f"{exact_stats['unique_after_exact_normalization']:,}"
    )

    print(
        f"Exact duplicate groups       : "
        f"{exact_stats['exact_duplicate_groups']:,}"
    )

    print(
        f"LSH candidate pairs          : "
        f"{near_stats['candidate_pairs_from_lsh']:,}"
    )

    print(
        f"True Jaccard >= "
        f"{args.min_jaccard:.2f} pairs       : "
        f"{near_stats['pairs_at_or_above_min_jaccard']:,}"
    )

    print()
    print("Jaccard bins:")

    for bin_name in [
        "060_070",
        "070_080",
        "080_090",
        "090_100",
    ]:
        count = near_stats["jaccard_bins"].get(
            bin_name,
            0,
        )

        print(
            f"  {bin_name}: {count:,}"
        )

    print()
    print(
        f"Output directory: {args.output_dir}"
    )

    print()
    print(
        "NOTE: No formal dataset was deleted, merged, "
        "or split by this script."
    )


# ============================================================
# Manual review candidate generation
# ============================================================

MANUAL_INPUT_PATH = Path(
    "data/processed/dedup_audit/near_duplicate_candidates.jsonl"
)
MANUAL_OUTPUT_PATH = Path(
    "data/processed/dedup_audit/manual_review_candidates.jsonl"
)


def build_manual_review_candidates() -> None:
    candidates = []

    for candidate in read_jsonl(MANUAL_INPUT_PATH):
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
    output_rows = []

    for pair_id, candidate in enumerate(candidates):
        candidate = {"pair_id": pair_id, **candidate}
        output_rows.append(candidate)
        group_counts[candidate["review_group"]] += 1
        pair_counts[
            make_source_pair(
                candidate["left"]["source"],
                candidate["right"]["source"],
            )
        ] += 1

    write_jsonl(MANUAL_OUTPUT_PATH, output_rows)
    null_review_count = sum(
        candidate["review"] is None for candidate in candidates
    )

    print(f"Total manual review candidates: {len(candidates)}")
    print(f"Cross-source candidates: {group_counts['cross_source']}")
    print(
        "Same-source high-similarity candidates: "
        f"{group_counts['same_source_high_similarity']}"
    )
    print()
    for pair, count in sorted(pair_counts.items()):
        print(f"{pair}: {count}")
    print()
    print(f"review == null: {null_review_count}")
    print(f"Output: {MANUAL_OUTPUT_PATH}")


# ============================================================
# Unique QA pool generation
# ============================================================

UNIQUE_DATASET_DIRS = {
    "nq": Path("data/processed/nq"),
    "hotpotqa": Path("data/processed/hotpotqa"),
    "aethersearch": Path("data/processed/aethersearch"),
}
REVIEW_PATH = Path(
    "data/processed/dedup_audit/manual_review_candidates_reviewed.jsonl"
)
UNIFIED_OUTPUT_DIR = Path("data/processed/unified")
POOL_PATH = UNIFIED_OUTPUT_DIR / "unique_qa_pool.jsonl"
MANIFEST_PATH = UNIFIED_OUTPUT_DIR / "dedup_manifest.jsonl"


def load_unique_records(source: str, directory: Path) -> list[dict]:
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


def build_unique_qa_pool() -> None:
    records = []
    input_counts = {}

    for source, directory in UNIQUE_DATASET_DIRS.items():
        source_records = load_unique_records(source, directory)
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

    write_jsonl(POOL_PATH, unique_records)
    write_jsonl(MANIFEST_PATH, manifests)
    remaining_counts = Counter(record["source"] for record in unique_records)

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


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/dedup_data.py "
            "{audit|manual-review|unique-pool}"
        )

    stage = sys.argv[1]
    if stage == "audit":
        run_audit(parse_audit_args(sys.argv[2:]))
    elif stage == "manual-review":
        if len(sys.argv) != 2:
            raise SystemExit("manual-review does not accept arguments")
        build_manual_review_candidates()
    elif stage == "unique-pool":
        if len(sys.argv) != 2:
            raise SystemExit("unique-pool does not accept arguments")
        build_unique_qa_pool()
    else:
        raise SystemExit(f"Unknown dedup stage: {stage}")


if __name__ == "__main__":
    main()
