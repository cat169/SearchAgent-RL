"""Build a read-only quality-review package for clean Teacher trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


INPUT_PATH = Path("data/processed/teacher/clean_teacher_trajectories.jsonl")
OUTPUT_DIR = Path("reports/teacher_quality_baseline")
TOKENIZER_NAME = "Qwen/Qwen3-4B-Base"
SEED = 123
RANDOM_PER_SOURCE = 30
TARGETED_COUNT = 20
THINK_CHAR_THRESHOLD = 4_000
THINK_TOKEN_THRESHOLD = 1_000
QUERY_JACCARD_THRESHOLD = 0.80

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "its", "their", "his", "her", "name", "called",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_terms(text: str) -> set[str]:
    return {word for word in normalize_text(text).split() if word not in STOPWORDS and len(word) > 1}


def jaccard(left: str, right: str) -> float:
    left_terms, right_terms = query_terms(left), query_terms(right)
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 1.0


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": max(values),
    }


def expected_event_types(search_count: int) -> list[str]:
    result: list[str] = []
    for _ in range(search_count):
        result.extend(["think", "search", "information"])
    result.extend(["think", "answer"])
    return result


def largest_remainder_quotas(populations: dict[str, int], total: int) -> dict[str, int]:
    population_total = sum(populations.values())
    exact = {key: total * count / population_total for key, count in populations.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(populations, key=lambda item: (-(exact[item] - quotas[item]), item)):
        if sum(quotas.values()) == total:
            break
        quotas[key] += 1
    return quotas


def add_flag(
    flags: list[dict], record: dict, risk_type: str, event_indices: list[int],
    related_text: list[dict], details: str,
) -> None:
    flags.append({
        "id": record["id"],
        "source": record["source"],
        "search_count": sum(event["type"] == "search" for event in record["events"]),
        "risk_type": risk_type,
        "event_indices": event_indices,
        "related_text": related_text,
        "details": details,
    })


def answer_match(record: dict) -> str:
    answers = [event["content"] for event in record["events"] if event["type"] == "answer"]
    if len(answers) != 1:
        return "unavailable"
    predicted = normalize_text(answers[0])
    golds = [normalize_text(answer) for answer in record["answers"]]
    return "normalized_exact" if predicted in golds else "not_normalized_exact"


def render_distribution_table(counter: Counter, keys: list) -> str:
    return "\n".join(f"| {key} | {counter.get(key, 0)} |" for key in keys)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    raw_lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines if line.strip()]
    if len(records) != 2_251:
        raise RuntimeError(f"Expected 2251 records, found {len(records)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Input IDs are not unique")
    raw_by_id = {record["id"]: line for record, line in zip(records, raw_lines)}
    record_by_id = {record["id"]: record for record in records}

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    source_counts = Counter(record["source"] for record in records)
    search_counts = Counter()
    source_search_counts: dict[str, Counter] = defaultdict(Counter)
    think_chars: list[int] = []
    think_tokens: list[int] = []
    per_record: dict[str, dict] = {}
    answer_matches = Counter()
    flags: list[dict] = []

    for record in records:
        events = record["events"]
        searches = [(index + 1, event["content"]) for index, event in enumerate(events) if event["type"] == "search"]
        informations = [(index + 1, event["content"]) for index, event in enumerate(events) if event["type"] == "information"]
        thinks = [(index + 1, event["content"]) for index, event in enumerate(events) if event["type"] == "think"]
        search_count = len(searches)
        char_count = sum(len(content) for _, content in thinks)
        token_count = sum(len(tokenizer.encode(content, add_special_tokens=False)) for _, content in thinks)
        per_record[record["id"]] = {
            "search_count": search_count,
            "think_chars": char_count,
            "think_tokens": token_count,
            "event_count": len(events),
        }
        search_counts[search_count] += 1
        source_search_counts[record["source"]][search_count] += 1
        think_chars.append(char_count)
        think_tokens.append(token_count)

        match_class = answer_match(record)
        answer_matches[match_class] += 1
        if match_class == "not_normalized_exact":
            answer_events = [
                {"event_index": index + 1, "event_type": event["type"], "text": event["content"]}
                for index, event in enumerate(events) if event["type"] == "answer"
            ]
            add_flag(flags, record, "answer_not_normalized_exact", [item["event_index"] for item in answer_events], answer_events,
                     "最终答案与任何 Gold 在 NFKC、大小写、标点和空白归一化后仍不完全一致；需要语义复核。")

        actual_types = [event["type"] for event in events]
        expected_types = expected_event_types(search_count)
        if actual_types != expected_types:
            add_flag(
                flags, record, "event_order_anomaly", list(range(1, len(events) + 1)),
                [{"event_index": index + 1, "event_type": event["type"], "text": event["content"]}
                 for index, event in enumerate(events)],
                f"Expected {expected_types}, observed {actual_types}.",
            )

        for index, content in informations:
            if not content.strip():
                add_flag(flags, record, "empty_information", [index],
                         [{"event_index": index, "event_type": "information", "text": content}],
                         "Search 后的 Information 内容为空。")

        long_related = [
            {"event_index": index, "event_type": "think", "text": content}
            for index, content in thinks
        ]
        if char_count > THINK_CHAR_THRESHOLD:
            add_flag(flags, record, "long_think_chars", [item["event_index"] for item in long_related], long_related,
                     f"Think 总字符数 {char_count} > {THINK_CHAR_THRESHOLD}。")
        if token_count > THINK_TOKEN_THRESHOLD:
            add_flag(flags, record, "long_think_tokens", [item["event_index"] for item in long_related], long_related,
                     f"Think 的 Qwen token 总数 {token_count} > {THINK_TOKEN_THRESHOLD}。")
        if search_count == 0 and (char_count > THINK_CHAR_THRESHOLD or token_count > THINK_TOKEN_THRESHOLD):
            add_flag(flags, record, "zero_search_long_think", [item["event_index"] for item in long_related], long_related,
                     f"零搜索轨迹的 Think 为 {char_count} 字符/{token_count} Qwen tokens；仅作为人工复核候选。")

        for left_pos in range(len(searches)):
            for right_pos in range(left_pos + 1, len(searches)):
                left_index, left_query = searches[left_pos]
                right_index, right_query = searches[right_pos]
                similarity = jaccard(left_query, right_query)
                related = [
                    {"event_index": left_index, "event_type": "search", "text": left_query},
                    {"event_index": right_index, "event_type": "search", "text": right_query},
                ]
                if normalize_text(left_query) == normalize_text(right_query):
                    add_flag(flags, record, "exact_repeated_search_query", [left_index, right_index], related,
                             "两个 Search Query 归一化后完全相同。")
                elif similarity >= QUERY_JACCARD_THRESHOLD:
                    add_flag(flags, record, "highly_similar_search_query", [left_index, right_index], related,
                             f"Query 词集合 Jaccard={similarity:.3f} >= {QUERY_JACCARD_THRESHOLD:.2f}。")

        # Lexical candidate only: does the next query introduce a content term seen in the immediately prior observation?
        if search_count >= 2:
            for current in range(1, len(searches)):
                previous_index, previous_query = searches[current - 1]
                current_index, current_query = searches[current]
                prior_information = next(
                    ((idx, text) for idx, text in informations if previous_index < idx < current_index), None
                )
                if prior_information is None:
                    continue
                info_index, info_text = prior_information
                introduced = query_terms(current_query) - query_terms(previous_query)
                evidence_terms = query_terms(info_text)
                if not (introduced & evidence_terms):
                    add_flag(
                        flags, record, "next_query_no_new_information_term_candidate",
                        [previous_index, info_index, current_index],
                        [
                            {"event_index": previous_index, "event_type": "search", "text": previous_query},
                            {"event_index": info_index, "event_type": "information", "text": info_text},
                            {"event_index": current_index, "event_type": "search", "text": current_query},
                        ],
                        "后续 Query 新增的内容词未在上一轮 Information 中词面出现；这只是多跳实体利用的启发式候选。",
                    )

    flag_counts = Counter(flag["risk_type"] for flag in flags)
    flags_by_id: dict[str, list[dict]] = defaultdict(list)
    for flag in flags:
        flags_by_id[flag["id"]].append(flag)

    rng = random.Random(SEED)
    random_entries: list[dict] = []
    strata_manifest: list[dict] = []
    for source in ("nq", "hotpotqa"):
        buckets: dict[str, list[str]] = {"0": [], "1": [], "2+": []}
        for record in records:
            if record["source"] != source:
                continue
            depth = per_record[record["id"]]["search_count"]
            bucket = str(depth) if depth < 2 else "2+"
            buckets[bucket].append(record["id"])
        quotas = largest_remainder_quotas({key: len(value) for key, value in buckets.items()}, RANDOM_PER_SOURCE)
        for bucket in ("0", "1", "2+"):
            population = sorted(buckets[bucket])
            quota = quotas[bucket]
            chosen = rng.sample(population, quota)
            strata_manifest.append({
                "source": source, "search_bucket": bucket, "population": len(population), "sample_size": quota,
                "inclusion_probability": quota / len(population), "shortfall": 0,
            })
            for uid in chosen:
                random_entries.append({
                    "id": uid, "source": source, "sample_group": "random", "search_bucket": bucket,
                    "search_count": per_record[uid]["search_count"],
                    "selection_reason": f"SEED={SEED} 的 source × search_bucket 分层随机抽样",
                    "inclusion_probability": quota / len(population),
                })

    random_ids = {entry["id"] for entry in random_entries}
    weights = {
        "event_order_anomaly": 100, "empty_information": 100, "answer_not_normalized_exact": 80,
        "zero_search_long_think": 50, "exact_repeated_search_query": 45,
        "highly_similar_search_query": 35, "long_think_tokens": 30, "long_think_chars": 20,
        "next_query_no_new_information_term_candidate": 15,
    }

    def risk_key(uid: str) -> tuple:
        risks = flags_by_id[uid]
        score = sum(weights[flag["risk_type"]] for flag in risks)
        return (-score, -per_record[uid]["think_tokens"], -per_record[uid]["search_count"], uid)

    eligible_by_source: dict[str, list[str]] = {}
    for source in ("nq", "hotpotqa"):
        eligible_by_source[source] = sorted(
            (uid for uid in flags_by_id if uid not in random_ids and record_by_id[uid]["source"] == source),
            key=risk_key,
        )
    targeted_ids = eligible_by_source["nq"][:5] + eligible_by_source["hotpotqa"][:5]
    remaining = sorted(
        (uid for uid in flags_by_id if uid not in random_ids and uid not in targeted_ids), key=risk_key
    )
    targeted_ids.extend(remaining[: TARGETED_COUNT - len(targeted_ids)])
    if len(targeted_ids) != TARGETED_COUNT:
        raise RuntimeError(f"Only {len(targeted_ids)} targeted candidates were available")
    targeted_entries = []
    for uid in targeted_ids:
        risk_types = sorted({flag["risk_type"] for flag in flags_by_id[uid]})
        targeted_entries.append({
            "id": uid, "source": record_by_id[uid]["source"], "sample_group": "targeted_risk",
            "search_bucket": str(per_record[uid]["search_count"]) if per_record[uid]["search_count"] < 2 else "2+",
            "search_count": per_record[uid]["search_count"],
            "selection_reason": "定向风险优先级：" + ", ".join(risk_types),
            "risk_types": risk_types,
            "inclusion_probability": None,
        })

    selections = random_entries + targeted_entries
    selected_ids = [entry["id"] for entry in selections]
    if len(selections) != 80 or len(set(selected_ids)) != 80:
        raise RuntimeError("Selection must contain exactly 80 unique IDs")
    if Counter(entry["sample_group"] for entry in selections) != {"random": 60, "targeted_risk": 20}:
        raise RuntimeError("Selection group sizes are incorrect")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_path = OUTPUT_DIR / "risk_flags.jsonl"
    risk_path.write_text("".join(json.dumps(flag, ensure_ascii=False) + "\n" for flag in flags), encoding="utf-8")
    selected_path = OUTPUT_DIR / "selected_80.jsonl"
    selected_path.write_text("".join(raw_by_id[uid] + "\n" for uid in selected_ids), encoding="utf-8")

    manifest = {
        "input": str(INPUT_PATH).replace("\\", "/"),
        "input_sha256": sha256(INPUT_PATH),
        "input_count": len(records),
        "seed": SEED,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "random_sampling": {
            "design": "source × search_bucket proportional stratified simple random sampling without replacement",
            "allocation": "largest remainder within each source; source totals fixed at 30 each",
            "strata": strata_manifest,
            "shortfall_handling": "No shortfall occurred; if quota exceeded population the script would fail rather than duplicate.",
        },
        "targeted_sampling": {
            "count": TARGETED_COUNT,
            "design": "risk-weight ordering after random IDs are excluded; minimum top five candidates per source, then global priority",
            "warning": "Targeted samples are not probability samples and must not estimate corpus error rates.",
        },
        "risk_thresholds": {
            "think_chars": THINK_CHAR_THRESHOLD,
            "think_qwen_tokens": THINK_TOKEN_THRESHOLD,
            "query_word_set_jaccard": QUERY_JACCARD_THRESHOLD,
        },
        "selections": selections,
    }
    manifest_path = OUTPUT_DIR / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manual_lines = [
        "# Teacher Clean 轨迹人工审查（80 条）", "",
        "> 随机样本与定向风险样本必须分开解释；风险标签不是质量失败结论。", "",
        "> 允许填写：PASS / REVIEW / FAIL / UNCERTAIN / NOT_APPLICABLE。当前所有评分均留空。", "",
    ]
    for number, entry in enumerate(selections, 1):
        record = record_by_id[entry["id"]]
        answer_events = [event["content"] for event in record["events"] if event["type"] == "answer"]
        manual_lines.extend([
            f"## {number:02d}. {record['id']}", "",
            f"- Sample group: `{entry['sample_group']}`",
            f"- Source: `{record['source']}`",
            f"- Search count: `{entry['search_count']}`",
            f"- Selection reason: {entry['selection_reason']}",
            f"- Question: {record['question']}",
            f"- Gold answers: `{json.dumps(record['answers'], ensure_ascii=False)}`",
            f"- Teacher final answer: `{json.dumps(answer_events, ensure_ascii=False)}`", "",
            "### 完整 Events", "",
        ])
        for event_index, event in enumerate(record["events"], 1):
            manual_lines.extend([
                f"#### Event {event_index}: {event['type']}", "", "```text", event["content"], "```", "",
            ])
        manual_lines.extend([
            "### 人工评分", "",
            "| 维度 | 评分 | 事件编号与具体证据 |", "|---|---|---|",
            "| Answer Correctness |  |  |", "| Search Decision |  |  |", "| Query Quality |  |  |",
            "| Evidence Grounding |  |  |", "| Reasoning Quality |  |  |",
            "| Overall SFT Suitability |  |  |", "", "---", "",
        ])
    manual_path = OUTPUT_DIR / "manual_review_80.md"
    manual_path.write_text("\n".join(manual_lines), encoding="utf-8")

    selected_source = Counter(entry["source"] for entry in selections)
    selected_depth = Counter(entry["search_count"] for entry in selections)
    targeted_source = Counter(entry["source"] for entry in targeted_entries)
    targeted_depth = Counter(entry["search_count"] for entry in targeted_entries)
    source_depth_rows = []
    for source in ("nq", "hotpotqa"):
        row = [source] + [source_search_counts[source].get(depth, 0) for depth in range(5)]
        source_depth_rows.append("| " + " | ".join(map(str, row)) + " |")
    summary = f"""# Teacher Clean 全量自动风险扫描

## 数据身份

- 输入：`{INPUT_PATH.as_posix()}`
- 记录数：**{len(records)}**
- SHA256：`{sha256(INPUT_PATH)}`
- Tokenizer：`{TOKENIZER_NAME}`（`{tokenizer.__class__.__name__}`，仅对 Think 原文计 token）

## 全量分布

| Source | Count |
|---|---:|
| NQ | {source_counts['nq']} |
| HotpotQA | {source_counts['hotpotqa']} |

| Source | Search 0 | Search 1 | Search 2 | Search 3 | Search 4 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(source_depth_rows)}
| Total | {search_counts[0]} | {search_counts[1]} | {search_counts[2]} | {search_counts[3]} | {search_counts[4]} |

Think 字符分布：`{json.dumps(describe(think_chars), ensure_ascii=False)}`  
Think Qwen token 分布：`{json.dumps(describe(think_tokens), ensure_ascii=False)}`

## Answer 与 Gold 的自动比较

- normalized exact：{answer_matches['normalized_exact']}
- 非 normalized exact、需语义复核：{answer_matches['not_normalized_exact']}
- 无法比较：{answer_matches['unavailable']}

比较仅做 NFKC、casefold、标点移除和空白合并；字符串不匹配不是答案错误，匹配也不证明证据充分。

## 风险信号

| Risk type | Flags |
|---|---:|
{render_distribution_table(flag_counts, sorted(flag_counts))}

- 被至少一个规则标记的唯一轨迹：**{len(flags_by_id)}**
- Flag 总数：**{len(flags)}**（同一轨迹可有多个）
- `long_think_chars`：Think 总字符数 > {THINK_CHAR_THRESHOLD}。
- `long_think_tokens`：Think 原文的 Qwen token 总数 > {THINK_TOKEN_THRESHOLD}。
- `highly_similar_search_query`：同一轨迹两个 Query 的内容词集合 Jaccard >= {QUERY_JACCARD_THRESHOLD:.2f}。
- `next_query_no_new_information_term_candidate`：后续 Query 新增内容词未在上一轮 Information 中词面出现；它尤其容易误报同义词、代词和常识桥接。
- 零搜索、长 Think、Query 词面相似都只是风险候选，不是失败判定。
- `normalized_exact` 只检查答案字符串，不检查 Evidence Grounding。

## 80 条抽样

- 分层随机：60（NQ 30，HotpotQA 30）。
- 定向风险：20（NQ {targeted_source['nq']}，HotpotQA {targeted_source['hotpotqa']}）。
- 全部样本来源：NQ {selected_source['nq']}，HotpotQA {selected_source['hotpotqa']}。
- 全部样本搜索深度：{dict(sorted(selected_depth.items()))}。
- 定向样本搜索深度：{dict(sorted(targeted_depth.items()))}。
- 具体 ID、抽样概率、风险理由见 `sampling_manifest.json`；完整原文见 `selected_80.jsonl` 和 `manual_review_80.md`。

## 解释限制

自动规则没有进行事实核验、语义等价判断或证据链正确性判定。20 条定向样本被刻意富集风险，不能用于估计全量错误率；60 条随机样本才保留已记录的概率抽样设计。
"""
    summary_path = OUTPUT_DIR / "automated_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    # Verify exact source-line preservation and complete event inclusion before packaging.
    selected_written = selected_path.read_text(encoding="utf-8").splitlines()
    if selected_written != [raw_by_id[uid] for uid in selected_ids]:
        raise RuntimeError("selected_80.jsonl does not exactly preserve input lines")
    manual_text = manual_path.read_text(encoding="utf-8")
    for uid in selected_ids:
        for event in record_by_id[uid]["events"]:
            if event["content"] not in manual_text:
                raise RuntimeError(f"Event content missing from manual report: {uid}")

    zip_path = OUTPUT_DIR / "quality_review_package.zip"
    members = [summary_path, risk_path, manifest_path, manual_path, selected_path]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(path.relative_to(OUTPUT_DIR).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(zip_path) as archive:
        if sorted(archive.namelist()) != sorted(path.name for path in members):
            raise RuntimeError("ZIP member validation failed")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")

    print(f"Input count: {len(records)}")
    print(f"Input SHA256: {sha256(INPUT_PATH)}")
    print(f"Risk flags: {len(flags)} across {len(flags_by_id)} records")
    print(f"Random samples: {len(random_entries)}")
    print(f"Targeted samples: {len(targeted_entries)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
