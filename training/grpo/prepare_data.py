"""将冻结的 GRPO QA 转换为保留原始消息的 veRL Parquet。"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from agent.system_prompt import AGENT_SYSTEM_PROMPT


DATA_DIR = Path("data/processed/grpo/nq_hotpotqa")
OUTPUT_DIR = DATA_DIR / "verl"


def to_verl_record(record: dict, split: str) -> dict:
    return {
        "data_source": record["source"],
        "prompt": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ],
        "ability": "qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": record["answers"],
        },
        "extra_info": {"id": record["id"], "split": split},
        "agent_name": "search_agent",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as input_file:
            records = [json.loads(line) for line in input_file if line.strip()]
        dataset = Dataset.from_list([to_verl_record(record, split) for record in records])
        output_path = OUTPUT_DIR / f"{split}.parquet"
        dataset.to_parquet(str(output_path))
        print(f"{split}: {len(dataset)} records -> {output_path}")


if __name__ == "__main__":
    main()
