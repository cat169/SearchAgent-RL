from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.render_sft_messages import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical SFT records as conversational messages JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input.open("r", encoding="utf-8") as input_file, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            output = {"messages": build_messages(record)}
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Records: {count}")


if __name__ == "__main__":
    main()
