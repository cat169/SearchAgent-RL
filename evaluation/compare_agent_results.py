"""Pair unified HF evaluation results by exact question identity."""
import argparse
import json
from pathlib import Path
from collections import Counter
from evaluation.evaluate_hf_agent import grouped


def compare(left, right):
    def index(rows):
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate IDs")
        return result
    a,b = index(left), index(right)
    if a.keys() != b.keys():
        raise ValueError("Evaluation ID sets differ")
    pairs=[]
    counts=Counter()
    for key,x in a.items():
        y=b[key]
        for field in ("question", "gold_answers", "source"):
            if x[field] != y[field]:
                raise ValueError(f"Question identity mismatch: {key}: {field}")
        counts[f"sft_{int(x['em'])}_grpo_{int(y['em'])}"]+=1
        pairs.append(dict(id=key, source=x["source"],
            sft_prediction=x["predicted_answer"], grpo_prediction=y["predicted_answer"],
            sft_search_count=x["search_count"], grpo_search_count=y["search_count"],
            search_delta=y["search_count"]-x["search_count"],
            sft_termination=x["termination_reason"], grpo_termination=y["termination_reason"],
            sft_em=x["em"], grpo_em=y["em"]))
    ga,gb=grouped(left),grouped(right)
    return dict(sft=ga,grpo=gb,delta={g:{k:gb[g][k]-ga[g][k] for k in ga[g]} for g in ga},
        correctness_pairs={k:counts[k] for k in ("sft_0_grpo_0","sft_0_grpo_1","sft_1_grpo_0","sft_1_grpo_1")},
        paired_results=pairs,
        differences=[p for p in pairs if p["sft_prediction"]!=p["grpo_prediction"] or p["search_delta"] or p["sft_termination"]!=p["grpo_termination"]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sft",type=Path); p.add_argument("grpo",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    read=lambda path:[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result=compare(read(args.sft),read(args.grpo))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# SFT / GRPO paired comparison", "", "Absolute differences are GRPO minus SFT. Search reduction alone does not establish improvement.", "", "|Group|Metric|SFT|GRPO|Delta|", "|---|---|---:|---:|---:|"]
    for group in result["sft"]:
        for metric,value in result["sft"][group].items():
            lines.append(f"|{group}|{metric}|{value:.4f}|{result['grpo'][group][metric]:.4f}|{result['delta'][group][metric]:+.4f}|")
    lines += ["", "## Correctness pairs", "", json.dumps(result["correctness_pairs"]), "", "## Different outcomes", "", "```json", json.dumps(result["differences"],indent=2), "```"]
    (args.output_dir/"comparison.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
