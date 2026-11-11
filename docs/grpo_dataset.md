# 正式 GRPO 数据划分

冻结日期：2026-11-11。版本：`nq_hotpotqa_grpo_20261111`。

这是基于现有原始 train QA 新建的数据版本，不是历史 GRPO 分区的恢复。历史成员及人工去重输入缺失，不能精确恢复。SFT 和固定 Eval200 保持不变。

## 文件与规模

所有输出位于 `data/processed/grpo/nq_hotpotqa/`，由 Git 忽略并保留在本地。

| 文件 | NQ | HotpotQA | 总数 |
|---|---:|---:|---:|
| train.jsonl | 1000 | 1000 | 2000 |
| validation.jsonl | 100 | 100 | 200 |

每条仅包含 `id`、`source`、`source_split`、`question`、`answers`。ID 与答案分别来自原始 `uid` 和 `gold_answers`，没有 Teacher 轨迹或 SFT Messages。全部来源 split 为原始 train。GRPO 训练将在线生成 Agent 轨迹，本次只准备 QA。

输入为 `data/processed/nq/train.jsonl` 和 `data/processed/hotpotqa/train.jsonl`。

## 隔离与确定性抽样

复用 `scripts.build_sft_dataset.normalize_question`：Unicode NFKC、casefold、连续空白合并、首尾去空白。

先排除与以下全部文件存在相同 ID 或规范化 Question 的候选：

- `data/processed/sft/nq_hotpotqa_v1/train.jsonl`
- `data/processed/sft/nq_hotpotqa_v1/validation.jsonl`
- `data/processed/evaluation/nq_hotpotqa_v1/eval.jsonl`
- `data/processed/nq/dev.jsonl`
- `data/processed/nq/test.jsonl`
- `data/processed/hotpotqa/dev.jsonl`

按 NQ、HotpotQA 顺序处理，候选按 UID 排序，规范化问题去重保留首条；也排除跨来源规范化问题重复。各来源独立使用 `random.Random(123).shuffle()`；前 1000 条用于 Train，随后 100 条用于 Validation。输出按来源顺序拼接，保留各自的抽样顺序。

Train/Validation 内部和相互之间、以及相对于全部保留集合的 ID 与规范化 Question 重叠均为 0。该检查不代表语义近重复风险全部消除，也不代表答案已完成逐题事实核验。

## 哈希与复现

| 文件 | SHA256 |
|---|---|
| train.jsonl | a384f752318bf33c314209a8e94dee381693ab4c15f40a1b691a45059596dac4 |
| validation.jsonl | 34f655ab58631c5c1ebdc7b8baa38b53dcf19ad40502e5b3f28be048fed1b3fb |

`split_manifest.json` 保存输入路径和哈希、抽样规则、种子、选中 ID（按输出顺序）、来源数量、输出哈希及必要隔离核验结果。没有时间戳或随机运行信息，内容可以确定性重建。

从仓库根目录运行：

```bash
python -m scripts.build_grpo_dataset
```

脚本直接检查输出配额、ID/问题唯一性、集合隔离、相同输入与种子的重复计算结果及输入哈希未变化。该命令会写入上述固定输出路径；本次没有创建额外 QA Test，最终 Evaluation 继续使用已冻结的 Eval200（NQ test 100 + HotpotQA dev 100）。

本次未实现 veRL、模型加载、Reward 或训练入口，也未修改现有 SFT 实验记录。
