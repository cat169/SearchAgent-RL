# SearchAgent-RL

面向外部搜索工具的 LLM Agent：学习何时搜索、如何使用检索证据以及何时回答。

技术路线：NQ + HotpotQA → Local BM25 → DeepSeek Teacher → Qwen3-4B-Base Agentic SFT → veRL GRPO → Agent Evaluation。

## 当前状态

已完成 NQ / HotpotQA 数据处理、Local BM25、DeepSeek Teacher 轨迹生成、SFT 数据构建、选择性控制 Token + LoRA SFT，以及对应 Agent Evaluation 实现。

历史数据探索包含 AetherSearch；当前正式 SFT 数据只包含 NQ 与 HotpotQA，不再将 AetherSearch 纳入训练。正式数据为 Train 1,898、Validation 150；固定 Agent Evaluation 为 NQ test 100 + HotpotQA dev 100。

当前保留的是此前 SFT v2 的有效实现。veRL GRPO、Reward 和 GRPO rollout 接口尚未实现。正式训练及评测产物未同步到当前本地仓库，指标缺口详见 [SFT 实验记录](docs/sft_results.md)。

## 目录

```text
agent/          # 公共协议、Teacher AgentLoop、环境、搜索工具和 Prompt
retrieval/      # Local BM25 Retriever
teacher/        # DeepSeek API 适配器
training/       # 正式 SFT 训练入口
evaluation/     # 正式 Student Agent Evaluation
scripts/        # QA 处理、Teacher 生成、SFT 构建与 Messages 导出
tests/          # 协议、历史渲染、超长上下文及断点续评测试
docs/           # 正式实验记录和技术文档
data/           # 本地数据、语料和索引（Git 忽略）
requirements-sft.txt
```

所有命令从仓库根目录执行。训练环境为 Linux、Python 3.11、RTX 4090；SFT 依赖见 `requirements-sft.txt`。Teacher 另外需要 `openai`、`python-dotenv`；检索需要已有 Pyserini/Java 环境、Lucene 索引和 wiki-18 corpus，SFT 依赖文件并不包含完整检索环境。API Key 通过被忽略的根目录 `.env` 提供。

## 数据准备

```bash
python -m scripts.process_nq
python -m scripts.process_hotpotqa
python -m scripts.build_sft_dataset
python -m scripts.build_agent_eval_set
```

正式构建链：
```text
processed QA → 已冻结 Teacher candidates → DeepSeek raw → 已筛选 clean
→ build_sft_dataset → canonical Train/Validation + Messages
→ build_agent_eval_set → 固定 Eval200
```

`build_sft_dataset` 从现有 Clean Teacher 读取轨迹，仅接受原始 train ID，隔离 NQ dev/test、HotpotQA dev 的规范化问题，并调用唯一 Renderer。冻结数据仍沿用 `data/processed/sft/nq_hotpotqa_v1/` 和 `data/processed/evaluation/nq_hotpotqa_v1/` 路径；这是数据版本名，不代表使用旧训练实现，不要在整理代码时重建或改名。

单独导出 Messages：
```bash
python -m scripts.render_sft_messages \
  --input data/processed/sft/nq_hotpotqa_v1/train.jsonl \
  --output /tmp/train_messages.jsonl
```

Teacher 生成入口：
```bash
python -m scripts.generate_teacher_trajectories --limit 4000 --resume
```

该入口依赖 `data/processed/splits/teacher_sft_candidates.jsonl`，本地当前缺失；不加 `--resume` 会覆盖对应输出。原始候选构建及 Raw→Clean 筛选程序在当前工作区未找到，保留的数据及审计报告不能替代缺失代码。因此目前可从 Clean 重建正式 SFT，不能宣称原始 QA→Clean 全链已能重新运行。

## SFT 训练

```bash
python -m training.train_sft \
  --model-path /root/models/Qwen3-4B-Base \
  --train-path data/processed/sft/nq_hotpotqa_v1/messages/train_messages.jsonl \
  --val-path data/processed/sft/nq_hotpotqa_v1/messages/validation_messages.jsonl \
  --output-dir outputs/sft_v2
```

LoRA r=16、alpha=32、dropout=0.05、all-linear；选择性训练 `<think>`、`</think>`、`<|im_end|>`，保持输入/输出权重共享。BF16、SDPA、assistant-only loss、max_length=4096、packing=False。保留 `--limit`、`--max-steps`、`--check-only`、`--resume-from-checkpoint`。新训练必须使用空输出目录。

`outputs/sft_v2` 是沿用的产物路径约定；本地没有该 Adapter，实际服务器 Checkpoint 需由正式记录确认。

## Agent Evaluation

```bash
python -m evaluation.evaluate_agent \
  --eval-path data/processed/evaluation/nq_hotpotqa_v1/eval.jsonl \
  --base-model /root/models/Qwen3-4B-Base \
  --adapter-path /root/SearchAgent-RL/outputs/sft_v2 \
  --index-path /root/blockdata/SearchAgent-RL/retrieval/wiki-18-bm25-index/bm25 \
  --corpus-path /root/blockdata/SearchAgent-RL/retrieval/wiki-18-corpus/wiki-18.jsonl \
  --output-dir outputs/sft_v2_agent_evaluation \
  --max-new-tokens 512 \
  --max-input-tokens 4096
```

同一目录续评时追加 `--resume`；脚本校验已有结果是 Eval 连续前缀。支持 `--limit 5` 冒烟。每题立即保存到 `results.jsonl`，最终生成 `summary.md`。统计 EM/F1、Valid Action、Strict Raw Protocol、EOS、Answer Completion、平均 Search、Max Turns、Input Too Long。超长上下文不截断，按独立失败状态记零分并继续下一题。

## 验证与后续

```bash
python -m unittest discover -s tests -v
python -m training.train_sft --help
python -m evaluation.evaluate_agent --help
```

本地未找到正式训练 Loss、验证 Loss 和最终 Eval200 结果，当前不提供未经文件确认的效果数值。下一阶段计划基于确认后的正式 SFT Adapter 实现 veRL GRPO，并复用冻结 Eval200；目前没有可运行的 GRPO 训练入口。
