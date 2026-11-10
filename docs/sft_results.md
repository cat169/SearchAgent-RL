# 正式 SFT 实验记录

核查范围：2026-11-10 当前本地工作区；未连接服务器。正式代码沿用此前 SFT v2，实现迁移不改变训练参数、解析协议、搜索行为或指标定义。

## 模型与训练实现

- Base：`Qwen/Qwen3-4B-Base`，本地模型目录约定 `/root/models/Qwen3-4B-Base`。
- 正式入口：`python -m training.train_sft`。
- LoRA：r=16、alpha=32、dropout=0.05、target_modules=all-linear、bias=none。
- 选择性 Token：`<think>` 151667、`</think>` 151668、`<|im_end|>` 151645；Tokenizer 实际验证 ID；`ensure_weight_tying=True`，初始化后检查 delta、可训练范围及输入/输出关联。
- BF16、SDPA；assistant_only_loss=True、packing=False、max_length=4096。
- Train/Eval batch size=1；梯度累积=8；gradient_checkpointing=True。
- AdamW、learning_rate=1e-4、2 epochs、seed/data_seed=123。
- 每 10 steps 日志，每 100 steps 验证与保存，最多保留 2 个 Checkpoint；TensorBoard 输出到训练目录的 `runs/`。
- 精确依赖版本见根目录 `requirements-sft.txt`。以上是现存代码配置，并非服务器日志的替代。

## 正式数据版本

依据本地 `data/processed/sft/nq_hotpotqa_v1/split_manifest.json`，以及对应文件实际数量和 SHA256：

| 文件 | 数量 | SHA256 |
|---|---:|---|
| train.jsonl | 1898 | ca406e4b63c15765ccc69f194e80866ceeb9c81d2aecdbb68655c4f1e2fcec45 |
| validation.jsonl | 150 | 2aa8c0e657210dc424011ca4f9e94b7b02537a89b30decd371c0c3364bf33823 |
| messages/train_messages.jsonl | 1898 | 0f4b514fa628cb4e3fa9100208a4d35a1b98b1aaf23f2094a2504672e06a6040 |
| messages/validation_messages.jsonl | 150 | a544cd91a4365673b4b2bbf7c383caedf7a78a7f8577fce187de59c69535b9e8 |

| 集合 | NQ | HotpotQA | 0-search | 1-search | 2-search | 3-search | 4-search |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 729 | 1169 | 1049 | 477 | 256 | 80 | 36 |
| Validation | 58 | 92 | 79 | 38 | 23 | 6 | 4 |

当前正式数据不含 AetherSearch。历史 AetherSearch 探索不作为本次实验结果。`nq_hotpotqa_v1` 是冻结数据版本名称，保留原路径及 Manifest。

Clean 输入 2,251 条（SHA256 `6f52e51efef8a22648f6b0dea2a87b46733b4a23f1cc47db0faca901dab699a5`）。构建记录保留原始 train 来源 2,049 条，排除 1 条缺少 Think 的 Renderer 不兼容样本 `nq_train_37781`，得到 2,048 条，再按 seed=123 分出 150 条 Validation。与保留 NQ dev/test、HotpotQA dev 的规范化 Question 重叠为 0；这不证明语义近重复全部消除。

## 训练命令与 Checkpoint

整理后的等价调用命令如下；本地没有保留当次服务器 Shell 命令，不能认定此文本是历史执行日志：

```bash
python -m training.train_sft \
  --model-path /root/models/Qwen3-4B-Base \
  --train-path data/processed/sft/nq_hotpotqa_v1/messages/train_messages.jsonl \
  --val-path data/processed/sft/nq_hotpotqa_v1/messages/validation_messages.jsonl \
  --output-dir outputs/sft_v2
```

当前评测默认 Adapter 路径为 `/root/SearchAgent-RL/outputs/sft_v2`。它是代码默认值，本地没有 Adapter 或 `trainer_state.json`，无法确认服务器最终 Checkpoint 的实际位置、476-step 完成状态、训练/验证 Loss 和 Gradient Norm。不得把该默认路径视为已核验产物。后续 GRPO 应加载该正式 SFT 实验经确认的 Adapter；目前尚无可核验的本地权重或权重哈希。

## 固定 Agent Evaluation

本地数据：`data/processed/evaluation/nq_hotpotqa_v1/eval.jsonl`，200 条，NQ 原始 test 100 + HotpotQA 原始 dev 100。SHA256：`11eee19490769c23dc2cb2cda09046a8bcb65786708dd0cdaa9954b209d13e1a`。

`eval_manifest.json` 记录 seed=123、选中 ID 和输入哈希；Eval 内部及其与 SFT Train/Validation 的 ID、规范化 Question 重叠均为 0。

正式入口：`python -m evaluation.evaluate_agent`，完整命令见根目录 README。保留 Top-K=3、MAX_TURNS=5、greedy、max_new_tokens=512、max_input_tokens=4096；Tokenizer 从 Adapter 目录加载实际保存的 Chat Template。独立维护 reasoning_content 与 action content；Environment 只接收 Action。

结果保存为调用者 `--output-dir` 下 `results.jsonl` 和 `summary.md`。支持连续前缀校验后 `--resume`。超长输入记为 `input_too_long`，EM/F1=0，保留成功回合并继续评测。

| 正式指标 | 本地记录状态 |
|---|---|
| EM / F1 | 未找到最终结果，不能填写 |
| Valid Action / Strict Raw Protocol / EOS | 未找到最终结果，不能填写 |
| Answer Completion / Average Search Count | 未找到最终结果，不能填写 |
| Max Turns / Input Too Long | 未找到最终结果，不能填写 |

当前工作区不存在 `outputs/`、`experiments/` 或最终评测文件；未检查远程服务器。本轮没有重新计算模型指标。

## 已确认问题及复现边界

旧训练入口未直接训练三个控制 Token，已由当前选择性 Token 实现代替。旧评测的动作-only strict 检查、整段输出回灌方式已由当前原始协议与结构化历史实现替代；这些是代码差异，不作为训练效果的独立证明。

原始 Raw 3,982 条、Clean 2,251 条及筛选报告本地仍保留。`teacher_filter_report.json` 记录 1,731 条拒绝（non_answer=386、correctness_mismatch=1345），但对应 audit/filter 程序当前缺失，不能声称完整 QA→Clean 流程已可复现。Teacher 生成所需的 4,000 条 candidates 文件也未在本地找到。保留当前 Teacher 入口用于依赖齐备后的轨迹生成；不在本轮重建任何候选或数据。

当前能够从现存 Clean 通过 `scripts.build_sft_dataset` 重建正式数据，并用 `scripts.build_agent_eval_set` 重建固定评测。业务代码、Prompt、Canonical、Messages、语料与索引均在本轮保持原内容。

## GRPO 状态

veRL GRPO、Reward、Student rollout 训练接口尚未实现。后续使用确认后的正式 SFT Adapter 和冻结 Eval200；本次整理没有新增 GRPO 代码，也没有改变实验条件。
