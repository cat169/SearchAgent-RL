# SearchAgent-RL 项目驱动学习路线

> 用途：作为 SearchAgent-RL 后续学习与项目推进的长期对齐文档，避免对话和学习内容跑偏。
>
> 核心原则：**以项目下一步要解决的问题为主线，只学习完成当前工程任务所必需的知识；学懂后立即落地到项目实现，再通过代码、实验和复盘巩固知识。**
>
> 项目固定路线：
>
> ```text
> QA Data
> → Data Processing / Dedup / Split
> → Local BM25 Retriever
> → DeepSeek Teacher Trajectory Generation
> → Teacher Quality Filtering
> → Agentic SFT
> → Qwen3 Online Rollout
> → veRL GRPO
> → Evaluation
> ```

---

## 1. 总体学习方法

后续不采用“先系统学完所有 LLM / SFT / RL 理论，再做项目”的方式。

固定采用下面的闭环：

```text
项目当前阶段
    ↓
明确下一步工程任务
    ↓
确定当前真正需要掌握的知识
    ↓
先学习必要知识
    ↓
自己理解关键设计
    ↓
再实现代码
    ↓
审代码 / 查结果 / 做实验
    ↓
复盘这一阶段的知识与设计决策
    ↓
进入下一阶段
```

也就是说：

```text
项目需要 renderer
→ 学 renderer 所需知识
→ 做 renderer

项目需要 SFT
→ 学 SFT 所需知识
→ 做 SFT

项目需要 evaluation
→ 学 evaluation 所需知识
→ 做 evaluation

项目需要 GRPO
→ 学 RL / GRPO 所需知识
→ 做 GRPO
```

这个路线的目标不是“尽快让代码跑起来”，而是：

1. 真正完成 SearchAgent-RL；
2. 通过项目掌握大模型 Agent / Post-training 的核心知识；
3. 能解释自己的设计、代码、实验和排查过程；
4. 最终能用于实习投递与面试。

---

## 2. 当前项目状态

目前已经完成：

```text
NQ / HotpotQA / AetherSearch 数据处理        ✓
统一 QA schema                               ✓
跨数据源去重                                 ✓
Teacher / GRPO / Val / Test 划分             ✓
Local BM25 Retriever                         ✓
DeepSeek Teacher trajectory                  ✓
Teacher audit                                ✓
Teacher deterministic filtering              ✓
AetherSearch dedup                            ✓
统一 SFT trajectory 数据                      ✓
Narrow / Standard SFT 数据划分               ✓
Qwen3 tokenizer 基础学习                      ✓
Causal LM next-token prediction 基础          ✓
shift logits / labels                         ✓
labels=-100                                   ✓
attention mask / causal mask / loss mask 区分  ✓
```

当前所在位置：

```text
SFT 数据已经准备好
        ↓
下一步：Agentic SFT Renderer
```

---

# 第一阶段：Agentic SFT Renderer

## 3. 阶段目标

把当前已经准备好的 trajectory：

```json
{
  "question": "...",
  "events": [
    {"type": "think", "content": "..."},
    {"type": "search", "content": "..."},
    {"type": "information", "content": "..."},
    {"type": "think", "content": "..."},
    {"type": "answer", "content": "..."}
  ]
}
```

转换为 Qwen3 可以用于 SFT 的训练样本：

```text
trajectory
↓
messages
↓
chat template
↓
formatted text
↓
tokenizer
↓
input_ids
↓
labels
↓
attention_mask
```

最终得到类似：

```python
{
    "input_ids": [...],
    "labels": [...],
    "attention_mask": [...]
}
```

---

## 4. Renderer 之前要学完的知识

在真正实现 renderer 之前，只学习下面四块。

### 4.1 Trajectory → Messages → Chat Template

需要理解：

```text
question
think
search
information
answer
```

如何映射成：

```text
system
user
assistant
```

需要搞懂：

```text
message 是什么
role 是什么
content 是什么
Chat Template 是什么
apply_chat_template() 做什么
```

重点问题：

```text
为什么 question 属于 user/context
为什么 think/search/answer 属于 assistant behavior
为什么 information 是 Retriever / environment observation
为什么 information 不应该被当成模型自己生成的内容
为什么不能简单地“一条 event = 一条 message”
```

这一部分完成后，需要能够手工构造一条 SearchAgent trajectory 对应的 `messages`。

### 4.2 Token-level Loss Mask

当前冻结原则：

```text
Question
→ 模型需要看到
→ 不学习生成
→ labels = -100

Information
→ 模型需要看到
→ 不学习生成
→ labels = -100

Think
→ 模型行为
→ 参与 loss

Search
→ 模型行为
→ 参与 loss

Answer
→ 模型行为
→ 参与 loss
```

但实现前还需要进一步明确：

```text
ChatML role token 算不算 loss？
<|im_start|> / <|im_end|> 怎么处理？
assistant 前缀是否监督？
<think> / </think> 是否监督？
<search> / </search> 是否监督？
<information> 标签和正文怎么 mask？
<answer> / </answer> 如何处理？
```

最终要把“assistant-only loss”从概念层落实到 **token 级别**。

### 4.3 Padding / Truncation / Max Sequence Length

需要理解：

```text
不同 trajectory 长度为什么不同
batch 为什么需要 padding
padding token 怎么处理
PAD 为什么 labels=-100
attention_mask 怎么配合 padding
max_seq_length 是什么
truncation 是什么
```

SearchAgent 特别需要处理的问题：

```text
question
→ think
→ search
→ information
→ think
→ search
→ information
→ answer
```

如果序列太长被截断：

```text
answer 可能消失
```

因此不能机械地采用普通聊天 SFT 的截断策略。

Renderer 实现后必须统计 Narrow / Standard 的真实 token 长度：

```text
mean
median
P90
P95
P99
max
```

再决定 `max_seq_length`。

### 4.4 DataCollator

需要真正理解 DataCollator 的作用：

```text
单条样本
{
  input_ids,
  attention_mask,
  labels
}

        ↓

DataCollator
        ↓

动态 padding
        ↓

batch tensor
```

最终 tensor 形状：

```text
input_ids      [B, L]
attention_mask [B, L]
labels         [B, L]
```

需要理解：

```text
为什么 DataCollator 不是 tokenizer
为什么 DataCollator 不是模型
为什么单条样本长度可以不同
为什么 batch 内最终必须对齐
```

---

## 5. Renderer 阶段实施顺序

知识学完后立刻进入项目实现：

```text
1. 确定 trajectory → messages 的规则
2. 确定 Qwen3 Chat Template 使用方式
3. 确定 token-level loss mask
4. 实现 renderer
5. 拿真实 narrow / standard 样本测试
6. 逐 token 检查 input_ids / labels
7. 验证 information 是否完全 mask
8. 验证 think/search/answer 是否正确监督
9. 统计 token 长度
10. 确定 max_seq_length
```

完成标准不是“脚本能跑”。

完成标准是：

> 能拿一条真实 trajectory，清楚解释每一段为什么进入 `input_ids`、为什么某些 token 是 `-100`、为什么某些 token 参与 loss。

---

# 第二阶段：正式 SFT 训练

## 6. 阶段目标

Renderer 正确后，进入真正的 Qwen3-4B-Base Agentic SFT。

这时才开始学习训练本身。

---

## 7. SFT 训练前需要学习的知识

### 7.1 Model Forward

需要理解：

```text
input_ids
↓
Embedding
↓
Transformer
↓
hidden states
↓
LM Head
↓
logits
↓
shift
↓
Cross Entropy
↓
loss
```

已经学习过：

```text
next-token prediction
shift logits / labels
labels=-100
```

这时会结合真实 Qwen forward 再完整串一次。

### 7.2 Batch / Step / Epoch

需要真正搞懂：

```text
sample
batch
batch size
training step
epoch
gradient accumulation
effective batch size
```

并能计算：

```text
数据量
batch size
gradient accumulation
epoch
→ 总 training steps
```

### 7.3 Optimizer 与 Learning Rate

学习：

```text
AdamW
learning rate
weight decay
warmup
scheduler
gradient clipping
```

目标不是推导优化理论，而是理解这些参数各自控制什么，以及调大 / 调小通常会产生什么影响。

### 7.4 BF16 / FP16 / FP32

理解：

```text
参数精度
计算精度
显存
数值稳定性
```

并结合实际租用 GPU 决定训练精度。

### 7.5 LoRA 与 Full Fine-tuning

这时再正式决定：

```text
LoRA
还是
Full Fine-tuning
```

需要学习：

```text
W' = W + ΔW
ΔW = BA
```

理解：

```text
为什么冻结原模型参数
为什么只训练低秩矩阵
rank 是什么
alpha 是什么
target modules 是什么
```

最终根据：

```text
Qwen3-4B
GPU
Narrow / Standard token budget
训练成本
实验目标
```

做决定。

不要在 renderer 之前提前拍脑袋确定。

### 7.6 Checkpoint 与训练状态

理解：

```text
model checkpoint
optimizer state
scheduler state
trainer state
resume training
```

避免以后“有模型文件但恢复不了训练”的问题。

---

## 8. SFT 正式实验设计

两个实验必须从同一个 Base model 出发：

```text
Qwen3-4B-Base
      │
      ├── Narrow SFT
      │
      └── Standard SFT
```

当前冻结：

```text
Narrow:
2833 samples
search_count <= 1

Standard:
2833 samples
相同 source 数量
包含全部 deep trajectory
```

必须关注一个潜在实验混杂：

```text
样本数相同
≠
训练 token 数相同
```

Standard trajectory 更深，可能平均更长。

所以训练前要比较：

```text
总 token 数
总 supervised token 数
平均 sequence length
```

必要时决定是否需要控制 training budget。

---

# 第三阶段：SFT Evaluation

## 9. 为什么 SFT 训练完成不能直接进入 GRPO

必须先回答：

> Narrow 和 Standard 到底学到了什么？

否则直接进入 GRPO，后面无法判断提升来自 SFT 还是 RL。

---

## 10. Evaluation 前要学习的知识

学习：

```text
model.generate()
greedy decoding
sampling
temperature
top_p
max_new_tokens
EOS
stop condition
```

理解训练和生成的区别：

```text
Training:
teacher forcing

Inference:
autoregressive generation
```

---

## 11. Base / Narrow / Standard 对比

至少比较：

```text
Base Qwen3-4B
Narrow-SFT
Standard-SFT
```

主要指标：

```text
answer correctness
answer rate
invalid action rate
max-turn rate
search_count
zero-search rate
multi-search rate
format validity
source breakdown
```

核心研究问题：

> SFT 中不同搜索深度行为覆盖，是否改变模型的 Agent 行为和后续 RL 起点？

---

# 第四阶段：Online Rollout 与 GRPO

## 12. 进入 GRPO 的前提

只有当：

```text
Renderer 正确
SFT 正常收敛
Narrow / Standard 能正常 rollout
Search action 格式稳定
Retriever interaction 正常
Evaluation 跑通
```

以后，才进入 GRPO。

---

## 13. GRPO 阶段再学习的知识

不要提前一次学完强化学习。

届时按工程需要学习：

```text
policy
rollout
reward
trajectory
advantage
group sampling
GRPO objective
KL regularization
old policy / reference model
online environment interaction
```

然后再进入 veRL。

---

## 14. Search-R1 / veRL 需要重点理解什么

Search-R1 主要作为 RL / Search-Agent 阶段参考。

重点不是机械复制仓库，而是理解：

```text
prompt
↓
Qwen rollout
↓
search action
↓
Retriever
↓
observation
↓
继续 rollout
↓
answer
↓
reward
↓
GRPO update
```

当前项目保持：

```text
DeepSeek Teacher trajectory
→ 只服务于 SFT

GRPO trajectory
→ Qwen3 在线 rollout
```

不要把 Teacher trajectory 直接当成 GRPO rollout。

---

# 第五阶段：GRPO Evaluation 与项目收尾

## 15. 最终实验结构

理想的完整实验链：

```text
Qwen3-4B-Base
      │
      ├── Narrow SFT
      │      ↓
      │   Narrow + GRPO
      │
      └── Standard SFT
             ↓
          Standard + GRPO
```

这样可以回答：

```text
SFT behavior coverage 是否重要？
深搜索 demonstration 是否改变初始 agent policy？
GRPO 是否能进一步提升 search behavior？
不同 SFT 起点经过 GRPO 后是否仍存在差异？
```

---

## 16. 最终需要掌握的项目能力

项目完成后，不要求记住所有框架内部代码。

但必须能解释：

```text
数据为什么这样处理
为什么需要 dedup
为什么 AetherSearch 只进 SFT
Teacher trajectory 为什么要 filter
为什么保留 0-search 样本
Narrow / Standard 为什么这样设计
trajectory 如何变成 SFT input
loss mask 为什么这样构造
LoRA / full FT 为什么这样选择
训练超参数怎么确定
evaluation 为什么这样做
GRPO 和 SFT 的区别
online rollout 是什么
Retriever 如何进入 Agent loop
实验结果说明了什么
失败案例如何排查
```

---

## 17. 以后每个阶段固定采用的协作流程

每当进入一个新模块，按照以下顺序：

```text
【1】先确认项目当前状态

【2】明确下一步工程任务

【3】列出完成该任务必须掌握的知识

【4】只学习当前必要知识，不提前扩散

【5】用 SearchAgent-RL 自己的数据和代码理解知识

【6】共同确定设计方案

【7】再让 Codex / ChatGPT 辅助实现代码

【8】对代码做真实审查

【9】跑 smoke test / audit / experiment

【10】复盘设计、问题、结果和面试表达

【11】进入下一模块
```

如果之后对话开始偏离，可以直接回到本文件判断：

> “当前问题是否服务于最近的下一步工程任务？”

如果不是，可以先记录，但不改变主线。

---

# 18. 当前最近任务

目前只推进这一条：

```text
Agentic SFT Renderer
```

学习顺序固定为：

```text
1. trajectory → messages
2. Qwen Chat Template
3. token-level loss mask
4. padding / truncation
5. DataCollator
6. renderer 实现
7. 真实样本审查
8. token length audit
9. 决定 max_seq_length
```

当前下一课：

```text
SearchAgent trajectory
↓
messages
↓
system / user / assistant role
↓
Qwen apply_chat_template()
```

---

# 19. 一句话原则

> **项目决定学什么，知识服务于当前实现；学懂后马上做，做完后马上复盘，再进入下一阶段。**
