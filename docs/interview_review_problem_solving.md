# SearchAgent-RL：问题、排查与设计决策复习笔记

> 用途：面试复习与项目设计回顾。
>
> 本文重点不是记录“最后代码长什么样”，而是总结项目推进过程中遇到的真实问题、为什么原方案存在风险、最终如何解决，以及这些问题背后的 Agent / Post-training 设计原则。

---

## 1. 项目整体路线

SearchAgent-RL 的目标是完成一个具有真实搜索交互能力的大模型 Agent 后训练项目，而不是机械复现某个开源仓库。

当前固定路线：

```text
QA Data
→ Data Processing / Dedup / Split
→ Local BM25 Retriever
→ DeepSeek Teacher Trajectory Generation
→ Teacher Quality Filtering
→ Agentic SFT
→ Qwen3 Online Rollout
→ veRL GRPO
→ Evaluation
```

核心配置：

- 数据：NQ + HotpotQA + AetherSearch
- Base model：Qwen3-4B
- Retriever：Local BM25
- Teacher：DeepSeek API
- Training：Agentic SFT + veRL GRPO

这条路线中最重要的一点是区分三种不同对象：

1. 原始 QA 数据
2. Teacher 产生的 Agent trajectory
3. GRPO 阶段由 Qwen3 在线产生的 rollout

三者不能混在一起设计。

---

## 2. 问题一：不同数据集 schema 不统一，后续流程容易污染

### 2.1 问题

NQ、HotpotQA 和 AetherSearch 原始结构不同。

如果每个下游模块都分别兼容三种 schema，会导致：

```text
retriever / split / teacher / SFT / evaluation
```

每一层都出现 source-specific 判断，代码会快速膨胀。

### 2.2 解决方案

先将 NQ 和 HotpotQA 统一到 canonical QA schema，例如：

```json
{
  "uid": "nq_train_0",
  "source": "nq",
  "source_split": "train",
  "question": "...",
  "gold_answers": ["..."],
  "metadata": {
    "original_id": "train_0"
  }
}
```

AetherSearch 本身包含现成 trajectory，因此保留其 trajectory-specific 信息，但仍明确其 source、question、answers 和 events。

### 2.3 设计原则

**在数据入口统一 schema，而不是在每个下游模块不断兼容原始格式。**

面试时可以这样解释：

> 数据处理阶段首先建立 canonical representation，使后续 Retriever、Teacher、SFT 和 Evaluation 面向统一的数据接口，从而避免数据源差异向训练管线扩散。

---

## 3. 问题二：数据泄漏不能只依赖原始 train/dev/test

### 3.1 问题

NQ 和 HotpotQA 本身存在各自的官方 split，但项目同时使用多个数据源，并且要重新构造：

```text
Teacher-SFT
GRPO
Val
Test
Reserve
```

如果简单从不同源独立抽样，可能存在语义重复或近重复问题，造成训练/测试泄漏。

### 3.2 解决方案

在重新划分之前先进行全局 question-level dedup。

使用的处理思路包括：

```text
Unicode NFKC
casefold
whitespace normalization
trailing punctuation normalization
exact duplicate
MinHash-LSH candidate retrieval
true Jaccard verification
manual review
Union-Find duplicate components
```

之后再进行固定 seed 的实验划分。

当前冻结划分：

```text
                     NQ       HotpotQA     Total
Teacher-SFT        2,000       2,000       4,000
GRPO               1,000       1,000       2,000
Val                   500         500       1,000
Test                1,000       1,000       2,000
```

### 3.3 关键认识

重新构建实验 split 后，Test 不能再简单称作“官方 NQ test”。

因为原始 split 只是 provenance，新的实验 Test 是项目重新采样得到的 evaluation split。

### 3.4 面试表达

> 多源 QA 数据不能只按原始 split 拼接，因为不同数据集之间可能存在重复问题。项目先做全局 question-level dedup，再构造互斥的 SFT、GRPO、Val 和 Test split，从数据层面减少信息泄漏。

---

## 4. 问题三：AetherSearch 是否应该进入 GRPO

### 4.1 问题

AetherSearch 本身提供了搜索 trajectory，很容易直觉上认为它既可以用于 SFT，也可以用于 GRPO。

但这会混淆 SFT 与 RL 的训练对象。

### 4.2 最终方案

AetherSearch：

```text
只用于 SFT
不进入 GRPO
```

GRPO split 只需要：

```json
{
  "question": "...",
  "gold_answers": ["..."]
}
```

GRPO 阶段的 trajectory 必须由 **SFT 后的 Qwen3 在线与 BM25 环境交互生成**。

### 4.3 为什么

SFT 学习的是 demonstration trajectory。

GRPO 优化的是当前 policy 自己产生的 rollout。

因此：

```text
Teacher / AetherSearch trajectory → SFT demonstration
Qwen3 online trajectory           → RL rollout
```

不能混为一谈。

---

## 5. 问题四：Retriever 应该只是一个工具，而不是和 Agent Loop 耦合

### 5.1 问题

Agent 项目很容易把 BM25 搜索、格式化结果、模型循环和数据保存全部写到同一个 runner 中。

这样短期能运行，但以后换 Retriever、做 batch、做 RL rollout 时会很难维护。

### 5.2 当前结构

```text
BM25Retriever
    ↓
SearchTool
    ↓
SearchEnvironment
    ↓
AgentLoop
```

职责分别为：

#### BM25Retriever

只负责信息检索：

```python
search(query, top_k)
```

#### SearchTool

负责把 Retriever 输出转换为模型可读文本。

#### SearchEnvironment

负责执行 Agent action，并产生真正的 environment observation。

#### AgentLoop

负责模型与环境之间的多轮交互和 rollout 生命周期。

### 5.3 设计原则

**Retriever 是环境能力，不应该直接写死在模型循环里。**

---

## 6. 问题五：模型输出多个 action 时，不能偷偷选择最后一个

### 6.1 旧问题

Agent 输出可能出现：

```text
<search>A</search>
<answer>B</answer>
```

或者：

```text
<search>A</search>
<search>B</search>
```

如果 parser 使用：

```python
matches[-1]
```

实际上是在“替模型决定”哪个 action 有效。

这会掩盖 policy 本身的格式错误。

### 6.2 最终规则

```text
0 个 action      → invalid
恰好 1 个 action → candidate valid
>=2 个 action    → invalid
空 action        → invalid
出现 information → invalid
```

也就是说 Agent protocol 要求每轮 visible response **恰好一个 action**。

### 6.3 设计原则

Parser 的职责是验证 protocol，而不是自动纠正模型输出。

面试表达：

> 对 Agent action parser 采用严格协议，而不是自动选择最后一个 action。否则模型格式错误会被 parser 隐式修复，训练和评测时无法真实反映 policy behavior。

---

## 7. 问题六：DeepSeek native thinking 应不应该重新拼回 context

### 7.1 问题

DeepSeek thinking mode 返回：

```text
reasoning_content
visible content
```

一个容易产生的错误设计是：

```text
reasoning_content
→ 拼成 <think>...</think>
→ 放回下一轮 runtime context
```

### 7.2 最终方案

```text
reasoning_content 单独保存
visible content 作为 model_output
下一轮 runtime context 只包含：
Question + 历史 search + 历史 information
```

`generate()` 只 return visible content。

### 7.3 为什么

Teacher 的 internal reasoning 是 API 返回的 hidden reasoning channel，不等同于环境 observable state。

下一轮搜索 Agent 真正应该看到的是：

```text
问题
+ 自己已经执行过的 action
+ environment 返回的信息
```

而不是重新注入上一轮 hidden reasoning。

---

## 8. 问题七：Teacher visible response 是否应该自己生成 information

### 8.1 问题

如果 Teacher 可以输出：

```text
<information>...</information>
```

那么模型就可能绕过 Retriever，自己“伪造搜索结果”。

### 8.2 解决方案

Teacher prompt 和 protocol 都明确：

```text
模型只能生成 search 或 answer
information 只能由 SearchEnvironment 产生
```

因此：

```text
Model → action
Environment → observation
```

边界清晰。

---

## 9. 问题八：run_teacher.py 通过字符串差分反推 observation

这是目前 Teacher runtime 重构中最重要的问题之一。

### 9.1 原实现

旧代码通过：

```python
next_context = raw_turns[index + 1]["context"]
prefix = turn["context"] + turn["model_output"]
observation = next_context[len(prefix):]
```

也就是：

```text
下一轮 context
-
上一轮 context + model_output
=
猜测上一轮 environment observation
```

### 9.2 为什么设计不合理

Environment 在运行时明明已经产生了 observation，但没有结构化保存。

之后 runner 再根据字符串拼接关系“反推”运行状态。

这会带来几个问题：

1. runtime truth 没有在产生时记录
2. 强依赖 context 拼接格式
3. runner 被迫了解 AgentLoop 内部实现
4. 后续修改 context formatting 时容易 silently break

### 9.3 最终方案

AgentLoop 每轮直接保存真实 structured step：

```python
@dataclass
class AgentStep:
    model_output: str
    action: AgentAction
    observation: str
```

运行过程：

```text
model.generate()
→ environment.step()
→ 当场记录 AgentStep
```

之后 `run_teacher.py` 直接使用：

```text
model.raw_turns
+
result.steps
```

组装 canonical events。

### 9.4 核心原则

**运行时产生的数据，应在产生时结构化记录，而不是事后从字符串恢复。**

这是一个很适合在 Agent 系统设计面试中讲的工程问题。

---

## 10. 问题九：action 为什么不能在 run_teacher.py 再 parse 一遍

### 10.1 问题

旧 runner 会再次：

```python
parse_action(turn["model_output"])
```

但同一条 output 在 Environment 中已经 parse 过一次。

### 10.2 风险

这样会出现两个“事实来源”：

```text
Environment 真正执行的 action
Runner 后来重新 parse 的 action
```

即使当前 parser 是纯函数，这种设计仍然是不必要的重复。

### 10.3 最终方案

直接使用：

```python
step.action
```

即 runtime 中真正执行过的 action。

### 10.4 原则

**Single Source of Truth。**

同一个 action 不应在 pipeline 不同位置被重复解释。

---

## 11. 问题十：Teacher invalid trajectory 是否需要继续生成

### 11.1 问题

旧逻辑中 invalid action 会收到：

```text
My previous action is invalid...
Let me try again.
```

然后模型继续下一轮。

这对于通用交互 Agent 可能是某种设计选择，但对于 Teacher SFT trajectory generation 并不合适。

### 11.2 最终决策

Teacher 一旦首次出现 invalid：

```text
invalid
→ 当前 sample 立即失败
→ AgentLoop return
→ 不追加纠错 observation
→ 不再次调用 DeepSeek
→ run_teacher.py 进入下一条 QA
```

### 11.3 为什么

Teacher 的目标是构造 clean demonstration。

如果一条 trajectory 已经 invalid，它最终本来就不会进入 clean SFT 数据。

继续调用 Teacher 只会：

- 浪费 API token
- 产生污染的 correction trajectory
- 增加数据结构复杂度

### 11.4 重要区分

```text
不使用 invalid trajectory 训练
≠
完全不记录失败信息
```

raw 层仍可以保留失败前已经真实发生的 partial trace，用于 pilot/debug。

---

## 12. 问题十一：INVALID_ACTION_OBSERVATION 是否应该属于 Environment

### 12.1 原设计

Environment 对 invalid action 返回一段 corrective text。

### 12.2 重新分析

SearchEnvironment 的语义应该是执行合法环境动作。

```text
search → BM25 → information
answer → terminate
```

invalid action 实际上没有执行任何环境动作。

因此人为生成：

```text
Your previous action is invalid...
```

更接近 rollout policy 的纠错策略，而不是 SearchEnvironment 的真实 observation。

### 12.3 当前最终方案

```text
search  → observation = <information>...</information>
answer  → observation = ""
invalid → observation = ""
```

Teacher AgentLoop 发现 invalid 后立即终止。

未来 GRPO 是否给 invalid penalty、是否允许 correction retry，在真正实现 GRPO 时再设计。

### 12.4 原则

**Environment semantics 与 rollout policy 分离。**

不要为了未知的未来需求提前把 retry、policy、handler 等抽象塞进当前代码。

---

## 13. 问题十二：Agent runtime 保存了过多重复状态

### 13.1 原问题

`StepResult` 曾同时保存：

```text
action
observation
done
valid
is_search
```

但：

```text
action.type == search
→ valid=True
→ done=False
→ is_search=True

action.type == answer
→ valid=True
→ done=True
→ is_search=False

action.type == invalid
→ valid=False
→ done=False
→ is_search=False
```

后三者完全可以从 `action.type` 推导。

### 13.2 最终设计原则

真正保存：

```python
@dataclass
class StepResult:
    action: AgentAction
    observation: str
```

为了代码可读性，可以提供 property：

```text
valid
done
is_search
```

但不把它们作为独立 state 保存。

### 13.3 为什么

否则可能出现逻辑互相矛盾的数据：

```text
action.type = search
is_search = false
```

使用派生属性后，`action.type` 成为唯一 source of truth。

---

## 14. 问题十三：AgentLoopResult 是否应该保存所有统计字段

### 14.1 原设计

曾保存：

```text
trajectory
final_answer
done
turns
search_count
valid_action_count
invalid_action_count
termination_reason
```

structured steps 引入后，大量字段都可以推导。

### 14.2 更合理的核心状态

```python
@dataclass
class AgentLoopResult:
    trajectory: str
    steps: list[AgentStep]
    termination_reason: TerminationReason
```

其中：

```text
turns        = len(steps)
search_count = search step 数量
done         = termination_reason == "answer"
final_answer = 最终 answer action content
```

这些可以通过 property 提供，但不应该重复存储。

`valid_action_count` 和 `invalid_action_count` 没有必要作为核心接口继续存在。

### 14.3 核心思想

**状态最小化不是为了代码短，而是为了消除不一致状态空间。**

这是很重要的软件设计原则。

---

## 15. 问题十四：Teacher raw data、clean data 和 SFT data 不能混成一层

### 15.1 问题

如果 Teacher 生成结束后直接把所有 trajectory 写成最终 SFT 数据，会导致：

- invalid sample 混入
- max_turns sample 混入
- gold answer correctness 尚未验证
- environment observation 和 model target 的 loss mask 尚未处理

### 15.2 最终三层设计

```text
① raw Teacher trajectories
   ├── success
   ├── invalid
   └── max_turns

        ↓ quality filter

② clean canonical trajectories
   └── only valid + correct trajectories

        ↓ SFT renderer

③ Qwen3 SFT samples
   └── chat template / tokenization / labels / loss mask
```

### 15.3 为什么 canonical events 不等于最终 SFT tokens

canonical event：

```json
{"type": "search", "content": "..."}
```

是一种中间语义表示。

最终 SFT 还需要：

- chat template
- tokenization
- label 构造
- loss mask

因此不能把两层混在一起。

---

## 16. 问题十五：environment observation 为什么不能参与 SFT loss

最终计划中：

```text
think        → 参与 loss
search       → 参与 loss
information  → 不参与 loss
answer       → 参与 loss
```

原因是：

```text
think / search / answer
```

属于模型应该生成的内容。

而：

```text
information
```

是 Environment observation。

模型需要读取它，但不能被训练成“自己生成检索结果”。

这是 Agentic SFT 和普通对话 SFT 的一个重要区别。

---

## 17. 问题十六：Teacher quality filter 为什么先做结构合法性，再做答案正确性

Teacher trajectory quality 包含两个维度：

### 17.1 Process validity

例如：

```text
termination_reason == answer
最后 event 是 answer
每个 search 后恰好一个 information
event sequence 合法
```

### 17.2 Answer correctness

在结构合法后再使用 gold answer 检查最终答案。

第一版采用 normalized EM against ANY gold answer。

至少考虑：

```text
Unicode normalization
casefold
punctuation
whitespace
英文 a / an / the
```

不使用简单 substring。

### 17.3 为什么分两阶段

一个答案即使碰巧正确，如果 Agent trajectory 本身 protocol invalid，也不应该作为高质量 Agent demonstration。

反过来，trajectory 合法但 answer 错误，也不能进入 SFT。

因此：

```text
合法过程
AND
正确答案
```

两者都必须满足。

---

## 18. 问题十七：为什么当前不提前设计 GRPO invalid policy

### 18.1 背景

Teacher 阶段已经确定：

```text
invalid → fail-fast
```

但 GRPO 阶段可能存在不同选择：

```text
invalid → terminate + negative reward
invalid → penalty + retry
invalid → protocol feedback + continue
```

### 18.2 当前决策

现在不设计。

原因是当前任务是 Teacher trajectory generation，而不是 GRPO rollout implementation。

如果提前加入：

```text
stop_on_invalid
invalid_policy
retry_policy
strategy class
callback framework
```

属于为未知需求过度抽象。

### 18.3 原则

**YAGNI：只为当前已经明确的行为建立抽象。**

以后真正实现 GRPO 时，再根据 reward design 和 rollout semantics 决定是否复用当前 AgentLoop。

---

## 19. 问题十八：为什么没有把所有代码包装成 src/searchagent_rl/... 的“大工程结构”

项目早期很容易为了“看起来工程化”创建：

```text
src/
configs/
utils/
training/
evaluation/
reward/
callbacks/
```

但如果模块还不存在真实职责，这些目录只会增加跳转和认知负担。

当前原则：

```text
真实需要时再扩展
```

目前结构保持：

```text
agent/
retrieval/
teacher/
scripts/
tests/
data/
docs/
```

足够表达真实模块边界。

面试时可表达为：

> 项目没有为了形式上的工程化过早建立复杂 package hierarchy，而是按照真实职责演化模块，避免 premature abstraction。

---

## 20. 当前 Teacher Agent 的最终职责划分

```text
DeepSeekTeacherModel
    │
    │ 产生 model_output + reasoning_content
    ▼
protocol.py
    │
    │ text → AgentAction
    ▼
SearchEnvironment
    │
    ├── search  → BM25 → information
    ├── answer  → empty observation
    └── invalid → empty observation
    ▼
AgentLoop
    │
    ├── search  → append observation → continue
    ├── answer  → success
    └── invalid → fail-fast
    ▼
run_teacher.py
    │
    ├── combine raw_turns + structured steps
    ├── save raw trajectory
    └── move to next QA
```

这个设计中最关键的是：

```text
Model 产生 action
Environment 产生 observation
Loop 管理 rollout
Runner 管理 dataset
```

每一层只负责自己的生命周期。

---

## 21. 面试高频问题与简洁回答

### Q1：为什么不用 DeepSeek native tool calling？

项目希望显式控制 Search-R1 风格的文本 action protocol，并让同一 protocol 后续可以直接用于 Qwen3 SFT / rollout，因此采用 `<search>` / `<answer>` 文本 action，而不是绑定 Teacher provider 的 native tool calling。

### Q2：为什么 reasoning_content 不回灌下一轮 context？

因为 reasoning_content 是 Teacher API 的内部 reasoning channel，不是 environment observable state。下一轮 Agent 应基于问题、历史 action 和检索 observation 决策，而不是重新注入上一轮 hidden reasoning。

### Q3：为什么 information 不参与 SFT loss？

因为 information 是 Retriever / Environment 产生的 observation。模型需要消费它，但不应该学习生成它，否则会混淆 Agent policy 与 environment。

### Q4：为什么 Teacher invalid 后直接丢弃，不让模型纠错？

Teacher 的目标是生成 clean demonstration。invalid trajectory 最终不会进入 SFT，继续调用 Teacher 只会浪费 API token，并把错误纠正过程混入 demonstration。

### Q5：为什么仍然保留失败样本的 partial trace？

不是为了训练，而是为了 pilot/debug。通过 partial trace 可以定位 invalid 出现在哪一轮、之前搜了什么、模型实际输出了什么。

### Q6：为什么要 structured AgentStep？

因为 action 和 observation 都是在 runtime 中真实产生的状态，应该在产生时直接记录。否则 runner 从字符串拼接结果反推 observation，会强耦合 context formatting，并产生多个事实来源。

### Q7：为什么 AgentStep 还要保存 model_output，既然已经有 action？

因为 parser 会把非法输出归一化为 `AgentAction(type="invalid", content="")`。如果不保留原始 model_output，就无法知道模型具体违反了什么 protocol。

### Q8：为什么要删 done、valid、is_search 等字段？

这些状态可以从 `action.type` 唯一推导。重复保存会增加不一致状态空间，例如 `action.type=search` 但 `is_search=False`。通过 property 推导可以保持接口可读，同时保证 single source of truth。

### Q9：SFT 和 GRPO 最大的数据区别是什么？

SFT 需要高质量 demonstration trajectory。GRPO 原始数据只需要 question 和 gold answers，trajectory 由当前 Qwen3 policy 在线与 BM25 环境交互生成。

### Q10：为什么 AetherSearch 只用于 SFT？

因为 AetherSearch 提供的是离线 demonstration trajectory，适合 imitation learning。GRPO 应优化当前 policy 自己生成的 rollout，而不是使用预先固定的 Teacher trajectory。

---

## 22. 这段开发过程体现出的工程原则

可以总结为以下几点：

### 1. Single Source of Truth

action、observation、termination state 不重复解释和重复存储。

### 2. Separation of Concerns

```text
Model
Protocol
Environment
Rollout Loop
Dataset Runner
```

职责分离。

### 3. Fail Fast

Teacher trajectory 已经 invalid 后立即终止，不继续消耗 API token。

### 4. Preserve Runtime Truth

运行时产生的数据当场结构化记录，不从最终字符串反推。

### 5. Avoid Premature Abstraction

GRPO 尚未实现时，不提前设计复杂 invalid policy framework。

### 6. Raw / Clean / Rendered 分层

审计数据、训练前 canonical 数据、最终 tokenized SFT 数据明确分离。

### 7. Agent Policy 与 Environment Observation 分离

模型负责生成 action，环境负责生成 observation，SFT loss 只监督模型应生成的内容。

---

## 23. 当前后续路线

Teacher runtime 重构完成后，后续顺序应为：

```text
1. 5-sample smoke test
2. 100-sample Teacher pilot
3. 分析 search_count / termination / invalid / answer quality
4. Teacher structural quality filter
5. normalized EM gold correctness
6. clean canonical trajectory
7. Qwen3 SFT renderer
8. Agentic SFT
9. Qwen3 online rollout
10. veRL GRPO
11. Evaluation
```

其中 `search_count >= 1` 是否作为 Teacher clean trajectory 的硬条件，应先观察 100-sample pilot 后再决定，不提前写死。

---

## 24. 最值得面试中强调的一点

这个项目不是简单把 Search-R1 的代码复制过来，而是在实现过程中不断明确以下边界：

```text
数据和 trajectory 的边界
Teacher 和 policy model 的边界
Agent action 和 environment observation 的边界
SFT demonstration 和 GRPO rollout 的边界
runtime state 和 serialized metadata 的边界
```

真正有价值的部分不是“写了一个 BM25 + API loop”，而是能够解释：

> 一个可训练的 Search Agent 系统中，哪些内容属于 policy，哪些属于 environment，哪些状态应该结构化记录，哪些数据应该进入 imitation learning，哪些行为应该留到 reinforcement learning 阶段在线产生。

这也是后续面试复习时最应该重点掌握的主线。
