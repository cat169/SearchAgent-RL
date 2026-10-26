# SearchAgent-RL Teacher v1 流程核查与 Teacher v2 实验准备

审查日期：2026-10-26（Asia/Shanghai）

## 0. 审查边界与本地版本状态

- 当前分支：`main`
- 本地 `HEAD`：`5b842f573a1ba602793dee1603e5e76273d1deca`
- `origin/main`：`5b842f573a1ba602793dee1603e5e76273d1deca`
- Teacher 运行链中的现存文件（`run_teacher.py`、`teacher/`、`agent/`、`retrieval/`）相对 `HEAD` 无差异。
- `scripts/audit_teacher_trajectories.py` 和 `scripts/filter_teacher_trajectories.py` 在本地工作树中已被删除，但删除尚未提交。本文只能用 `HEAD` 中的版本解释现有 `teacher_audit_report.json` 和 `teacher_filter_report.json`，不能称其为当前可直接运行的本地入口。
- 同批本地未提交删除还包括 `build_splits.py`、`build_sft_data.py`、`build_sft_splits.py` 等数据入口，因此当前工作树不能从头复现完整数据链。
- 当前磁盘不存在 `data/processed/splits/`、`data/processed/sft/` 和整个 `reports/` 历史审查目录；任务背景中的 KEEP=3,142、Standard=2,000、Validation=150 以及“30 条人工审查结论”无法从本地文件重新核验。本文不会用历史口述替代文件证据。
- 当前决定“正式主线不使用 AetherSearch”尚未体现在 `HEAD` 的历史构建逻辑中：`HEAD:scripts/build_sft_data.py` 仍读取 2,251 条 clean Teacher 和 1,999 条 AetherSearch，并在固定排除 4 条后断言总数 4,246。

本轮未调用 DeepSeek API，未修改任何 JSONL、Prompt 或 Python 业务代码。

## 1. Teacher v1 完整运行流程与代码定位

```text
teacher_sft_candidates.jsonl
  -> run_teacher.py 选择 5 / 100 / 4000 条
  -> DeepSeekTeacherModel.generate(messages)
       system: AGENT_SYSTEM_PROMPT
       history: Question + visible actions + environment Information
  -> AgentLoop
       parse_action -> SearchEnvironment -> Local BM25 top-3
       最多 5 个模型回合
  -> extract_events(raw_turns, steps)
       reasoning_content -> think
       parsed action      -> search / answer
       observation        -> information
  -> raw_teacher_trajectories.jsonl（逐样本 flush，可 resume）
  -> audit_teacher_trajectories.py（当前本地缺失，HEAD 版本）
  -> filter_teacher_trajectories.py（当前本地缺失，HEAD 版本）
  -> clean_teacher_trajectories.jsonl
```

### 1.1 输入选择和输出

- `scripts/run_teacher.py:24-35`：seed=123、`MAX_TURNS=5`、`TOP_K=3`，候选、Lucene index 和 corpus 路径。
- `scripts/run_teacher.py:114-120`：只支持 `--limit 5/100/4000` 和 `--resume`。
- `scripts/run_teacher.py:160-182`：正式 4,000 条要求 NQ=2,000、HotpotQA=2,000。
- `scripts/run_teacher.py:199-223`：5/100/4000 分别写 smoke、pilot、raw 文件，并有对应 empty-skipped ID 文件。
- `scripts/run_teacher.py:241-250`：fresh run 使用 `w`，resume 使用 `a`。
- `scripts/run_teacher.py:266-323`：逐条运行、构造输出、立即写入并 `flush()`。

### 1.2 单条 Agent 执行

- `scripts/run_teacher.py:270-272`：每条清空 `raw_turns`，只把 `Question: ...` 传给 loop。
- `agent/loop.py:68-73`：初始上下文只有 user question。
- `agent/loop.py:76-102`：每轮将模型**可见输出**作为 assistant 消息追加；若为 Search，再把环境 observation 作为 user 消息追加。
- `agent/loop.py:104-117`：Answer 结束；Invalid 立即结束；5 轮耗尽为 `max_turns`。
- `agent/environment.py:32-65`：Search 才调用 SearchTool，并把结果唯一封装为 `<information>...</information>`；Answer/Invalid observation 为空。
- `agent/search_tool.py:23-39,68-89`：默认 top-k，按 rank/title/content 格式化，不截断（当前构造未设置 `max_text_chars`）。
- `retrieval/bm25_retriever.py:58-97`：query strip 后用本地 Lucene BM25；空 query 报错。

### 1.3 动作协议

- `agent/protocol.py:15-18`：只识别完整配对的 `<search>` 或 `<answer>` 块。
- `agent/protocol.py:21-47`：伪造 `<information>`、0 个/多个 action、空 action 都返回 Invalid。
- 注意：parser 要求“恰好一个匹配 action”，但没有对整个 response 做 `fullmatch`。因此含一个合法 action 且前后夹有普通文本的输出仍可能被接受；这比 system prompt 的“action 前后不得有任何内容”更宽松。

## 2. v1 当前真实生成参数

| 项目 | 实际值 | 代码位置 |
|---|---|---|
| Model | `deepseek-flash` | `teacher/deepseek_teacher.py:15,30` |
| Base URL | `https://api.deepseek.com` | `teacher/deepseek_teacher.py:18,42-45` |
| Temperature | `0.0` | `teacher/deepseek_teacher.py:17,31,61` |
| max_tokens | `4096`（每次 API completion） | `teacher/deepseek_teacher.py:16,32,62` |
| reasoning_effort | `high` | `teacher/deepseek_teacher.py:63` |
| Thinking | `extra_body.thinking.type=enabled` | `teacher/deepseek_teacher.py:64-68` |
| Top-K | `3` | `scripts/run_teacher.py:27,252-258` |
| 最大模型回合 | `5` | `scripts/run_teacher.py:26,260-264`；`agent/loop.py:76` |
| 随机种子 | `123`（只用于候选抽样，不传给 API） | `scripts/run_teacher.py:25,99` |
| API 重试 | 无 | `teacher/deepseek_teacher.py:58-69` |

代码没有显式设置 `top_p`、presence/frequency penalty、seed 或 stop sequences；这些字段采用 API 默认行为。

### 2.1 System Prompt 的完整来源

`teacher/deepseek_teacher.py:9,20,51-57` 将 `agent.system_prompt.AGENT_SYSTEM_PROMPT` 直接别名为 `TEACHER_SYSTEM_PROMPT`，每次 API 调用在消息首部插入。不存在 Teacher 独立 prompt。

Prompt 的核心完整约束位于 `agent/system_prompt.py:1-99`：

1. reasoning 由 thinking mode 内部处理，visible response 禁止 `<think>`；
2. 每轮只能基于原问题和已有 Information 做一个决定；
3. 可见输出必须恰好为一个 `<search>query</search>` 或 `<answer>short answer</answer>`；
4. Search 后等待环境，禁止模型伪造 Information；
5. 信息足够时立即回答，禁止冗余确认搜索；
6. Answer 应为最短答案 span，不附解释、证据或前导语。

### 2.2 每轮模型实际看到什么

API 输入是 system prompt 加 `AgentLoop` 的 `messages`（`deepseek_teacher.py:51-60`）：

- 始终看到原始 Question；
- 看到此前每轮的 visible `<search>` action；
- 看到环境返回的 `<information>` BM25 top-3；
- **看不到此前轮次的 `reasoning_content`**。`AgentLoop` 只把 `model_output`（visible content）追加为 assistant 消息（`agent/loop.py:77-82`）；reasoning 只进入 `model.raw_turns`，没有回填进后续 messages。

这符合“不暴露内部思考”的实现，但多轮中模型需要根据 visible action 和新 evidence 重新建立状态。是否因此导致重复推理是待验证假设，不能仅凭代码认定。

### 2.3 Gold 隔离

- 生成调用只收到 `record['question']`（`run_teacher.py:272`）。
- `record['gold_answers']` 直到 rollout 完成后才复制进输出的 `answers`（`run_teacher.py:301-318`）。
- 因而在现有代码路径中 Gold 没有进入 Teacher prompt、AgentLoop 或 BM25 query。

### 2.4 Think / Search / Information / Answer 的保存

- `deepseek_teacher.py:70-103`：分别读取 provider 的 `reasoning_content` 与 visible `content`，strip 后记录原始 turn。
- `run_teacher.py:46-84`：每轮优先把非空 `reasoning_content` 保存为 Think；为空时才从 visible `<think>...</think>` 取最后一个匹配。
- action 来自 parser 后的结构化 `AgentStep.action`，Search observation 必须完整匹配环境 `<information>` wrapper。
- `run_teacher.py:283-299`：保存前强制 search 事件数等于 runtime search 数，Information 数等于 Search 数。
- `run_teacher.py:308-318`：同时保存完整 `raw_model_turns` 和 termination reason，可追溯事件映射。

### 2.5 终止与失败

- Empty visible content：抛 `EmptyVisibleContentError`，记录到 skipped IDs，不写主轨迹（`deepseek_teacher.py:76-96`; `run_teacher.py:271-281`）。异常含 finish reason、reasoning chars、completion/reasoning tokens，不打印完整 reasoning。
- Invalid action：首个 invalid turn 立即结束并保存轨迹，termination=`invalid_action`。
- max turns：若 5 轮均为 Search，则 termination=`max_turns`，可有 5 次 Search。
- Answer 轨迹最多有 4 次 Search，因为第 5 回合必须用于 Answer。
- 其他网络/API 异常不吞掉、不自动 retry；此前已 flush 的 checkpoint 保留，当前样本不写入。

### 2.6 覆盖文件风险

存在明确风险：默认不带 `--resume` 时，主轨迹和 skipped 文件均以 `w` 打开（`run_teacher.py:241-250`），会覆盖相同 limit 对应的既有文件；pilot candidate 文件也由 `build_pilot()` 每次以 `w` 重建（`run_teacher.py:106-109`）。`--resume` 才会读取已完成/empty IDs、查重并 append（`run_teacher.py:227-250`）。

## 3. 数据阶段、数量与质量统计

### 3.1 当前实际存在的数据

| 阶段 | 文件 | 数量 | SHA256 | 说明 |
|---|---|---:|---|---|
| Raw | `data/processed/teacher/raw_teacher_trajectories.jsonl` | 3,982 | `429d1195bef21223bef24ffcf7745af4a92e700a76580ea5bda28f87a313d8fb` | API/loop 已产生并保存的轨迹，含非 Answer |
| Empty skipped | `raw_teacher_empty_skipped_ids.txt` | 18 | — | empty visible content，没有主轨迹 |
| Clean | `clean_teacher_trajectories.jsonl` | 2,251 | `6f52e51efef8a22648f6b0dea2a87b46733b4a23f1cc47db0faca901dab699a5` | 结构有效、Answer 终止且通过高精度答案匹配 |
| Final SFT KEEP | `data/processed/sft/all.jsonl` | 本地缺失 | — | 背景称 3,142，但无法核实 |
| Standard | `data/processed/sft/standard.jsonl` | 本地缺失 | — | 背景称 2,000，但无法核实 |

Raw 与 Clean 均保留 `id/source/question/answers/events/raw_model_turns/runtime/split`。历史 SFT 构建会丢弃 `raw_model_turns/runtime` 并重新计算 search_count，但当前 SFT 文件不可用。

### 3.2 生成成功与失败

分母均为每来源 2,000 个 Teacher candidate。

| Source | Raw 写入 | Raw 成功率 | Empty | Answer 终止 | Answer 成功率 | Invalid | Max turns | Clean 保留 | Clean/候选 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | 1,997 | 99.85% | 3 | 1,828 | 91.40% | 2 | 167 | 909 | 45.45% |
| HotpotQA | 1,985 | 99.25% | 15 | 1,768 | 88.40% | 8 | 209 | 1,342 | 67.10% |
| Total | 3,982 | 99.55% | 18 | 3,596 | 89.90% | 10 | 376 | 2,251 | 56.28% |

主要未保留原因（`teacher_filter_report.json`）：

- non-answer：386（10 invalid + 376 max-turn）；
- correctness mismatch：1,345（NQ Answer 中 919 条未进入 clean，HotpotQA 426 条未进入 clean）；
- canonical invalid：0。

### 3.3 Answer 与参考答案匹配

旧 audit 的“exact match”实际是标准化 EM（小写、去冠词/标点、合并空白），不是语义正确性评估。

| Source | 有效 Answer | normalized EM | EM rate | Clean 高精度保留 |
|---|---:|---:|---:|---:|
| NQ | 1,828 | 869 | 47.54% | 909 |
| HotpotQA | 1,768 | 1,332 | 75.34% | 1,342 |
| Total | 3,596 | 2,201 | 61.21% | 2,251 |

Clean 比 normalized EM 多 50 条：date equivalent 6、number equivalent 17、multi-answer equivalent 27。它仍只是字符串/结构型高精度规则，不能证明 Search 和 evidence chain 合理。

### 3.4 Search 次数分布

| 阶段/来源 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Raw NQ | 1,066 | 334 | 265 | 114 | 51 | 167 |
| Raw HotpotQA | 805 | 562 | 280 | 86 | 43 | 209 |
| Raw total | 1,871 | 896 | 545 | 200 | 94 | 376 |
| Clean NQ | 591 | 155 | 110 | 36 | 17 | 0 |
| Clean HotpotQA | 655 | 408 | 198 | 56 | 25 | 0 |
| Clean total | 1,246 | 563 | 308 | 92 | 42 | 0 |

Raw 中 5-search 全部对应 max-turn；Answer 轨迹没有 5-search。

### 3.5 Think 长度：字符与 Token 分开统计

定义：每条轨迹把所有 Think event 的长度相加。

- 字符：Python `len(content)`。
- Token：本地缓存 `Qwen/Qwen3-4B-Base` 的 `Qwen2Tokenizer`、Transformers 5.17.0，对每个 Think 原文执行 `encode(add_special_tokens=False)` 后求和。
- Token 数**不含** Chat Template、`<think>` 标签、action 和 Information；它不是完整 SFT 序列长度，也不是 DeepSeek API usage token。

| 阶段/来源 | 字符 Median | 字符 P90 | 字符 P95 | 字符 Max | Token Median | Token P90 | Token P95 | Token Max | Token >1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw NQ | 555 | 1,957 | 2,830 | 20,718 | 133 | 485 | 702 | 5,838 | 41 |
| Raw HotpotQA | 843 | 3,351 | 4,879 | 24,591 | 207 | 819 | 1,200 | 5,945 | 142 |
| Raw total | 685 | 2,625 | 3,896 | 24,591 | 169 | 676 | 947 | 5,945 | 183 |
| Clean NQ | 411 | 1,286 | 1,664 | 8,877 | 97 | 315 | 432 | 2,103 | 8 |
| Clean HotpotQA | 669 | 2,142 | 2,965 | 15,021 | 168 | 528 | 726 | 3,665 | 27 |
| Clean total | 555 | 1,867 | 2,561 | 15,021 | 138 | 458 | 658 | 3,665 | 35 |

超长候选：

- Raw：183 条 Think 原文累计超过 1,000 Qwen tokens；其中零搜索 18 条。
- Clean：35 条超过 1,000 tokens；其中零搜索 8 条。
- 以字符 >4,000 为另一独立阈值：Raw 182、Clean 37；零搜索分别 17、9。

### 3.6 重复 Search 候选

| 阶段 | 同轨迹 exact normalized query 重复 | 不同 query 的词集合 Jaccard ≥0.8 候选 |
|---|---:|---:|
| Raw | 8 | 184 |
| Clean | 0 | 19 |

第二列仅是轻量词面候选规则，不代表语义重复；需要人工检查实体、关系和检索结果是否产生新证据。

## 4. 超长 Think 的代码层面解释

### 已确认的机制

1. 每轮明确启用 thinking，且 `reasoning_effort="high"`（`deepseek_teacher.py:63-68`）。
2. 每次 completion 的 `max_tokens=4096`，没有独立 reasoning 长度上限（`deepseek_teacher.py:16,58-68`）。
3. 一条轨迹最多 5 次 API 调用，表中轨迹 Think 长度是多轮累计值。
4. Prompt 约束 visible answer/query 简短，但没有要求内部 reasoning 简洁或设置 reasoning budget（`system_prompt.py:3-4,55-64`）。
5. 旧 filter 只要求结构、Answer termination 和答案匹配，不以 Think 长度、搜索重复或 evidence use 为门槛（`HEAD:filter_teacher_trajectories.py:186-223`）。因此 35 条 >1,000-token Think 仍可进入 clean。

### 待验证假设

- `reasoning_effort=high` 是否是超长 Think 的主要因果因素，需要在同题、同 prompt、同检索环境下与另一 effort 做配对实验；当前统计不能证明因果。
- 先前 reasoning 不进入下一轮上下文，可能使模型重复建立推理状态，但是否造成实际重复必须逐轨迹比较。
- `max_tokens=4096` 是否包含/如何分配 reasoning 与 visible tokens 取决于 API 服务端语义；18 条 empty-visible 的诊断可能与 reasoning 消耗有关，但仅凭当前保存文件无法确认。
- HotpotQA 的 Think 长尾显著长于 NQ，可能来自多跳问题复杂度、检索不命中或模型行为；不能只归因于 API 参数。

## 5. 原始 split 与数据泄漏风险

### 5.1 实际进入 Teacher 阶段的原始 split

| 阶段/来源 | train | dev | test | Total |
|---|---:|---:|---:|---:|
| Raw NQ | 1,739 | 173 | 85 | 1,997 |
| Raw HotpotQA | 1,848 | 137 | 0 | 1,985 |
| Clean NQ | 788 | 83 | 38 | 909 |
| Clean HotpotQA | 1,261 | 81 | 0 | 1,342 |

因此 `nq_test_*`、`nq_dev_*`、`hotpotqa_dev_*` 确实进入了 Teacher raw 和 clean。

### 5.2 已确认与尚未确认

- 历史 `HEAD:scripts/build_splits.py` 对全局去重后的 unique pool 按来源、UID 排序、seed=123 shuffle，再切 Teacher/GRPO/Val/Test/Reserve；这些内部集合按 UID 明确互斥。
- 但它没有按原始 train/dev/test 隔离，所以 Teacher 集本身包含原始 NQ test 和 HotpotQA dev。
- 当前 `splits/`、SFT 和最终 evaluation 配置缺失，无法证明这些轨迹是否进入当前 Standard，也无法证明计划中的独立 Agent Evaluation 是否与其重叠。
- 如果最终评测直接使用完整 NQ test 或 HotpotQA dev，则上述 clean 中的 38 条 NQ test 和 81 条 HotpotQA dev 构成明确的候选污染；只有确认最终评测 ID/question 集后才能判定实际泄漏。

### 5.3 Teacher v1/v2 对照候选池建议

本轮不实际划分。建议候选池定义为：

1. 只使用 `nq_train_*` 和 `hotpotqa_train_*`；
2. 与拟定 Agent Evaluation 的 ID 和规范化 question 做精确排除；
3. v1/v2 使用完全相同的 question、Gold 和 BM25 snapshot；Gold 只用于离线评价，绝不进入 rollout；
4. 独立保留 `nq_dev/test`、`hotpotqa_dev` 作为评测候选，待用户确定最终 benchmark 后再冻结。

## 6. 已确认问题、潜在风险与证据缺口

### 已确认

- 默认 fresh run 会覆盖相同输出路径。
- 10 条 raw 为 invalid action，376 条达到 max turns，18 条 empty-visible。
- Clean 中仍有 35 条 Think >1,000 Qwen tokens，8 条为零搜索长 Think。
- Clean 中仍有 19 条高词面相似的多 Query 候选。
- Teacher Gold 输入隔离正确；Information 只由环境生成。
- 当前 parser 比 prompt 宽松，允许唯一 action 前后存在普通文本。
- 当前工作树缺少 audit/filter 和数据构建入口，复现链不完整。

### 潜在风险

- 高 effort thinking 可能产生大量对 Student 并非必要的监督 Token。
- 多轮不保留历史 reasoning，可能引发状态重建和冗余，但尚未证实。
- 答案匹配筛选不能验证 query 是否瞄准缺失事实、Information 是否被使用或多跳链是否成立。
- 原始 dev/test 混入 Teacher 候选可能污染未来外部 benchmark。

### 尚无法确认

- 当前最终 KEEP/Standard 中具体保留了哪些 NQ/HotpotQA 轨迹。
- 用户提到的 30 条人工审查的逐条结论；本地报告目录不存在。
- empty-visible 是否由 reasoning token 消耗导致。
- 19 条近重复 Query 候选中多少是真正无效重复。

## 7. Teacher v2 最小改进建议（最多三项）

### P1：Teacher-only Prompt 层——强化“缺失事实—查询—证据—停止”闭环

不要修改共享的 `AGENT_SYSTEM_PROMPT`。为 Teacher v2 单独增加附加 prompt 或独立 Teacher prompt，要求内部决策遵守：

1. Search Query 只针对当前尚缺失的一条实体/关系；
2. 收到 Information 后先判断它是否填补该缺口，再决定下一跳；
3. evidence 不足时禁止猜测 Answer；
4. evidence 已足够时禁止冗余搜索并立即回答。

这是 Prompt 层单变量改动，目标是 query relevance、evidence utilization 和 stopping，不要求输出可见 reasoning 或引用。

### P2：API 参数层——单独消融 reasoning effort

保留 prompt、temperature=0、max_tokens=4096、BM25 top-3 和 max-turns=5，只把 `reasoning_effort` 从 `high` 改为一个 API 实际支持的较低档位进行小规模配对 pilot。不要同时调整 max_tokens，否则无法区分 Think 变短来自 effort 还是硬截断。评价 empty-visible、Answer 率、Think tokens、EM 和人工语义质量后再决定是否采用。

### P3：质量筛选层——答案匹配之外增加行为风险隔离

保留现有高精度答案匹配，同时把以下样本先标记/隔离而非自动判错：

- exact 重复 query；
- 高词面相似 query；
- Think >1,000 Qwen raw-content tokens；
- 零搜索且 Think >1,000 tokens；
- Answer 虽匹配 Gold，但 Information 中缺少可定位支撑的多跳样本。

前四项可程序化产生 review queue；最后一项必须人工或独立 evaluator 判定。这样避免把“答案碰巧正确但轨迹不可教”直接送入 SFT。

## 8. 可复现的 v1/v2 对照实验草案

### 8.1 冻结候选

- 从 `nq_train_*` 和 `hotpotqa_train_*` 中各取 150 条，共 300 条。
- 每来源按 v1 的 0-search、1-search、2+-search 三层各 50 条；层内按 UID 排序后用 `random.Random(123)` shuffle。
- 排除未来 Agent Evaluation 的 ID 和规范化 question；若评测集尚未冻结，先不启动 API 实验。
- 保存候选 ID manifest、question/Gold hash、BM25 index/corpus hash、代码 commit 和 prompt hash。

### 8.2 实验臂

| Arm | Prompt | reasoning_effort | 其他参数 |
|---|---|---|---|
| v1 baseline | 当前共享 prompt | high | temperature=0, max_tokens=4096, top-k=3, max-turns=5 |
| v2-P | Teacher-only 闭环 prompt | high | 其余与 v1 完全相同 |
| v2-E（仅在 v2-P 后） | 与 v2-P 相同 | 较低支持档位 | 其余不变 |

先比较 v1 与 v2-P；只有 Prompt 改进仍存在明显 Think 长尾时，才运行 v2-E。不要一次同时改变 prompt、effort 和 max_tokens。

### 8.3 输出隔离与恢复

- 使用新的版本化输出文件，例如 `teacher_v2_prompt_pilot.jsonl`，绝不复用 v1 raw 路径。
- 沿用逐样本 flush/resume，但 manifest 必须记录已完成/empty/异常 ID。
- v1 若不重新调用 API，可使用既有 raw 对应 ID；需要明确这不是同时间 API snapshot 的严格复现。

### 8.4 指标

程序指标：

- empty/invalid/max-turn/Answer 率；
- normalized/high-precision answer match；
- search_count 分布；
- exact/近重复 query 候选；
- Think 字符和 Qwen raw-content token 的 Median/P90/P95；
- >1,000 Think token 和零搜索长 Think 数。

盲人工指标（同一 ID 成对、隐藏 arm）：

- Query 是否瞄准真正缺失的信息；
- Information 是否支持下一步；
- 多跳实体链是否成立；
- evidence 不足时是否猜测；
- evidence 足够后是否及时停止；
- Answer 是否正确且由 evidence 支撑。

### 8.5 决策门槛

v2 不能只靠 Think 变短通过。至少要求 Answer/高精度匹配不下降，empty/invalid/max-turn 不恶化，并在盲审中减少无效 query、证据断链和错误停止。高风险定向样本与随机样本需分开报告，不能把定向样本错误率外推到总体。

## 9. 下一步必须由用户确认

1. 恢复还是正式删除当前工作树中缺失的 audit/filter/build 脚本；在此之前复现链不完整。
2. 提供或恢复最终 KEEP/Standard/Validation 文件与 30 条人工审查结果，才能核对当前正式训练数据，而不是只审计旧 clean。
3. 冻结 Agent Evaluation 的 ID/question 集，确认是否排除所有 NQ dev/test 与 HotpotQA dev。
4. 确认 Teacher v2 首个单变量实验选择 P1（Teacher-only prompt）还是 P2（reasoning effort）；建议先 P1。
