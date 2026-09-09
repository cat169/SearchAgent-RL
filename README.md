# SearchAgent-RL

基于 Search-R1 思想的大语言模型自主检索 Agent 后训练项目。

本项目探索如何通过 **Teacher-generated Agent Trajectory、Agentic SFT
以及 GRPO
强化学习**，提升大语言模型在开放域问答任务中的自主检索、工具调用和多步推理能力。

------------------------------------------------------------------------

## 项目简介

近年来，大语言模型（Large Language Models,
LLMs）在自然语言理解和文本生成任务中取得了显著进展。

然而，传统 LLM
主要依赖模型参数中存储的知识进行回答，在面对复杂问题时仍存在以下局限：

-   知识更新困难；
-   缺少主动获取外部信息的能力；
-   无法自主判断何时调用外部工具；
-   缺少多步检索和推理能力。

Search-R1 提出了一种基于强化学习的搜索 Agent
训练范式，使语言模型能够自主决定是否调用搜索工具，并通过强化学习优化搜索与推理过程。

本项目基于 Search-R1 的核心思想，构建一个完整的大语言模型 Agent
后训练流程，通过 Agentic SFT + GRPO
的方式，使小规模语言模型获得自主检索能力。

------------------------------------------------------------------------

# 项目目标

本项目主要目标：

1.  构建具备自主检索能力的大语言模型 Agent；
2.  利用 Teacher LLM 生成高质量 Agent trajectory 数据；
3.  通过 Agentic SFT 训练模型学习基础工具调用能力；
4.  通过 GRPO 强化学习优化模型搜索决策能力；
5.  建立完整的大模型 Agent 后训练实验流程。

整体训练流程：

    基础语言模型
            ↓
    Teacher-generated Agent Trajectory
            ↓
    Agentic SFT
            ↓
    GRPO Reinforcement Learning
            ↓
    Autonomous Search Agent

------------------------------------------------------------------------

# 技术路线

整体流程：

    开放域问答数据集
    (NQ / HotpotQA)

            ↓

    数据清洗与划分

            ↓

    Teacher LLM生成Agent轨迹

            ↓

    Agentic SFT

            ↓

    GRPO强化学习优化

            ↓

    模型评估

------------------------------------------------------------------------

# 核心模块

## 1. 数据构建

基于开放域问答数据集构建 Agent 训练数据。

主要包括：

-   数据加载；
-   数据清洗；
-   重复样本过滤；
-   数据划分；
-   SFT 与 GRPO 数据构造。

目标：

将传统 QA 数据转换为适用于 Agent 后训练的数据格式。

------------------------------------------------------------------------

## 2. Search Agent 环境

构建语言模型与搜索工具交互环境。

Agent 能够：

-   分析当前问题；
-   判断是否需要搜索；
-   生成搜索请求；
-   接收检索结果；
-   根据外部信息完成最终回答。

------------------------------------------------------------------------

## 3. Agentic SFT

传统监督微调通常学习：

    Question → Answer

本项目进一步构造包含工具调用过程的 Agent trajectory 数据：

    Question

    ↓

    <think>
    问题分析
    </think>

    ↓

    <search>
    搜索请求
    </search>

    ↓

    <information>
    检索结果
    </information>

    ↓

    <answer>
    最终答案
    </answer>

通过监督微调，使模型学习：

-   工具调用格式；
-   搜索行为；
-   信息利用方式；
-   多步推理流程。

------------------------------------------------------------------------

## 4. GRPO 强化学习

在 Agentic SFT 基础上，引入 Group Relative Policy Optimization（GRPO）。

训练过程中：

模型针对同一问题生成多个候选轨迹：

    Question

    ↓

    Trajectory 1

    Trajectory 2

    Trajectory 3

根据任务完成情况计算奖励：

    Trajectory

    ↓

    Reward

    ↓

    Advantage

    ↓

    GRPO Update

进一步优化：

-   搜索策略；
-   工具调用决策；
-   推理过程；
-   最终回答质量。

------------------------------------------------------------------------

# 实验设计

为了分析不同训练阶段的作用，计划设计以下实验：

  实验                 训练方式                    目的
  -------------------- --------------------------- -------------------------
  Base Model           未训练基础模型              测试初始能力
  QA-SFT               普通问答监督微调            分析传统 SFT 效果
  Agentic-SFT          Agent trajectory 监督微调   验证 Agent 行为学习效果
  Agentic-SFT + GRPO   SFT 后强化学习              验证 RL 优化效果

重点评估：

-   Answer Accuracy；
-   Search Success Rate；
-   Tool Calling Ability；
-   Multi-step Reasoning Performance。

------------------------------------------------------------------------

# 项目结构

    SearchAgent-RL

    ├── data/
    ├── agent/
    ├── sft/
    ├── grpo/
    ├── evaluation/
    ├── configs/
    ├── scripts/
    └── docs/

------------------------------------------------------------------------

# 技术栈

-   Python
-   PyTorch
-   HuggingFace Transformers
-   PEFT (LoRA)
-   HuggingFace TRL
-   Large Language Model Agent
-   Reinforcement Learning

------------------------------------------------------------------------

# 项目进展

## 已完成

-   Search-R1整体流程学习
-   Agent Loop理解
-   Retriever交互机制理解
-   Agent trajectory理解
-   GRPO算法理解

## 开发中

-   数据处理 Pipeline
-   Agent 环境构建
-   Teacher trajectory 生成
-   Agentic SFT 训练
-   GRPO 训练
-   实验分析

------------------------------------------------------------------------

# References

-   Search-R1: An Efficient, Scalable RL Training Framework for
    Reasoning & Search Engine Calling Interleaved LLM
-   DeepSeek-R1
-   HuggingFace Transformers
-   HuggingFace TRL
