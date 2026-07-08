# Project Brief

## Title

Graph-based RAG Frameworks: Reproduction, Comparative Study and Ablation Analysis

## Positioning

这是一个面向保研简历和科研训练的 RAG 复现与实验分析项目。项目不以完成普通问答应用为目标，而是以复现、对比、消融和改进为核心，展示论文阅读、系统实现、实验设计、指标分析和工程复现能力。

## Core Idea

围绕 LightRAG 与 GraphRAG-style 方法，构建统一实验框架。通过相同数据集、相同查询集合、相同评价指标比较不同 RAG 方法，并系统分析 chunk、embedding、retriever、top-k、reranker、query rewrite 和 LLM backbone 对结果的影响。

## Scope

第一阶段只完成可跑通、可评估、可展示的最小闭环：

- Naive RAG
- Vector RAG
- LightRAG reproduction or controlled integration
- HotpotQA small subset
- Retrieval metrics
- Basic answer quality metrics
- First experiment report

第二阶段再扩展：

- GraphRAG-style baseline
- Hybrid retriever
- Reranker
- Query rewrite
- More datasets
- Full README charts

## Non-goals

- 不训练大模型。
- 不追求完整复现微软 GraphRAG 的全部工程。
- 不把重点放在 Web UI。
- 不人工构造大规模数据集。
- 不追求一次性覆盖所有 embedding、LLM 和 retriever 组合。

## Expected Outputs

- 可复现实验代码
- 实验配置文件
- 评测结果 CSV/JSON
- 可视化图表
- GitHub README
- 技术报告
- 简历项目描述

## Resume Value

该项目能体现：

- 对 RAG 与 graph-based retrieval 的理解
- 复现论文方法的能力
- 设计公平对比实验的能力
- 使用公开数据集和指标进行评估的能力
- 对实验结果进行分析和解释的能力
- 对开源项目进行工程化整理的能力

