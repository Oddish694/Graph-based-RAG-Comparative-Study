# 项目简介 / Project Brief

## 项目名称

Graph-based RAG Frameworks: Reproduction, Comparative Study and Ablation Analysis

中文名称：图增强 RAG 框架复现、对比实验与消融分析

## 项目定位

这是一个面向科研训练、保研简历和工程展示的 RAG 实验项目。项目重点不是搭建一个普通问答应用，而是设计一套可复现的实验框架，用统一数据集、统一配置和统一指标比较不同 RAG 方法的效果。

项目希望体现的能力包括：

- 阅读和复现 RAG / GraphRAG / LightRAG 相关方法
- 设计公平对比实验
- 使用公开数据集进行评估
- 实现可复现的实验脚本和配置系统
- 分析检索质量、回答质量、延迟和计算成本
- 将实验结果整理成 README、图表和技术报告

English summary: This project is a reproducible experimental framework for comparing vector-only and graph-enhanced RAG methods on multi-hop QA tasks.

## 核心研究问题

本项目主要想回答：

> 在多跳问答任务中，图增强 RAG 是否比普通向量 RAG 更能找全问题所需证据？

更具体地说：

1. Vector RAG 只依赖 embedding 相似度，它在 HotpotQA 上能召回多少 supporting facts？
2. GraphRAG-style RAG 通过实体和关系构建图，是否能提升多跳证据覆盖率？
3. LightRAG 的轻量图增强检索是否能在质量和延迟之间取得更好平衡？
4. chunk size、embedding model、top-k、reranker、query rewrite 等因素分别影响哪些指标？

## 当前范围

当前已经完成第一阶段，即最小可复现实验闭环：

- HotpotQA 小样本加载
- QASample 统一数据结构
- fixed-size chunking
- Vector RAG baseline
- Recall@k、MRR、Hit Rate 指标
- YAML 配置驱动实验
- CSV/JSON 结果输出
- 单元测试

后续阶段将继续扩展：

- Hybrid RAG
- BM25 retriever
- GraphRAG-style baseline
- LightRAG controlled runner
- reranker improvement
- query rewrite improvement
- chunk/top-k/embedding 消融实验
- 表格、图表和最终报告

## 非目标 / Non-goals

本项目暂时不做：

- 不训练大模型
- 不追求完整复现微软 GraphRAG 的全部工程系统
- 不把重点放在 Web UI
- 不人工构造大规模数据集
- 不一次性覆盖所有 embedding、LLM 和 retriever 组合

## 预期产出

- 可运行代码
- 实验配置文件
- HotpotQA 小样本检索结果
- per-query CSV
- aggregate metrics JSON
- 可视化图表
- GitHub README
- 技术报告
- 简历项目描述

## 简历价值

这个项目可以体现：

- 对 RAG、dense retrieval、graph-based retrieval 的理解
- 对多跳问答任务的理解
- 设计 baseline 和公平实验的能力
- 使用公开数据集和标准指标评估模型的能力
- 把研究想法工程化、配置化、可复现化的能力
- 对实验结果进行解释和复盘的能力
