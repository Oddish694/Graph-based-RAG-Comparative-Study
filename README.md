# Graph-based RAG 框架复现、对比实验与消融分析

本项目是一个面向科研训练和简历展示的 RAG 实验框架。项目目标不是做一个简单的知识库问答 demo，而是在同一套数据、同一套评估指标、同一套实验配置下，比较不同 RAG 方法在多跳问答任务上的检索效果、回答质量、延迟和成本，最后分析图增强 RAG 是否相比普通向量 RAG 更有优势。

English summary: This project builds a reproducible experimental framework for comparing Vector RAG, Hybrid RAG, GraphRAG-style RAG, LightRAG and improved RAG variants on multi-hop question answering tasks.

## 项目核心问题

本项目围绕下面几个问题展开：

1. 普通 Vector RAG 在多跳问答任务中能找回多少正确证据？
2. GraphRAG-style RAG 或 LightRAG 是否能比普通向量检索更好地覆盖多跳证据？
3. chunk size、embedding model、retriever、top-k、reranker、query rewrite 等组件分别会怎样影响检索质量和系统延迟？
4. 如果图增强方法效果更好，它带来的索引成本、查询延迟和实现复杂度是否值得？

## 当前实现状态

当前已经完成 Phase 1：HotpotQA 小样本上的 Vector RAG baseline。

Phase 1 已实现：

- HotpotQA 数据加载与 JSONL 缓存
- 统一 QA 样本结构 `QASample`
- fixed-size chunking
- hashing embedding smoke test
- sentence-transformers embedding adapter
- 本地向量索引 `VectorIndex`
- Vector RAG top-k 检索器
- Recall@k、MRR@k、Hit Rate@k 检索指标
- YAML 配置驱动实验
- per-query CSV 和 aggregate JSON 结果输出
- 单元测试覆盖核心模块

Phase 1 真实实验结果：

| Setting | Value |
| --- | --- |
| Dataset | HotpotQA validation |
| Sample size | 100 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Retriever | Vector RAG / dense retrieval |
| Top-k | 5 |
| Recall@5 | 0.74 |
| MRR@5 | 0.8275 |
| Hit Rate@5 | 0.95 |

这些结果说明：普通向量检索通常能在 top-5 中命中至少一个相关证据，但在多跳问答中仍然不能完全找齐所有 supporting facts。这正是后续 GraphRAG / LightRAG 需要尝试改进的地方。

## 方法对比范围

| Method | 中文说明 | English description |
| --- | --- | --- |
| Naive RAG | 最基础的 chunk 检索加生成流程 | Basic chunk retrieval plus generation |
| Vector RAG | 基于 embedding 相似度的向量检索 baseline | Dense embedding retrieval baseline |
| Hybrid RAG | BM25 关键词检索加向量检索融合 | BM25 plus dense retrieval |
| GraphRAG-style RAG | 抽取实体关系并构建图，再进行图感知检索 | Entity-relation graph construction and graph-aware retrieval |
| LightRAG | 复现或集成 LightRAG 的轻量图增强检索流程 | Lightweight graph-enhanced retrieval |
| Improved LightRAG | 在 LightRAG 上加入 reranker、query rewrite 或 HyDE 等增强 | LightRAG with reranking or query enhancement |

## 数据集计划

| Dataset | 用途 |
| --- | --- |
| HotpotQA small subset | 多跳问答数据，带 supporting facts，适合评估证据召回 |
| 2WikiMultiHopQA small subset | 多跳路径更显式，适合测试图结构检索 |
| BEIR subset | 标准检索 benchmark，适合比较 retriever 本身 |

## 指标体系

Retrieval metrics：

- Recall@k：top-k 中找回了多少 gold evidence
- Precision@k：top-k 中有多少结果是相关证据
- MRR：第一个相关证据排名是否靠前
- NDCG：相关证据是否排在更靠前的位置
- Hit Rate：top-k 中是否至少命中一个 supporting context

Generation metrics：

- Exact Match
- F1
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

Efficiency metrics：

- Index construction time
- Retrieval latency
- Query latency
- Token usage
- Storage size

## 项目结构

```text
.
├── README.md
├── configs/
│   ├── experiment_matrix.yaml
│   ├── phase1_vector_rag.yaml
│   └── phase1_vector_rag_sentence_transformer.yaml
├── docs/
│   ├── experiment_plan.md
│   ├── implementation_plan.md
│   ├── phase1_details.md
│   ├── project_brief.md
│   └── resume_notes.md
├── experiments/
│   └── run_phase1_vector_rag.ps1
├── src/
│   ├── chunking/
│   ├── datasets/
│   ├── evaluation/
│   ├── indexing/
│   ├── retrieval/
│   └── run_experiment.py
└── tests/
```

## 快速运行

创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

运行 smoke test 配置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase1_vector_rag.ps1 -Python .\.venv\Scripts\python.exe
```

运行真实 HotpotQA + sentence-transformers 配置：

```powershell
.\.venv\Scripts\python.exe -m src.run_experiment --config configs\phase1_vector_rag_sentence_transformer.yaml
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 简历表述

中文：

> 构建面向 HotpotQA 多跳问答的 RAG 对比实验框架，实现数据标准化、文本切块、向量检索、检索指标评估与配置化实验流程；在 100 条 HotpotQA validation 样本上完成 Vector RAG baseline，取得 Recall@5 0.74、MRR@5 0.8275、Hit Rate@5 0.95，为后续 GraphRAG / LightRAG 对比实验提供可复现基线。

English:

> Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, including dataset normalization, chunking, dense vector retrieval, retrieval metrics and YAML-driven experiments. Established a Vector RAG baseline with Recall@5 of 0.74, MRR@5 of 0.8275 and Hit Rate@5 of 0.95 on 100 HotpotQA validation samples.
