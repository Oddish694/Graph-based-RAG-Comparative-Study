# Graph-based RAG Comparative Study / 图增强 RAG 对比实验框架

本项目构建一个可复现的 RAG 对比实验框架，用于在同一套数据、同一套评估指标、同一套实验配置下，比较 Vector RAG、BM25、Hybrid RAG、GraphRAG-style RAG、LightRAG 以及改进版 LightRAG 在多跳问答任务上的表现。

English summary: This project builds a reproducible experimental framework for comparing dense vector retrieval, lexical retrieval, hybrid retrieval and graph-enhanced RAG methods on multi-hop question answering tasks.

## Project Goal / 项目目标

项目最终要回答的问题是：

> 在 HotpotQA / 2WikiMultiHopQA 这类多跳问答任务上，图增强 RAG 是否比普通向量 RAG 更能找全证据？如果有效，它带来的质量提升是否值得额外的索引成本、查询延迟和工程复杂度？

对比维度包括：

- 检索效果：Recall@k、MRR@k、Hit Rate@k、后续可扩展 NDCG / Precision@k。
- 回答质量：后续接入生成模型后评估 EM、F1、Faithfulness、Answer Relevance。
- 系统效率：索引构建时间、检索延迟、端到端查询延迟、存储成本、token 使用量。
- 工程复杂度：是否需要实体抽取、关系抽取、图构建、reranker 或 query rewrite。

## Current Status / 当前状态

当前已经完成：

- Phase 1: HotpotQA 小样本 + Vector RAG baseline。
- Phase 2: BM25 lexical retrieval + Hybrid RAG baseline。

Phase 1 建立了 dense retrieval 对照组。Phase 2 在相同 HotpotQA subset 上新增 BM25 和 Hybrid RAG，使后续 GraphRAG / LightRAG 的收益可以和普通向量检索、关键词检索、混合检索分别比较。

## Implemented Features / 已实现内容

- HotpotQA validation 数据加载与 JSONL 缓存。
- 统一 QA 样本结构 `QASample`。
- fixed-size chunking，并保留 `sample_id`、`doc_id`、`chunk_id` 等元数据。
- hashing embedding，用于离线 smoke test。
- sentence-transformers embedding adapter，用于真实实验。
- 本地向量索引 `VectorIndex`。
- BM25 lexical index `BM25Index`。
- Vector RAG、BM25 RAG、Hybrid RAG retriever。
- Hybrid RAG 支持 weighted score fusion 和 reciprocal rank fusion。
- Recall@k、MRR@k、Hit Rate@k 检索指标。
- YAML 配置驱动实验。
- per-query CSV 和 aggregate JSON 结果输出。
- 单元测试覆盖 loader、chunker、retriever、metrics、runner。

## Results / 当前实验结果

实验设置：

| Setting | Value |
| --- | --- |
| Dataset | HotpotQA validation |
| Sample size | 100 |
| Chunk size / overlap | 64 / 8 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Top-k | 5 |

结果：

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit Rate@5 | Avg Retrieval Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.375 | 0.650 | 0.740 | 0.8275 | 0.950 | previous Phase 1 run |
| BM25 | 0.355 | 0.595 | 0.710 | 0.8200 | 0.990 | 0.0046s |
| Hybrid RAG | 0.410 | 0.690 | 0.785 | 0.8908 | 0.980 | 0.0158s |

初步结论：Hybrid RAG 在这批 100 条 HotpotQA validation 样本上比单独 Vector RAG 和单独 BM25 获得更高的 Recall@5 和 MRR@5，说明 BM25 的关键词精确匹配和 dense retrieval 的语义匹配有互补性。但 Hybrid 检索延迟高于 BM25，因为它同时执行 lexical 和 dense 两路检索。

## Roadmap / 阶段路线

| Phase | Method | Goal | Status |
| --- | --- | --- | --- |
| Phase 1 | Vector RAG baseline | 建立普通 dense retrieval 对照组 | 已完成 |
| Phase 2 | BM25 + Hybrid RAG baseline | 比较 lexical、dense、hybrid retrieval | 已完成基础版 |
| Phase 3 | GraphRAG-style RAG | 构建轻量实体关系图，测试图检索是否提升多跳证据召回 | 待实现 |
| Phase 4 | LightRAG controlled integration | 将 LightRAG 纳入统一数据、query、指标框架 | 待实现 |
| Phase 5 | Improved LightRAG / improved graph retrieval | 加入 reranker、query rewrite 或 HyDE 等增强策略 | 待实现 |
| Phase 6 | Ablation and reporting | 做 chunk size、top-k、embedding、retriever、reranker 等消融实验并整理报告 | 待实现 |

更细的工程任务拆分见 [docs/implementation_plan.md](docs/implementation_plan.md)。Phase 1 细节见 [docs/phase1_details.md](docs/phase1_details.md)，Phase 2 细节见 [docs/phase2_details.md](docs/phase2_details.md)。

## Project Structure / 项目结构

```text
README.md
configs/
  phase1_vector_rag.yaml
  phase1_vector_rag_sentence_transformer.yaml
  phase2_bm25_rag.yaml
  phase2_hybrid_rag.yaml
  phase2_hybrid_rag_sentence_transformer.yaml
docs/
  implementation_plan.md
  phase1_details.md
  phase2_details.md
  project_brief.md
  resume_notes.md
experiments/
  run_phase1_vector_rag.ps1
  run_phase2_hybrid_rag.ps1
src/
  chunking/
  datasets/
  evaluation/
  indexing/
    bm25_index.py
    vector_index.py
  retrieval/
    bm25_rag.py
    hybrid_rag.py
    vector_rag.py
  run_experiment.py
tests/
```

## Quick Start / 快速运行

创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

运行 Phase 1 smoke test：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase1_vector_rag.ps1 -Python .\.venv\Scripts\python.exe
```

运行 Phase 2 Hybrid smoke test：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase2_hybrid_rag.ps1 -Python .\.venv\Scripts\python.exe
```

运行真实 HotpotQA + sentence-transformers Hybrid 配置：

```powershell
.\.venv\Scripts\python.exe -m src.run_experiment --config configs\phase2_hybrid_rag_sentence_transformer.yaml
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Resume Description / 简历表述

中文：

> 构建面向 HotpotQA 多跳问答的可复现 RAG 对比实验框架，统一数据标准化、文本切块、向量检索、BM25 关键词检索、Hybrid score fusion 和 Recall@k / MRR / Hit Rate 指标评估。在 100 条 HotpotQA validation 样本上完成 Vector RAG、BM25 与 Hybrid RAG baseline 对比，Hybrid RAG 取得 Recall@5 0.785、MRR@5 0.8908，相比 Vector RAG 的 Recall@5 0.74 展示出关键词匹配与语义检索的互补性，为后续 GraphRAG-style RAG 和 LightRAG 对比实验提供统一 baseline。

English:

> Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, including dataset normalization, chunking, dense retrieval, BM25 retrieval, hybrid score fusion and retrieval metrics such as Recall@k, MRR and Hit Rate. Compared Vector RAG, BM25 and Hybrid RAG baselines on 100 HotpotQA validation samples, where Hybrid RAG achieved Recall@5 of 0.785 and MRR@5 of 0.8908, providing controlled baselines for later GraphRAG-style and LightRAG experiments.
