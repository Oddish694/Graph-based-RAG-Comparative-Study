# Graph-based RAG Comparative Study / 图增强 RAG 对比实验框架

本项目构建一个可复现的 RAG 对比实验框架，用于在同一套数据、同一套评估指标、同一套实验配置下，比较 Vector RAG、BM25、Hybrid RAG、GraphRAG-style RAG、LightRAG 以及改进版图检索方法在多跳问答任务上的表现。

English summary: This project builds a reproducible experimental framework for comparing dense vector retrieval, lexical retrieval, hybrid retrieval and graph-enhanced RAG methods on multi-hop question answering tasks.

## Project Goal / 项目目标

项目最终要回答的问题是：

> 在 HotpotQA / 2WikiMultiHopQA 这类多跳问答任务上，图增强 RAG 是否比普通向量检索和 Hybrid 检索更能找全 supporting facts？如果有效，它带来的质量提升是否值得额外的图构建成本、查询延迟和工程复杂度？

当前优先完成 retrieval-side research loop，也就是先研究检索阶段是否能找全多跳证据链，再考虑后续生成答案质量评估。

## Current Status / 当前状态

当前已经完成：

- Phase 1: HotpotQA 小样本 + Vector RAG baseline。
- Phase 2: BM25 lexical retrieval + Hybrid RAG baseline。
- Phase 2.5: fair baseline configs + upgraded retrieval metrics。
- Phase 3: lightweight GraphRAG-style baseline。
- Phase 4: Evidence-aware Improved LightRAG baseline。
- Phase 4.5: LightRAG controlled integration layer。

Phase 2.5 统一了 Vector / BM25 / Hybrid 的 chunking、top-k、数据集和指标设置，并补充 `Precision@k`、`NDCG@k`、`Evidence Hit Count@k`、`Full Evidence Recall@k` 和 `Retrieved Context Tokens`。Phase 3 在此基础上实现了轻量 GraphRAG-style retriever，用实体图邻居扩展测试图增强检索是否能提升多跳证据召回。Phase 4 进一步加入 coverage-aware reranking，形成 Improved LightRAG-style retriever。Phase 4.5 新增 LightRAG controlled integration layer，用统一 schema 接收 LightRAG-style 对照组结果。

## Implemented Features / 已实现内容

- HotpotQA validation 数据加载与 JSONL 缓存。
- 统一 QA 样本结构 `QASample`。
- fixed-size chunking，并保留 `sample_id`、`doc_id`、`chunk_id` 等元数据。
- hashing embedding，用于离线 smoke test。
- sentence-transformers embedding adapter，用于真实实验。
- 本地向量索引 `VectorIndex`。
- BM25 lexical index `BM25Index`。
- Vector RAG、BM25 RAG、Hybrid RAG retriever。
- 轻量 GraphRAG-style retriever：规则实体抽取、entity-chunk graph、entity co-occurrence graph、graph neighbor expansion。
- LightRAG controlled integration：统一结果 schema、`local_compat` 后端和外部 runner 接口。
- Retrieval metrics: Recall@k、Precision@k、MRR@k、NDCG@k、Hit Rate@k、Evidence Hit Count@k、Full Evidence Recall@k、Retrieved Context Tokens。
- YAML 配置驱动实验。
- per-query CSV 和 aggregate JSON 结果输出。
- 单元测试覆盖 loader、chunker、retriever、graph index、metrics、runner。

## Results / 当前实验结果

统一设置：

| Setting | Value |
| --- | --- |
| Dataset | HotpotQA validation |
| Sample size | 100 |
| Seed | 42 |
| Chunk size / overlap | 64 / 8 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Top-k | 10 |
| Metric k values | 1, 3, 5, 10 |

主结果：

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.725 | 0.290 | 0.6901 | 0.8325 | 0.510 | 0.940 | 0.0108s |
| BM25 | 0.710 | 0.284 | 0.6668 | 0.8200 | 0.430 | 0.990 | 0.0053s |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |
| GraphRAG-style | 0.805 | 0.322 | 0.7727 | 0.9125 | 0.620 | 0.990 | 0.0866s |
| Improved LightRAG | 0.820 | 0.328 | 0.7851 | 0.9158 | 0.650 | 0.990 | 0.0584s |
| LightRAG controlled | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0199s |

Top-10 evidence coverage:

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.835 | 0.167 | 0.7339 | 0.670 | 1.000 |
| BM25 | 0.885 | 0.177 | 0.7356 | 0.770 | 1.000 |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |
| GraphRAG-style | 0.905 | 0.181 | 0.8129 | 0.810 | 1.000 |
| Improved LightRAG | 0.910 | 0.182 | 0.8209 | 0.820 | 1.000 |
| LightRAG controlled | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |

结论：Hybrid RAG 是强 baseline；增强后的 GraphRAG-style 在 top-10 证据覆盖上进一步超过 Hybrid。Improved LightRAG 通过 coverage-aware reranking 将 Recall@5 提升到 0.820、Full Evidence Recall@5 提升到 0.650、NDCG@10 提升到 0.8209，为后续消融实验提供了更有说服力的核心方法。LightRAG controlled 当前使用 `local_compat` 后端，作用是验证外部 LightRAG 对照组的统一接入流程，不能当作官方 LightRAG 最终成绩。

## Roadmap / 阶段路线

| Phase | Method | Goal | Status |
| --- | --- | --- | --- |
| Phase 1 | Vector RAG baseline | 建立普通 dense retrieval 对照组 | 已完成 |
| Phase 2 | BM25 + Hybrid RAG baseline | 比较 lexical、dense、hybrid retrieval | 已完成 |
| Phase 2.5 | Fair baselines + upgraded metrics | 统一 baseline 配置并补齐多跳证据指标 | 已完成 |
| Phase 3 | GraphRAG-style RAG | 构建轻量实体关系图，测试图检索是否提升多跳证据召回 | 已完成可复现实验版本 |
| Phase 4 | Evidence-aware Improved LightRAG | 图邻居扩展 + 证据覆盖感知重排序 | 已完成可复现实验版本 |
| Phase 4.5 | LightRAG controlled integration | 将 LightRAG 对照组纳入统一评估接口 | 已完成可控接入层 |
| Phase 5 | Ablation experiments | 验证图扩展、coverage reranking 等模块贡献 | 下一步 |
| Phase 6 | Scale-up + case study | 扩大样本规模并分析成功/失败案例 | 待实现 |
| Phase 7 | Final report | 整理报告、图表和简历材料 | 待实现 |

更细的工程任务拆分见 [docs/implementation_plan.md](docs/implementation_plan.md)。Phase 2.5 细节见 [docs/phase2_5_fair_baselines.md](docs/phase2_5_fair_baselines.md)，Phase 3 细节见 [docs/phase3_graph_rag_style.md](docs/phase3_graph_rag_style.md)，Phase 4 细节见 [docs/phase4_improved_lightrag.md](docs/phase4_improved_lightrag.md)，Phase 4.5 细节见 [docs/phase4_5_lightrag_controlled.md](docs/phase4_5_lightrag_controlled.md)。

## Project Structure / 项目结构

```text
README.md
configs/
  phase2_vector_rag_fair.yaml
  phase2_bm25_rag_fair.yaml
  phase2_hybrid_rag_fair.yaml
  phase3_graph_rag_style.yaml
  phase4_5_lightrag.yaml
docs/
  implementation_plan.md
  phase2_5_fair_baselines.md
  phase3_graph_rag_style.md
  phase4_improved_lightrag.md
  phase4_5_lightrag_controlled.md
  resume_notes.md
experiments/
  run_fair_baselines.ps1
  run_phase3_graph_rag_style.ps1
  run_phase4_improved_lightrag.ps1
  run_phase4_5_lightrag.ps1
src/
  graph/
    entity_extractor.py
    graph_builder.py
  indexing/
    bm25_index.py
    graph_index.py
    vector_index.py
  retrieval/
    bm25_rag.py
    graph_rag_style.py
    improved_lightrag.py
    lightrag_runner.py
    coverage_reranker.py
    hybrid_rag.py
    vector_rag.py
  run_experiment.py
tests/
```

## Quick Start / 快速运行

运行 Phase 2.5 fair baselines：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_fair_baselines.ps1 -Python .\.venv\Scripts\python.exe
```

运行 Phase 3 GraphRAG-style：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase3_graph_rag_style.ps1 -Python .\.venv\Scripts\python.exe
```

运行 Phase 4 Improved LightRAG：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase4_improved_lightrag.ps1 -Python .\.venv\Scripts\python.exe
``` 

运行 Phase 4.5 LightRAG controlled integration：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase4_5_lightrag.ps1 -Python .\.venv\Scripts\python.exe
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Resume Description / 简历表述

中文：

> 构建面向 HotpotQA 多跳问答的可复现 RAG 对比实验框架，统一数据标准化、文本切块、Vector RAG、BM25 关键词检索、Hybrid score fusion 和多维检索指标评估。在统一 chunking、top-k 和数据配置下完成 Vector / BM25 / Hybrid fair baseline 对比，并新增 Precision@k、NDCG@k、Full Evidence Recall@k 等多跳证据完整性指标。进一步实现增强版 GraphRAG-style retriever，通过规则实体抽取、entity-chunk graph、实体频率加权和图距离衰减评分，将 Recall@10 提升到 0.905、Full Evidence Recall@10 提升到 0.810；实现 Improved LightRAG-style coverage reranking，将 Recall@5 提升到 0.820、Full Evidence Recall@5 提升到 0.650、NDCG@10 提升到 0.8209，并补充 LightRAG controlled integration layer，为后续外部方法对照和消融实验提供统一评估接口。

English:

> Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, including dataset normalization, chunking, Vector RAG, BM25 retrieval, hybrid score fusion and retrieval metrics. Established fair Vector / BM25 / Hybrid baselines under unified chunking, top-k and dataset settings, and added multi-hop evidence metrics such as Precision@k, NDCG@k and Full Evidence Recall@k. An enhanced GraphRAG-style retriever improved Recall@10 to 0.905 and Full Evidence Recall@10 to 0.810 through rule-based entity extraction, an entity-chunk graph, entity-frequency weighting and graph-distance decay. An Improved LightRAG-style coverage reranker improved Recall@5 to 0.820, Full Evidence Recall@5 to 0.650 and NDCG@10 to 0.8209, with a LightRAG controlled integration layer added for external method comparison.


