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

Phase 2.5 统一了 Vector / BM25 / Hybrid 的 chunking、top-k、数据集和指标设置，并补充 `Precision@k`、`NDCG@k`、`Evidence Hit Count@k`、`Full Evidence Recall@k` 和 `Retrieved Context Tokens`。这使后续 GraphRAG-style / Improved LightRAG 的对比更可信。

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
- Retrieval metrics: Recall@k、Precision@k、MRR@k、NDCG@k、Hit Rate@k、Evidence Hit Count@k、Full Evidence Recall@k、Retrieved Context Tokens。
- YAML 配置驱动实验。
- fair baseline 一键运行脚本。
- per-query CSV 和 aggregate JSON 结果输出。
- 单元测试覆盖 loader、chunker、retriever、metrics、runner。

## Fair Baseline Results / 公平基线结果

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

Top-10 evidence coverage:

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.835 | 0.167 | 0.7339 | 0.670 | 1.000 |
| BM25 | 0.885 | 0.177 | 0.7356 | 0.770 | 1.000 |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |

结论：Hybrid RAG 在统一配置下仍然是当前最强 baseline。它在 Recall@5、NDCG@5、MRR@5 和 Full Evidence Recall@5 上都优于 Vector RAG 和 BM25，说明关键词匹配与语义检索对多跳证据召回有互补性。后续 GraphRAG-style 和 Improved LightRAG 应重点尝试超过 Hybrid RAG，而不仅仅是超过 Vector RAG。

## Roadmap / 阶段路线

| Phase | Method | Goal | Status |
| --- | --- | --- | --- |
| Phase 1 | Vector RAG baseline | 建立普通 dense retrieval 对照组 | 已完成 |
| Phase 2 | BM25 + Hybrid RAG baseline | 比较 lexical、dense、hybrid retrieval | 已完成 |
| Phase 2.5 | Fair baselines + upgraded metrics | 统一 baseline 配置并补齐多跳证据指标 | 已完成基础版 |
| Phase 3 | GraphRAG-style RAG | 构建轻量实体关系图，测试图检索是否提升多跳证据召回 | 下一步 |
| Phase 4 | Evidence-aware Improved LightRAG | 图邻居扩展 + 证据覆盖感知重排序 | 待实现 |
| Phase 4.5 | LightRAG controlled integration | 将外部 LightRAG 纳入统一评估，作为可选对照组 | 可选 |
| Phase 5 | Ablation experiments | 验证图扩展、coverage reranking 等模块贡献 | 待实现 |
| Phase 6 | Scale-up + case study | 扩大样本规模并分析成功/失败案例 | 待实现 |
| Phase 7 | Final report | 整理报告、图表和简历材料 | 待实现 |

更细的工程任务拆分见 [docs/implementation_plan.md](docs/implementation_plan.md)。Phase 2.5 细节见 [docs/phase2_5_fair_baselines.md](docs/phase2_5_fair_baselines.md)。

## Project Structure / 项目结构

```text
README.md
configs/
  phase1_vector_rag.yaml
  phase1_vector_rag_sentence_transformer.yaml
  phase2_bm25_rag.yaml
  phase2_hybrid_rag.yaml
  phase2_hybrid_rag_sentence_transformer.yaml
  phase2_vector_rag_fair.yaml
  phase2_bm25_rag_fair.yaml
  phase2_hybrid_rag_fair.yaml
docs/
  implementation_plan.md
  phase1_details.md
  phase2_details.md
  phase2_5_fair_baselines.md
  project_brief.md
  resume_notes.md
experiments/
  run_phase1_vector_rag.ps1
  run_phase2_hybrid_rag.ps1
  run_fair_baselines.ps1
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

运行 Phase 2.5 fair baselines：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_fair_baselines.ps1 -Python .\.venv\Scripts\python.exe
```

运行单个 fair baseline：

```powershell
.\.venv\Scripts\python.exe -m src.run_experiment --config configs\phase2_hybrid_rag_fair.yaml
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Resume Description / 简历表述

中文：

> 构建面向 HotpotQA 多跳问答的可复现 RAG 对比实验框架，统一数据标准化、文本切块、Vector RAG、BM25 关键词检索、Hybrid score fusion 和多维检索指标评估。在统一 chunking、top-k 和数据配置下完成 Vector / BM25 / Hybrid fair baseline 对比，并新增 Precision@k、NDCG@k、Full Evidence Recall@k 等多跳证据完整性指标。实验显示 Hybrid RAG 在 100 条 HotpotQA validation 样本上取得 Recall@5 0.790、Full Evidence Recall@5 0.590、NDCG@5 0.7582，为后续 GraphRAG-style 和 Improved LightRAG 提供强 baseline。

English:

> Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, including dataset normalization, chunking, Vector RAG, BM25 retrieval, hybrid score fusion and retrieval metrics. Established fair Vector / BM25 / Hybrid baselines under unified chunking, top-k and dataset settings, and added multi-hop evidence metrics such as Precision@k, NDCG@k and Full Evidence Recall@k. Hybrid RAG achieved Recall@5 of 0.790, Full Evidence Recall@5 of 0.590 and NDCG@5 of 0.7582 on 100 HotpotQA validation samples, providing a strong baseline for later GraphRAG-style and Improved LightRAG experiments.
