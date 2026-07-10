# 实施计划 / Implementation Plan

## 1. 总体目标

本项目的最终目标是构建一个可复现的 RAG 对比实验框架，用于比较 Vector RAG、BM25、Hybrid RAG、GraphRAG-style RAG、LightRAG 以及改进版 LightRAG 在多跳问答任务上的表现。

这里的“比较”不是只看一个 demo 能不能回答问题，而是在同一套数据、同一套 query、同一套评估指标和同一套实验配置下，系统比较这些方法的：

- 检索效果：Recall@k、MRR@k、Hit Rate@k、NDCG 等。
- 回答质量：Exact Match、F1、Faithfulness、Answer Relevance 等。
- 系统效率：索引时间、检索延迟、端到端查询延迟、存储成本、token 使用量。
- 工程复杂度：是否需要图构建、实体抽取、关系抽取、reranker 或 query rewrite。

项目最终要回答的问题是：

> 在 HotpotQA / 2WikiMultiHopQA 这类多跳问答任务上，图增强 RAG 是否比普通向量 RAG 更能找全证据？如果有效，它的质量提升是否值得额外的延迟和工程成本？

## 2. 阶段路线

| 阶段 | 方法 | 主要目标 | 当前状态 |
| --- | --- | --- | --- |
| Phase 1 | Vector RAG baseline | 建立普通 dense retrieval 对照组 | 已完成 |
| Phase 2 | BM25 + Hybrid RAG baseline | 比较 lexical、dense、hybrid retrieval | 已完成基础版 |
| Phase 3 | GraphRAG-style RAG | 实现轻量实体关系图检索 | 待实现 |
| Phase 4 | LightRAG controlled integration | 将 LightRAG 纳入统一评估框架 | 待实现 |
| Phase 5 | Improved LightRAG / improved graph retrieval | 加入 reranker、query rewrite 或 HyDE | 待实现 |
| Phase 6 | Ablation and reporting | 做消融实验和结果报告 | 待实现 |

这个顺序的原因是：先有最基础的 dense baseline，再加入关键词检索和混合检索，然后再进入图结构方法。这样后续如果 GraphRAG / LightRAG 有提升，才能判断提升到底来自图结构，还是只是来自 BM25、embedding、top-k、chunking 等基础组件。

## 3. 模块结构

```text
src/
  datasets/
    schema.py
    hotpotqa_loader.py
    twowiki_loader.py                  # 后续
  chunking/
    fixed_chunker.py
    recursive_chunker.py
    semantic_chunker.py                # 后续
  indexing/
    vector_index.py
    bm25_index.py                      # Phase 2 已完成
    graph_index.py                     # Phase 3
  retrieval/
    vector_rag.py
    bm25_rag.py                        # Phase 2 已完成
    hybrid_rag.py                      # Phase 2 已完成
    graph_rag_style.py                 # Phase 3
    lightrag_runner.py                 # Phase 4
  graph/
    entity_extractor.py                # Phase 3
    relation_extractor.py              # Phase 3
    graph_builder.py                   # Phase 3
  enhancement/
    reranker.py                        # Phase 5
    query_rewrite.py                   # Phase 5
    hyde.py                            # 可选
  evaluation/
    retrieval_metrics.py
    answer_metrics.py                  # 后续
    efficiency_metrics.py              # 后续
  reporting/
    tables.py                          # Phase 6
    plots.py                           # Phase 6
  run_experiment.py
```

## 4. Phase 1: Vector RAG baseline, 已完成

### 4.1 目标

建立普通向量检索 baseline：

```text
question -> embedding -> vector search over chunks -> top-k evidence
```

Vector RAG 在本项目中不使用图结构、不做实体关系建模、不加 reranker、不做 query rewrite。它的作用是作为后续所有增强方法的对照组。

### 4.2 已实现文件

| 文件 | 作用 |
| --- | --- |
| `src/datasets/schema.py` | 定义统一 QA 样本结构 `QASample` |
| `src/datasets/hotpotqa_loader.py` | 加载 HotpotQA、标准化格式、保存 JSONL 缓存 |
| `src/chunking/fixed_chunker.py` | 将 context documents 切成固定大小 chunks |
| `src/indexing/vector_index.py` | 实现 hashing embedding、sentence-transformers adapter 和本地向量索引 |
| `src/retrieval/vector_rag.py` | 封装 Vector RAG top-k 检索接口 |
| `src/evaluation/retrieval_metrics.py` | 实现 Recall@k、MRR@k、Hit Rate@k |
| `src/run_experiment.py` | 串起数据加载、切块、索引、检索和评估流程 |
| `configs/phase1_vector_rag.yaml` | 离线 smoke test 配置 |
| `configs/phase1_vector_rag_sentence_transformer.yaml` | 真实 HotpotQA + sentence-transformers 配置 |
| `experiments/run_phase1_vector_rag.ps1` | PowerShell 运行入口 |

### 4.3 当前结果

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.375 |
| Recall@3 | 0.650 |
| Recall@5 | 0.740 |
| MRR@1 | 0.750 |
| MRR@3 | 0.820 |
| MRR@5 | 0.8275 |
| Hit Rate@1 | 0.750 |
| Hit Rate@3 | 0.920 |
| Hit Rate@5 | 0.950 |

解释：Vector RAG 在 top-5 中通常能命中至少一个相关证据，但还不能完整找齐所有多跳 supporting facts。这为后续图增强方法提供了明确对照。

## 5. Phase 2: BM25 + Hybrid RAG baseline, 已完成基础版

### 5.1 目标

在 Phase 1 的 Vector RAG 基础上加入 BM25 lexical retrieval，并通过 score fusion 融合 BM25 和 dense retrieval 结果。

Hybrid RAG 的检索流程是：

```text
question
  -> BM25 lexical retrieval
  -> dense vector retrieval
  -> weighted score fusion / reciprocal rank fusion
  -> top-k evidence
```

### 5.2 为什么要做

Vector RAG 擅长语义相似，但有时会漏掉精确关键词、实体名、年份、专有名词。BM25 擅长 exact match，但不理解语义改写。Hybrid RAG 测试的是“关键词匹配 + 语义匹配”是否能比单独 dense retrieval 更稳。

### 5.3 已实现文件

| 文件 | 作用 |
| --- | --- |
| `src/indexing/bm25_index.py` | 构建 BM25 索引，返回 lexical top-k |
| `src/retrieval/bm25_rag.py` | 封装 BM25-only retriever |
| `src/retrieval/hybrid_rag.py` | 融合 BM25 和 dense retrieval 结果 |
| `src/run_experiment.py` | 支持 `vector`、`bm25`、`hybrid` 三种检索方法 |
| `configs/phase2_bm25_rag.yaml` | BM25-only 真实实验配置 |
| `configs/phase2_hybrid_rag.yaml` | Hybrid smoke test 配置 |
| `configs/phase2_hybrid_rag_sentence_transformer.yaml` | Hybrid 真实实验配置 |
| `experiments/run_phase2_hybrid_rag.ps1` | Hybrid 一键运行脚本 |
| `tests/test_bm25_index.py` | BM25 行为测试 |
| `tests/test_hybrid_retrieval.py` | fusion 和 top-k 输出测试 |
| `tests/test_phase2_runner.py` | Phase 2 runner 输出测试 |

### 5.4 核心实现点

- 对每个 chunk 建 BM25 索引，使用 token frequency、document frequency、document length 和 average document length 计算 Okapi BM25 分数。
- dense retrieval 继续复用 Phase 1 的 `VectorIndex`。
- Hybrid 检索时先取两路候选集，例如 top-20 BM25 和 top-20 dense。
- weighted fusion 会分别归一化 BM25 分数和 dense 分数，再按 `bm25_weight` 与 `dense_weight` 加权求和。
- RRF fusion 使用 reciprocal rank fusion，根据两个排序列表中的名次合并结果，不依赖原始分数尺度。
- 检索结果统一返回 `doc_id`、`chunk_id`、`text`、`score`、`retriever_source`，便于统一评估。

### 5.5 当前结果

同一批 100 条 HotpotQA validation 样本、同一 chunk 配置、同一 top-k 下：

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit Rate@5 | Avg Retrieval Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.375 | 0.650 | 0.740 | 0.8275 | 0.950 | previous Phase 1 run |
| BM25 | 0.355 | 0.595 | 0.710 | 0.8200 | 0.990 | 0.0046s |
| Hybrid RAG | 0.410 | 0.690 | 0.785 | 0.8908 | 0.980 | 0.0158s |

初步解释：Hybrid RAG 的 Recall@5 和 MRR@5 高于单独 Vector RAG 与单独 BM25，说明 lexical matching 和 semantic matching 具有互补性。BM25 的 Hit Rate@5 很高，说明关键词检索经常能碰到至少一个 gold document，但 Recall@5 较低，说明它不一定能找全多跳证据。

## 6. Phase 3: GraphRAG-style RAG

### 6.1 目标

实现一个轻量 GraphRAG-style baseline。它不追求完整复现微软 GraphRAG 的全部工程流程，而是实现图增强检索的核心思想：从文本中抽取实体和关系，构建图结构，在检索时利用实体邻居扩展多跳证据。

基本流程：

```text
contexts / chunks
  -> entity extraction
  -> relation extraction
  -> graph construction
  -> query entity linking
  -> vector retrieval + graph neighbor expansion
  -> top-k evidence
```

### 6.2 计划实现文件

| 文件 | 作用 |
| --- | --- |
| `src/graph/entity_extractor.py` | 从 chunk 中抽取实体 |
| `src/graph/relation_extractor.py` | 抽取轻量关系或共现关系 |
| `src/graph/graph_builder.py` | 构建 entity-document-chunk 图 |
| `src/indexing/graph_index.py` | 存储图结构和实体到 chunk 的映射 |
| `src/retrieval/graph_rag_style.py` | 实现 graph-aware retrieval |
| `configs/phase3_graph_rag.yaml` | GraphRAG-style 配置 |
| `experiments/run_phase3_graph_rag.ps1` | 运行脚本 |
| `tests/test_graph_index.py` | 图构建测试 |
| `tests/test_graph_retrieval.py` | 图检索测试 |

## 7. Phase 4: LightRAG controlled integration

目标是将 LightRAG 纳入统一评估框架，而不是只运行 LightRAG demo。LightRAG 必须使用同一 HotpotQA subset、同一 query set、同一 top-k 和同一评估指标，这样才能和 Vector RAG、BM25、Hybrid RAG、GraphRAG-style RAG 公平比较。

计划实现：

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/lightrag_runner.py` | LightRAG 输入适配、索引构建、查询和结果捕获 |
| `configs/phase4_lightrag.yaml` | LightRAG 实验配置 |
| `experiments/run_phase4_lightrag.ps1` | 运行脚本 |
| `tests/test_lightrag_adapter.py` | 数据适配测试 |

## 8. Phase 5: Improved LightRAG / improved graph retrieval

在 LightRAG 或 GraphRAG-style 检索结果上加入轻量增强策略，观察是否能进一步提升检索质量或回答质量。

优先方向：

- reranker-based context selection。
- query rewrite enhanced retrieval。
- HyDE-style hypothetical document retrieval。

## 9. Phase 6: 消融实验与报告

系统分析不同组件对结果的影响，并把实验结果整理成表格、图和技术报告。

消融维度包括：

| 维度 | 候选值 |
| --- | --- |
| Chunk size | 64, 128, 256, 512 |
| Top-k | 3, 5, 10, 20 |
| Embedding model | all-MiniLM-L6-v2, bge-small, bge-base, bge-m3 |
| Retriever | BM25, dense, hybrid, graph-aware |
| Reranker | none, cross-encoder, bge-reranker |
| Query enhancement | none, query rewrite, HyDE |

最终报告计划包括：

1. 项目背景和研究问题。
2. 数据集和实验设置。
3. 方法说明：Vector、BM25、Hybrid、GraphRAG-style、LightRAG、Improved LightRAG。
4. 检索指标对比。
5. 回答质量对比。
6. 延迟和成本分析。
7. 消融实验。
8. 成功案例和失败案例。
9. 结论：图增强 RAG 是否值得使用。

## 10. 当前优先级

下一步建议：

1. Phase 3 GraphRAG-style RAG：实现轻量实体关系图，直接对应项目主题。
2. Phase 4 LightRAG：把外部 LightRAG 纳入统一评估。
3. Phase 5 Reranker / query rewrite：做轻量改进。
4. Phase 6 Reporting：整理结果图表和最终报告。
