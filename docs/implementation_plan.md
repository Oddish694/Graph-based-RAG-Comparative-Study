# 实施计划 / Implementation Plan

## 1. 总体目标

本项目的最终目标是构建一个可复现的 RAG 对比实验框架，用于比较 Vector RAG、Hybrid RAG、GraphRAG-style RAG、LightRAG 以及改进版 LightRAG 在多跳问答任务上的表现。

这里的“比较”不是只看一个 demo 能不能回答问题，而是在同一套数据、同一套 query、同一套评估指标和同一套实验配置下，系统比较这些方法的：

- 检索效果：Recall@k、MRR、Hit Rate、NDCG 等
- 回答质量：Exact Match、F1、Faithfulness、Answer Relevance 等
- 系统效率：索引时间、检索延迟、端到端查询延迟、存储成本、token 使用量
- 工程复杂度：是否需要图构建、实体抽取、关系抽取、reranker 或 query rewrite

项目最终要回答的问题是：

> 在 HotpotQA / 2WikiMultiHopQA 这类多跳问答任务上，图增强 RAG 是否比普通向量 RAG 更能找全证据？如果有效，它的质量提升是否值得额外的延迟和工程成本？

## 2. 整体方法路线

项目按方法逐步扩展，而不是一开始就实现最复杂的 GraphRAG。

| 阶段 | 方法 | 主要目标 | 当前状态 |
| --- | --- | --- | --- |
| Phase 1 | Vector RAG baseline | 建立普通向量检索对照组 | 已完成 |
| Phase 2 | Hybrid RAG baseline | 比较 BM25、dense、hybrid 检索 | 待实现 |
| Phase 3 | GraphRAG-style RAG | 实现轻量实体关系图检索 | 待实现 |
| Phase 4 | LightRAG controlled integration | 将 LightRAG 纳入统一评估框架 | 待实现 |
| Phase 5 | Improved LightRAG / improved graph retrieval | 加入 reranker 或 query rewrite | 待实现 |
| Phase 6 | Ablation and reporting | 做消融实验和结果报告 | 待实现 |

这种顺序的原因是：先有简单 baseline，再加检索增强，再加图结构，最后分析提升来自哪里。否则直接跑复杂方法，很难判断效果提升到底来自图结构、embedding、top-k，还是其它实现细节。

## 3. 项目模块划分

当前与后续计划的目录结构如下：

```text
src/
├── datasets/
│   ├── schema.py
│   ├── hotpotqa_loader.py
│   └── twowiki_loader.py                  # 后续
├── chunking/
│   ├── fixed_chunker.py
│   ├── recursive_chunker.py
│   └── semantic_chunker.py                # 后续
├── indexing/
│   ├── vector_index.py
│   ├── bm25_index.py                      # Phase 2
│   └── graph_index.py                     # Phase 3
├── retrieval/
│   ├── naive_rag.py
│   ├── vector_rag.py
│   ├── hybrid_rag.py                      # Phase 2
│   ├── graph_rag_style.py                 # Phase 3
│   └── lightrag_runner.py                 # Phase 4
├── graph/
│   ├── entity_extractor.py                # Phase 3
│   ├── relation_extractor.py              # Phase 3
│   └── graph_builder.py                   # Phase 3
├── enhancement/
│   ├── reranker.py                        # Phase 5
│   ├── query_rewrite.py                   # Phase 5
│   └── hyde.py                            # 可选
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── answer_metrics.py                  # 后续
│   └── efficiency_metrics.py              # 后续
├── reporting/
│   ├── tables.py                          # Phase 6
│   └── plots.py                           # Phase 6
└── run_experiment.py
```

## 4. Phase 1：Vector RAG baseline，已完成

### 4.1 目标

建立普通向量检索 baseline。

Vector RAG 在本项目中指的是：

```text
question -> embedding -> vector search over chunks -> top-k evidence
```

它不使用图结构、不做实体关系建模、不加 reranker、不做 query rewrite。它的作用是作为后续所有增强方法的对照组。

### 4.2 输入

- 数据集：HotpotQA validation split
- 数据来源：Hugging Face `hotpotqa/hotpot_qa`
- 当前实验规模：100 条 validation 样本
- 样本字段：question、answer、contexts、supporting_facts

### 4.3 已实现模块

| 文件 | 作用 |
| --- | --- |
| `src/datasets/schema.py` | 定义统一 QA 样本结构 `QASample` |
| `src/datasets/hotpotqa_loader.py` | 加载 HotpotQA、标准化格式、保存 JSONL 缓存 |
| `src/chunking/fixed_chunker.py` | 将 context documents 切成固定大小 chunks |
| `src/indexing/vector_index.py` | 实现 hashing embedding、sentence-transformers embedding 和本地向量索引 |
| `src/retrieval/vector_rag.py` | 封装 Vector RAG top-k 检索接口 |
| `src/evaluation/retrieval_metrics.py` | 实现 Recall@k、MRR、Hit Rate |
| `src/run_experiment.py` | 串起数据加载、切块、索引、检索和评估流程 |
| `configs/phase1_vector_rag.yaml` | 离线 smoke test 配置 |
| `configs/phase1_vector_rag_sentence_transformer.yaml` | 真实 HotpotQA + sentence-transformers 配置 |
| `experiments/run_phase1_vector_rag.ps1` | PowerShell 运行入口 |

### 4.4 输出

Phase 1 会输出：

```text
results/phase1_vector_rag*/per_query_results.csv
results/phase1_vector_rag*/aggregate_metrics.json
```

`per_query_results.csv` 保存每个问题的检索结果和指标。

`aggregate_metrics.json` 保存整体平均指标。

### 4.5 当前结果

真实实验配置：

- Dataset：HotpotQA validation
- Sample size：100
- Embedding：`sentence-transformers/all-MiniLM-L6-v2`
- Retriever：Vector RAG
- Top-k：5

结果：

| Metric | Value |
| --- | --- |
| Recall@1 | 0.375 |
| Recall@3 | 0.65 |
| Recall@5 | 0.74 |
| MRR@1 | 0.75 |
| MRR@3 | 0.82 |
| MRR@5 | 0.8275 |
| Hit Rate@1 | 0.75 |
| Hit Rate@3 | 0.92 |
| Hit Rate@5 | 0.95 |

解释：Vector RAG 在 top-5 中通常能命中至少一个相关证据，但还不能完整找齐所有多跳 supporting facts。这个结果为后续图增强方法提供了明确对照。

## 5. Phase 2：Hybrid RAG baseline

### 5.1 目标

在 Vector RAG 基础上加入 BM25 lexical retrieval，并通过 score fusion 融合 BM25 和 dense retrieval 结果。

Hybrid RAG 的检索流程是：

```text
question
├── BM25 lexical retrieval
└── dense vector retrieval
        ↓
score fusion / rank fusion
        ↓
top-k evidence
```

### 5.2 为什么要做

Vector RAG 擅长语义相似，但有时会漏掉精确关键词。BM25 擅长 exact match，但不理解语义改写。Hybrid RAG 可以测试“关键词匹配 + 语义匹配”是否比单独 dense retrieval 更稳。

### 5.3 计划实现文件

| 文件 | 作用 |
| --- | --- |
| `src/indexing/bm25_index.py` | 构建 BM25 索引，返回 lexical top-k |
| `src/retrieval/hybrid_rag.py` | 融合 BM25 和 dense retrieval 结果 |
| `src/evaluation/retrieval_metrics.py` | 复用 Phase 1 指标 |
| `configs/phase2_hybrid_rag.yaml` | Hybrid RAG 配置 |
| `experiments/run_phase2_hybrid_rag.ps1` | 一键运行脚本 |
| `tests/test_bm25_index.py` | BM25 行为测试 |
| `tests/test_hybrid_retrieval.py` | fusion 和 top-k 输出测试 |

### 5.4 核心实现点

- 对每个 chunk 建 BM25 索引
- dense retrieval 继续复用 Phase 1 的 `VectorIndex`
- 将 BM25 分数和 dense 分数归一化
- 支持两种融合方式：weighted score fusion 和 reciprocal rank fusion
- 返回统一格式：doc_id、chunk_id、text、score、retriever_source

### 5.5 评估方式

在同一 HotpotQA subset 上比较：

| 方法 | 指标 |
| --- | --- |
| Vector RAG | Recall@k、MRR、Hit Rate、latency |
| BM25 RAG | Recall@k、MRR、Hit Rate、latency |
| Hybrid RAG | Recall@k、MRR、Hit Rate、latency |

重点分析：Hybrid 是否提高 Recall@k，以及增加了多少检索延迟。

## 6. Phase 3：GraphRAG-style RAG

### 6.1 目标

实现一个轻量 GraphRAG-style baseline。它不追求完整复现微软 GraphRAG 的全部工程，而是实现图增强检索的核心思想：从文本中抽取实体和关系，构建图结构，在检索时利用实体邻居扩展多跳证据。

基本流程：

```text
contexts / chunks
        ↓
entity extraction
        ↓
relation extraction
        ↓
graph construction
        ↓
query entity linking
        ↓
vector retrieval + graph neighbor expansion
        ↓
top-k evidence
```

### 6.2 为什么要做

HotpotQA 的很多问题需要跨文档推理。普通 Vector RAG 只看 question 和 chunk 的语义相似度，可能找到第一跳证据，但漏掉第二跳证据。GraphRAG-style 方法希望通过实体关系把多个相关文档连接起来，提高 supporting facts 的覆盖率。

### 6.3 计划实现文件

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

### 6.4 核心实现点

第一版可以采用轻量策略：

- 实体抽取先用 spaCy / regex / simple noun phrase heuristic，避免过早依赖 LLM 抽取。
- 关系抽取先用共现关系，例如同一 chunk 或同一 document 中出现的实体建立边。
- 图节点包括 entity、document、chunk。
- 图边包括 entity-entity、entity-document、document-chunk。
- 查询时先抽取 query entities，再找到相关实体邻居和对应 chunks。
- 最终分数可以融合 vector score 和 graph proximity score。

### 6.5 评估方式

和 Vector RAG / Hybrid RAG 使用同一批 HotpotQA 样本、同一 top-k、同一指标。

重点看：

- Recall@k 是否提升
- 第二跳证据是否更容易被找回
- graph construction time 增加多少
- retrieval latency 增加多少
- 错误实体抽取是否会引入噪声

## 7. Phase 4：LightRAG controlled integration

### 7.1 目标

把 LightRAG 纳入项目统一评估框架。

这里的重点不是简单跑 LightRAG demo，而是让 LightRAG 使用同一 HotpotQA subset、同一 query set、同一 top-k 设置和同一评估指标，这样才能和 Vector RAG、Hybrid RAG、GraphRAG-style RAG 公平比较。

### 7.2 为什么要做

LightRAG 是一个轻量图增强 RAG 方法。它试图用较低的图构建和查询成本获得图增强检索能力。这个项目需要验证：LightRAG 在多跳问答任务中是否比普通 dense retrieval 更能找全证据，同时延迟是否可接受。

### 7.3 计划实现文件

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/lightrag_runner.py` | LightRAG 输入适配、索引构建、查询和结果捕获 |
| `configs/phase4_lightrag.yaml` | LightRAG 实验配置 |
| `experiments/run_phase4_lightrag.ps1` | 运行脚本 |
| `tests/test_lightrag_adapter.py` | 数据适配测试 |

### 7.4 核心实现点

- 将 `QASample.contexts` 转成 LightRAG 需要的 document 输入格式
- 固定 LightRAG query mode、top-k 和 embedding/LLM 配置
- 捕获 LightRAG 返回的 retrieved contexts 或 evidence references
- 将 LightRAG 结果转换成统一 retrieval result schema
- 计算 Recall@k、MRR、Hit Rate
- 记录 index time、retrieval latency、storage size

### 7.5 评估方式

对比表至少包含：

| 方法 | Recall@5 | MRR@5 | Hit Rate@5 | Index Time | Retrieval Latency |
| --- | --- | --- | --- | --- | --- |
| Vector RAG | 已有结果 | 已有结果 | 已有结果 | 已有结果 | 已有结果 |
| Hybrid RAG | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| GraphRAG-style RAG | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| LightRAG | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

## 8. Phase 5：Improved LightRAG / improved graph retrieval

### 8.1 目标

在 LightRAG 或 GraphRAG-style 检索结果上加入轻量增强策略，观察是否能进一步提升检索质量或回答质量。

优先考虑两个方向：

- reranker-based context selection
- query rewrite enhanced retrieval

### 8.2 Reranker 改进

Reranker 的流程：

```text
query -> retrieve top-20 candidates -> reranker re-score -> choose top-5
```

计划实现文件：

| 文件 | 作用 |
| --- | --- |
| `src/enhancement/reranker.py` | reranker 接口和实现 |
| `configs/phase5_reranker.yaml` | reranker 实验配置 |
| `tests/test_reranker.py` | reranker 输出格式测试 |

评估重点：

- Recall@5 / MRR@5 是否提升
- context precision 是否提升
- reranking latency 增加多少

### 8.3 Query rewrite 改进

Query rewrite 的流程：

```text
original question -> rewritten retrieval query -> retrieval -> evaluation
```

计划实现文件：

| 文件 | 作用 |
| --- | --- |
| `src/enhancement/query_rewrite.py` | query rewrite 接口和 prompt |
| `configs/phase5_query_rewrite.yaml` | query rewrite 实验配置 |
| `tests/test_query_rewrite.py` | 改写结果约束测试 |

评估重点：

- 改写是否提高 Recall@k
- 改写是否引入原问题没有的假设
- 改写带来的额外 LLM 调用成本
- 成功案例和失败案例

## 9. Phase 6：消融实验与报告

### 9.1 目标

系统分析不同组件对结果的影响，并把实验结果整理成表格、图和报告。

### 9.2 消融维度

| 维度 | 候选值 |
| --- | --- |
| Chunk size | 256, 512, 1024 |
| Top-k | 3, 5, 10, 20 |
| Embedding model | all-MiniLM-L6-v2, bge-small, bge-base, bge-m3 |
| Retriever | BM25, dense, hybrid, graph-aware |
| Reranker | none, cross-encoder, bge-reranker |
| Query enhancement | none, query rewrite, HyDE |

### 9.3 计划实现文件

| 文件 | 作用 |
| --- | --- |
| `configs/ablation_chunk_topk.yaml` | chunk size 和 top-k 消融配置 |
| `configs/ablation_embedding_retriever.yaml` | embedding 和 retriever 消融配置 |
| `src/reporting/tables.py` | 汇总 CSV/JSON 结果成对比表 |
| `src/reporting/plots.py` | 生成指标图和延迟图 |
| `docs/final_report.md` | 最终技术报告 |

### 9.4 最终报告结构

最终报告计划包含：

1. 项目背景和研究问题
2. 数据集和实验设置
3. 方法说明：Vector、Hybrid、GraphRAG-style、LightRAG、Improved LightRAG
4. 检索指标对比
5. 回答质量对比
6. 延迟和成本分析
7. 消融实验
8. 成功案例和失败案例
9. 结论：图增强 RAG 是否值得使用

## 10. 当前优先级

推荐后续实现顺序：

1. Phase 2 Hybrid RAG：先补 BM25 和 hybrid baseline，成本低，能快速形成第二个对照组。
2. Phase 3 GraphRAG-style RAG：实现轻量图检索，直接对应项目主题。
3. Phase 4 LightRAG：把外部 LightRAG 纳入统一评估。
4. Phase 5 Reranker / query rewrite：做轻量改进。
5. Phase 6 Reporting：整理结果图表和最终报告。

这样项目会从“只有 Vector RAG baseline”逐步变成完整的 RAG 方法对比研究。
