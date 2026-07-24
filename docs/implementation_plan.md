# 实施计划 / Implementation Plan

## 1. 项目定位

本项目的目标是构建一个可复现的 RAG 对比实验框架，用于研究不同检索增强方法在多跳问答任务上的表现。

当前项目已经完成 Vector RAG、BM25 和 Hybrid RAG 的基础实验框架，具备数据加载、文本切块、检索器接口、指标评估、配置驱动实验、结果落盘和单元测试能力。后续工作要从“完成 baseline”推进到“验证图增强检索是否能提升多跳证据完整召回”。

项目最终要回答的问题是：

> 在 HotpotQA / 2WikiMultiHopQA 这类多跳问答任务上，图增强 RAG 是否比普通向量检索和 Hybrid 检索更能找全 supporting facts？如果有效，这种提升是否值得额外的图构建成本、检索延迟和工程复杂度？

本项目优先完成 retrieval-side research loop。也就是说，当前重点不是先做生成答案，而是先把“检索是否找全证据”这件事研究清楚。

## 2. 当前已完成内容

### 2.1 Phase 1: Vector RAG baseline，已完成

已实现普通 dense retrieval baseline：

```text
question -> embedding -> vector search over chunks -> top-k evidence
```

已实现文件：

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

Phase 1 的作用是建立最基础的 dense retrieval 对照组。它不使用图结构、不使用 BM25、不做 reranker、不做 query rewrite。

### 2.2 Phase 2: BM25 + Hybrid RAG baseline，已完成基础版

已实现 lexical retrieval 和 hybrid retrieval：

```text
question
  -> BM25 lexical retrieval
  -> dense vector retrieval
  -> weighted score fusion / reciprocal rank fusion
  -> top-k evidence
```

已实现文件：

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

当前 100 条 HotpotQA validation 初步结果：

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit Rate@5 | Avg Retrieval Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.375 | 0.650 | 0.740 | 0.8275 | 0.950 | previous Phase 1 run |
| BM25 | 0.355 | 0.595 | 0.710 | 0.8200 | 0.990 | 0.0046s |
| Hybrid RAG | 0.410 | 0.690 | 0.785 | 0.8908 | 0.980 | 0.0158s |

注意：这张表目前只能作为初步结果。历史配置中 Vector RAG 使用过 `chunk_size=128, overlap=20`，BM25 / Hybrid 使用 `chunk_size=64, overlap=8`。后续主结果必须用完全统一的 fair configs 重新跑。

## 3. 后续总路线

后续路线调整为：

```text
Phase 2.5: Fair baseline + metrics upgrade
  -> Phase 3: GraphRAG-style baseline
  -> Phase 4: Evidence-aware Improved LightRAG
  -> Phase 4.5: LightRAG controlled integration
  -> Phase 5: Ablation experiments
  -> Phase 6: Scale-up experiments + case study
  -> Phase 7: Final report + resume packaging
```

阶段优先级：

| 阶段 | 方法 / 工作 | 目标 | 优先级 |
| --- | --- | --- | --- |
| Phase 2.5 | Fair baseline + metrics upgrade | 统一 Vector / BM25 / Hybrid 配置，补齐更严格检索指标 | 最高 |
| Phase 3 | GraphRAG-style baseline | 实现轻量实体关系图检索，进入项目主题 | 已完成可复现实验版本 |
| Phase 4 | Evidence-aware Improved LightRAG | 实现图邻居扩展和证据覆盖感知重排序 | 已完成可复现实验版本 |
| Phase 4.5 | LightRAG controlled integration | 将 LightRAG 对照组纳入统一评估接口 | 已完成可控接入层 |
| Phase 5 | Ablation experiments | 验证每个改进模块是否真的有效 | 下一步 |
| Phase 6 | Scale-up + case study | 扩大样本规模，并分析成功和失败案例 | 中等 |
| Phase 7 | Reporting | 整理报告、README、简历材料和结果表格 | 最高 |

## 4. Phase 2.5: Fair Baseline + Metrics Upgrade

### 4.1 目标

把当前 Vector RAG、BM25 和 Hybrid RAG 变成严格公平、可引用的实验基线。

必须统一：

- 同一数据集和 split。
- 同一 `sample_size` 和 `seed`。
- 同一 chunking strategy、chunk size 和 overlap。
- 同一 top-k 设置。
- 同一 embedding model。
- 同一评估指标。
- 同一结果输出格式。

这一阶段完成后，后续 GraphRAG / LightRAG 的提升才不会被质疑来自配置差异。

### 4.2 需要新增或修改的文件

| 文件 | 作用 |
| --- | --- |
| `src/evaluation/retrieval_metrics.py` | 补充 Precision@k、NDCG@k、Evidence Hit Count@k、Full Evidence Recall@k |
| `configs/phase2_vector_rag_fair.yaml` | Vector RAG fair baseline 配置 |
| `configs/phase2_bm25_rag_fair.yaml` | BM25 fair baseline 配置 |
| `configs/phase2_hybrid_rag_fair.yaml` | Hybrid RAG fair baseline 配置 |
| `experiments/run_fair_baselines.ps1` | 一次性运行三种 fair baseline |
| `tests/test_retrieval_metrics.py` | 扩展指标测试 |
| `docs/phase2_5_fair_baselines.md` | 记录公平 baseline 设置与结果 |

### 4.3 推荐 fair baseline 配置

主实验建议先用：

```text
Dataset: HotpotQA validation
Sample size: 100 for development, 500 for main experiment
Seed: 42
Chunk size: 64
Overlap: 8
Embedding: sentence-transformers/all-MiniLM-L6-v2
Top-k: 1, 3, 5, 10
```

`k=10` 需要加入，因为图扩展和 reranking 往往需要看更宽的候选范围。

### 4.4 指标升级

当前已有：

- Recall@k
- MRR@k
- Hit Rate@k

需要新增：

| 指标 | 含义 | 为什么需要 |
| --- | --- | --- |
| Precision@k | top-k 中相关证据比例 | 衡量噪声，防止图扩展只提高召回但引入大量无关内容 |
| NDCG@k | 相关证据是否整体排得靠前 | 比 MRR 更能反映多个 evidence 的排序质量 |
| Evidence Hit Count@k | top-k 中命中的 gold evidence 数量 | 直观看到找回了几个 supporting docs |
| Full Evidence Recall@k / Complete Evidence Hit@k | 一个问题所需全部 supporting evidence 是否都出现在 top-k 中 | 多跳问答最关键，衡量证据链是否完整 |
| Retrieved Context Tokens | 返回上下文长度 | 估算后续 generation token 成本 |

### 4.5 阶段产出

- 100 samples fair baseline 表格。
- 500 samples fair baseline 表格。
- 更完整的 retrieval metrics。
- 一份可在报告和简历中引用的 baseline 结论。

### 4.6 已完成内容与 100 samples fair baseline 结果

Phase 2.5 基础版已经完成：

- 已新增 `precision@k`、`ndcg@k`、`evidence_hit_count@k`、`full_evidence_recall@k`、`retrieved_context_tokens@k`。
- 已新增 `phase2_vector_rag_fair.yaml`、`phase2_bm25_rag_fair.yaml`、`phase2_hybrid_rag_fair.yaml`。
- 已新增 `experiments/run_fair_baselines.ps1`。
- 已在 100 条 HotpotQA validation 样本上完成统一配置实验。

统一设置：

```text
Dataset: HotpotQA validation
Sample size: 100
Seed: 42
Chunk size: 64
Overlap: 8
Embedding: sentence-transformers/all-MiniLM-L6-v2
Top-k: 10
Metric k values: 1, 3, 5, 10
```

主结果：

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.725 | 0.290 | 0.6901 | 0.8325 | 0.510 | 0.940 | 0.0108s |
| BM25 | 0.710 | 0.284 | 0.6668 | 0.8200 | 0.430 | 0.990 | 0.0053s |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |

结论：Hybrid RAG 在统一配置下是当前最强 baseline。后续 GraphRAG-style 和 Improved LightRAG 应重点比较 `Full Evidence Recall@5`、`NDCG@5`、`Precision@5` 和检索延迟，而不是只看 Recall@5。

## 5. Phase 3: GraphRAG-style Baseline

### 5.1 目标

实现一个轻量、可解释、可控的 GraphRAG-style baseline。它不追求完整复现微软 GraphRAG，而是实现图增强检索的核心思想：利用实体关系帮助多跳证据发现。

基本流程：

```text
contexts / chunks
  -> entity extraction
  -> entity-chunk graph construction
  -> query entity linking
  -> seed retrieval
  -> graph neighbor expansion
  -> score fusion
  -> top-k evidence
```

### 5.2 设计原则

第一版 GraphRAG-style baseline 应该保持轻量：

- 实体抽取优先用 regex、simple noun phrase heuristic 或 spaCy，避免过早依赖 LLM。
- 图结构先做 entity-chunk graph 和 entity co-occurrence graph。
- 查询时可以从 query entities 或 seed chunks 出发扩展邻居。
- 最终排序可以融合 seed retriever score 和 graph proximity score。

### 5.3 计划文件

| 文件 | 作用 |
| --- | --- |
| `src/graph/entity_extractor.py` | 从 chunk 和 query 中抽取实体 |
| `src/graph/graph_builder.py` | 构建 entity-chunk 和 entity-entity 共现图 |
| `src/indexing/graph_index.py` | 存储实体、chunk、邻接关系和实体到 chunk 的映射 |
| `src/retrieval/graph_rag_style.py` | 实现 graph-aware retrieval |
| `configs/phase3_graph_rag_style.yaml` | GraphRAG-style 配置 |
| `experiments/run_phase3_graph_rag_style.ps1` | 运行脚本 |
| `tests/test_entity_extractor.py` | 实体抽取测试 |
| `tests/test_graph_index.py` | 图索引测试 |
| `tests/test_graph_retrieval.py` | 图检索测试 |
| `docs/phase3_graph_rag_style.md` | 方法说明和实验结果 |

### 5.4 初版 scoring 方案

可以从简单加权开始：

```text
final_score = seed_score + graph_weight * graph_proximity_score
```

其中：

- `seed_score` 来自 Vector / BM25 / Hybrid。
- `graph_proximity_score` 来自 query entity、seed chunk 与候选 chunk 在图上的距离或实体重合度。
- `graph_weight` 作为配置项调节。

### 5.5 阶段产出

- GraphRAG-style baseline 实现。
- GraphRAG-style vs Hybrid RAG 对比表。
- 图构建时间、图节点数、图边数、检索延迟。
- 初步 case study：图扩展找到了哪些 Hybrid 漏掉的证据。

### 5.6 已完成内容与 100 samples 结果

Phase 3 可复现实验版本已经完成：

- 已实现 `SimpleEntityExtractor`，用轻量规则抽取大写专有名词和多词实体。
- 已实现 `GraphIndex`，存储 entity-to-chunk、chunk-to-entity 和 entity co-occurrence graph。
- 已实现 `GraphRAGStyleRetriever`，使用 Hybrid seed retrieval + graph neighbor expansion + score fusion。
- 已新增 `configs/phase3_graph_rag_style.yaml` 和 `experiments/run_phase3_graph_rag_style.ps1`。
- 已新增实体抽取、图索引、图检索和 Phase 3 runner 测试。
- 已在 100 条 HotpotQA validation fair baseline 设置下完成实验。

与 Hybrid RAG fair baseline 对比：

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |
| GraphRAG-style | 0.805 | 0.322 | 0.7718 | 0.9095 | 0.620 | 0.990 | 0.0410s |

图统计：

| Metric | Value |
| --- | ---: |
| Graph entities | 8238 |
| Graph edges | 66592 |
| Index / graph construction time | 22.94s |
| Avg retrieval latency | 0.0410s |

结论：轻量 GraphRAG-style 在 top-5 证据召回、完整证据链召回和排序质量上小幅超过 Hybrid RAG，但延迟和构建成本更高。这说明图扩展有价值，也说明后续 Phase 4 需要做 Evidence-aware Graph Expansion 和 Coverage-aware Reranking 来进一步控制噪声和成本。

## 6. Phase 4: Evidence-aware Improved LightRAG

### 6.1 目标

在 GraphRAG-style baseline 基础上实现项目的核心改进方法：Evidence-aware Improved LightRAG。

这个阶段的重点不是简单接入官方 LightRAG demo，而是提出一个面向多跳证据完整召回的改进版图检索流程。

推荐主流程：

```text
query
  -> Hybrid seed retrieval
  -> query entity linking
  -> graph neighbor expansion
  -> collect candidate chunks
  -> evidence-aware scoring
  -> coverage-aware reranking
  -> final top-k evidence
```

### 6.2 改进一：Evidence-aware Graph Expansion

目标：让图扩展优先补全多跳证据链，而不是机械扩展所有邻居。

可调参数：

- `expansion_depth`: 1 or 2。
- `max_neighbors_per_entity`。
- `entity_overlap_weight`。
- `graph_distance_weight`。
- `seed_retriever`: Vector / BM25 / Hybrid。

可解释 scoring：

```text
graph_expansion_score =
  seed_relevance
  + entity_overlap_score
  - graph_distance_penalty
```

### 6.3 改进二：Coverage-aware Reranking

多跳问答需要 top-k 整体覆盖多个 supporting docs，而不是返回多个相似 chunk。Coverage-aware reranking 用来鼓励结果覆盖不同文档、实体或证据来源。

推荐 scoring：

```text
final_score =
  relevance_score
  + graph_proximity_score
  + entity_overlap_score
  + coverage_diversity_score
```

其中：

- `relevance_score` 衡量 chunk 与 query 的语义或关键词相关性。
- `graph_proximity_score` 衡量候选 chunk 与 query entities / seed chunks 的图距离。
- `entity_overlap_score` 衡量候选 chunk 与 query / seed evidence 的实体重合。
- `coverage_diversity_score` 鼓励 top-k 覆盖不同文档或不同实体簇。

### 6.4 计划文件

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/improved_lightrag.py` | Improved LightRAG 主检索器 |
| `src/retrieval/coverage_reranker.py` | 覆盖感知重排序模块 |
| `configs/phase4_improved_lightrag.yaml` | 改进方法配置 |
| `experiments/run_phase4_improved_lightrag.ps1` | 运行脚本 |
| `tests/test_coverage_reranker.py` | 覆盖重排序测试 |
| `tests/test_improved_lightrag.py` | Improved LightRAG 检索测试 |
| `docs/phase4_improved_lightrag.md` | 方法说明、结果和案例 |

### 6.5 阶段产出

- Evidence-aware Improved LightRAG 实现。
- Improved LightRAG vs GraphRAG-style vs Hybrid RAG 主结果表。
- Full Evidence Recall@k 和 Precision@k 的重点分析。
- 图扩展带来的延迟和噪声分析。

### 6.6 已完成内容与 100 samples 结果

Phase 4 可复现实验版本已经完成：

- 已实现 `CoverageAwareReranker`，支持 document coverage 和 entity coverage 奖励。
- 已实现 `ImprovedLightRAGRetriever`，封装 GraphRAG-style candidate generation + coverage-aware reranking。
- 已支持 `use_graph_expansion` 和 `use_coverage_reranking` 两个消融开关。
- 已新增 `configs/phase4_improved_lightrag.yaml` 和 `experiments/run_phase4_improved_lightrag.ps1`。
- 已新增 coverage reranker、Improved LightRAG 和 Phase 4 runner 测试。
- 已在 100 条 HotpotQA validation fair baseline 设置下完成实验。

与 Hybrid RAG 和 GraphRAG-style 对比：

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |
| GraphRAG-style | 0.805 | 0.322 | 0.7718 | 0.9095 | 0.620 | 0.990 | 0.0410s |
| Improved LightRAG | 0.805 | 0.322 | 0.7751 | 0.9095 | 0.620 | 0.990 | 0.0306s |

Top-10 对比：

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |
| GraphRAG-style | 0.885 | 0.177 | 0.8042 | 0.770 | 1.000 |
| Improved LightRAG | 0.890 | 0.178 | 0.8090 | 0.780 | 1.000 |

结论：Phase 4 基础版主要提升排序质量和 top-10 覆盖，`NDCG@5`、`NDCG@10`、`Recall@10` 和 `Full Evidence Recall@10` 相比 GraphRAG-style 有小幅提升。下一步需要进入 Phase 5 做系统消融，验证收益来自 graph expansion 还是 coverage reranking。

## 7. Phase 4.5: LightRAG Controlled Integration

### 7.1 定位

LightRAG controlled integration 用来把 LightRAG 对照组接入本项目统一评估框架，但它不是当前主线改进方法。

原因：直接接入官方 LightRAG demo 的研究故事不如自己的 GraphRAG-style + Evidence-aware Improved LightRAG 清楚。LightRAG 更适合作为外部方法对照，用来说明本项目方法和已有图增强 RAG 系统相比处在什么位置。

### 7.2 要求

如果接入 LightRAG，必须满足：

- 使用同一 HotpotQA subset。
- 使用同一 query set。
- 使用同一 top-k。
- 使用同一 retrieval metrics。
- 将 LightRAG 输出转换成统一 retrieval result schema。

计划文件：

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/lightrag_runner.py` | LightRAG 输入适配、索引构建、查询和结果捕获 |
| `configs/phase4_5_lightrag.yaml` | LightRAG 实验配置 |
| `experiments/run_phase4_5_lightrag.ps1` | 运行脚本 |
| `tests/test_lightrag_adapter.py` | 数据适配测试 |

### 7.3 已完成内容与 100 samples controlled result

Phase 4.5 已经完成 controlled integration layer：

- 已新增 `LightRAGResultAdapter`，把外部 LightRAG 结果转换为统一 retrieval result schema。
- 已新增 `LightRAGControlledRetriever`，支持 `method: lightrag` 进入统一 runner。
- 已支持 `backend: local_compat`，用于离线 smoke test 和指标管线验证。
- 已预留 `backend: external`，后续可接入真实 LightRAG runner。
- 已新增 `configs/phase4_5_lightrag.yaml` 和 `experiments/run_phase4_5_lightrag.ps1`。
- 已新增 `tests/test_lightrag_adapter.py`。
- 已在 100 条 HotpotQA validation fair baseline 设置下完成 controlled run。

当前结果：

| Method | Backend | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightRAG controlled | local_compat | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0199s |

注意：`local_compat` 使用 Hybrid RAG 作为离线兼容后端，因此这个结果不能作为官方 LightRAG 效果结论。它的意义是证明 LightRAG 对照组已经能进入统一数据、统一 top-k、统一指标和统一结果落盘流程。真实官方 LightRAG 对照需要后续提供 metadata-preserving external runner，并保证返回结果包含 `chunk_id` 和 `doc_id`。

## 8. Phase 5: Ablation Experiments

### 8.1 目标

通过消融实验验证改进模块确实有效，而不是偶然调参结果。

### 8.2 主消融设置

```text
Hybrid RAG
GraphRAG-style
Improved LightRAG
Improved LightRAG w/o graph expansion
Improved LightRAG w/o coverage reranking
Improved LightRAG depth=1
Improved LightRAG depth=2
```

### 8.3 辅助消融设置

```text
top_k = 3 / 5 / 10
chunk_size = 64 / 128 / 256
bm25_weight / dense_weight = 0.3/0.7, 0.5/0.5, 0.7/0.3
max_neighbors_per_entity = 3 / 5 / 10
```

### 8.4 重点分析

- Recall@5 是否提升。
- Full Evidence Recall@5 是否提升。
- Precision@5 是否下降。
- NDCG@5 是否提升。
- 检索延迟增加多少。
- 图扩展是否引入噪声。

### 8.5 阶段产出

- `results/ablation_table.csv`
- `docs/ablation_analysis.md`
- 方法模块贡献分析。
- 效果与成本 trade-off 分析。

## 9. Phase 6: Scale-up Experiments + Case Study

### 9.1 目标

让实验具备报告和面试说服力。

推荐规模：

```text
开发调试：100 samples
主实验：500 samples
可选扩展：1000 samples
```

### 9.2 Case Study 类型

需要从 per-query CSV 中筛选成功和失败案例，解释不同方法的行为差异。

建议至少分析：

- Hybrid RAG 找不到第二跳证据，但 GraphRAG-style 或 Improved LightRAG 找到了。
- Graph expansion 提高 Recall，但降低 Precision。
- Coverage-aware reranking 帮助 top-k 覆盖不同 supporting docs。
- 图噪声导致错误扩展。

### 9.3 阶段产出

- `docs/case_studies.md`
- 成功案例表。
- 失败案例表。
- 针对不同问题类型的效果分析。

## 10. Phase 7: Final Report + Resume Packaging

### 10.1 目标

把项目整理成可展示、可复现、可讲述的科研项目。

### 10.2 计划文件

| 文件 | 作用 |
| --- | --- |
| `docs/final_report.md` | 最终技术报告 |
| `docs/experiment_analysis.md` | 实验分析与图表说明 |
| `docs/resume_notes.md` | 简历与面试表述 |
| `results/main_comparison_table.csv` | 主结果表 |
| `results/ablation_table.csv` | 消融结果表 |

### 10.3 最终报告结构

1. 研究背景：RAG 与多跳问答。
2. 研究问题：图增强检索是否提升 supporting facts 完整召回。
3. 数据集：HotpotQA validation subset，后续可扩展 2WikiMultiHopQA。
4. 方法：Vector RAG、BM25、Hybrid RAG、GraphRAG-style、Evidence-aware Improved LightRAG。
5. 改进方法：Evidence-aware Graph Expansion 和 Coverage-aware Reranking。
6. 实验设置：统一 chunking、top-k、embedding、sample size。
7. 主实验结果。
8. 消融实验。
9. 效率和成本分析。
10. 成功案例与失败案例。
11. 结论与局限性。

## 11. 推荐实验矩阵

### 11.1 主方法对比

| 方法 | 使用图 | 使用 Hybrid seed | 使用 coverage reranking | 目的 |
| --- | --- | --- | --- | --- |
| Vector RAG | 否 | 否 | 否 | dense retrieval baseline |
| BM25 | 否 | 否 | 否 | lexical retrieval baseline |
| Hybrid RAG | 否 | 是 | 否 | 强检索 baseline |
| GraphRAG-style | 是 | 可选 | 否 | 图增强 baseline |
| Improved LightRAG | 是 | 是 | 是 | 核心改进方法 |
| LightRAG | 是 | 视实现而定 | 视实现而定 | 可选外部对照组 |

### 11.2 主评估指标

| 指标 | 用途 |
| --- | --- |
| Recall@k | 衡量 gold evidence 覆盖率 |
| Precision@k | 衡量检索结果噪声 |
| MRR@k | 衡量第一个相关证据是否靠前 |
| NDCG@k | 衡量整体排序质量 |
| Hit Rate@k | 衡量是否至少命中一个 gold evidence |
| Evidence Hit Count@k | 衡量命中的 gold evidence 数量 |
| Full Evidence Recall@k / Complete Evidence Hit@k | 衡量是否找全多跳证据链 |
| Retrieval Latency | 衡量查询阶段成本 |
| Index / Graph Construction Time | 衡量构建成本 |
| Retrieved Context Tokens | 衡量后续 generation 成本 |

### 11.3 推荐 k 值

主实验：

```text
k = 1, 3, 5, 10
```

报告重点：

```text
Recall@5
Full Evidence Recall@5
NDCG@5
Precision@5
Retrieval Latency
```

## 12. 当前下一步执行清单

Phase 4.5 LightRAG controlled integration 已经完成。下一步进入 Phase 5: Ablation experiments。

建议按下面顺序实现：

1. 新增消融配置：`phase5_improved_lightrag_no_graph.yaml`、`phase5_improved_lightrag_no_coverage.yaml`、`phase5_improved_lightrag_depth2.yaml`。
2. 新增 `experiments/run_phase5_ablations.ps1`，一次性跑 Hybrid、GraphRAG-style、Improved LightRAG 和消融变体。
3. 汇总 `results/*/aggregate_metrics.json`，生成 `results/ablation_table.csv`。
4. 分析 `Full Evidence Recall@5`、`NDCG@5`、`Precision@5`、`Retrieval Latency` 的变化。
5. 更新 `docs/ablation_analysis.md`，解释 graph expansion 和 coverage reranking 的模块贡献。
6. 如果消融结果稳定，再扩展到 500 条样本。

## 13. 简历叙事方向

当前阶段可以写：

> 构建面向 HotpotQA 多跳问答任务的可复现 RAG 对比实验框架，统一实现数据加载、文本切块、Vector RAG、BM25、Hybrid RAG、Recall@k / MRR / Hit Rate 指标评估与结果追踪，为 GraphRAG / LightRAG 的公平对比提供 baseline。

完成后续主线后，可以升级为：

> 构建面向 HotpotQA 多跳问答的可复现 RAG 对比实验框架，统一实现 Vector RAG、BM25、Hybrid RAG 与 GraphRAG-style 检索方法，并提出 Evidence-aware Improved LightRAG，通过图邻居扩展与证据覆盖感知重排序提升 supporting facts 完整召回。设计 Recall@k、Precision@k、MRR、NDCG、Full Evidence Recall、检索延迟等指标，在统一配置下完成多方法对比与消融实验，分析图增强检索在多跳问答中的效果与成本权衡。



