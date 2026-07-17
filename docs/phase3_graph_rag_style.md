# Phase 3 详细说明：GraphRAG-style Baseline

## 1. 阶段目标

Phase 3 的目标是实现一个轻量、可解释、可控的 GraphRAG-style baseline，用来验证图结构是否能帮助多跳问答检索找全 supporting facts。

这一版不是完整复现微软 GraphRAG，也不使用 LLM 做实体关系抽取，而是先实现 retrieval-side 的核心图增强思想：

```text
chunks
  -> entity extraction
  -> entity-chunk graph construction
  -> Hybrid seed retrieval
  -> graph neighbor expansion
  -> seed score + graph score fusion
  -> final top-k evidence
```

## 2. 为什么先做轻量版

Phase 2.5 已经证明 Hybrid RAG 是当前强 baseline。Phase 3 的重点不是堆复杂模型，而是回答一个更基础的问题：

> 如果在 Hybrid seed retrieval 之后，用实体图扩展相关 chunk，是否更容易找全多跳 supporting facts？

轻量版的好处是：

- 可解释：每个扩展结果都能追溯到实体重合或实体邻居。
- 成本低：不用调用 LLM 抽实体，方便快速迭代。
- 可控：可以调 expansion depth、graph weight、max neighbors 等参数。
- 适合做后续 Improved LightRAG 的基础模块。

## 3. 已实现文件

| 文件 | 作用 |
| --- | --- |
| `src/graph/entity_extractor.py` | 规则实体抽取，识别大写专有名词和多词实体 |
| `src/graph/graph_builder.py` | 从 chunks 构建 GraphIndex |
| `src/indexing/graph_index.py` | 存储 entity-to-chunk、chunk-to-entity 和 entity co-occurrence 图 |
| `src/retrieval/graph_rag_style.py` | GraphRAG-style 检索器 |
| `configs/phase3_graph_rag_style.yaml` | Phase 3 fair experiment 配置 |
| `experiments/run_phase3_graph_rag_style.ps1` | Phase 3 运行脚本 |
| `tests/test_entity_extractor.py` | 实体抽取测试 |
| `tests/test_graph_index.py` | 图索引测试 |
| `tests/test_graph_retrieval.py` | 图检索测试 |
| `tests/test_phase3_runner.py` | Phase 3 runner 端到端测试 |

## 4. 实体抽取

当前实体抽取使用 `SimpleEntityExtractor`，基于规则识别英文文本中的大写短语，例如：

```text
Ada Lovelace
Analytical Engine
Charles Babbage
```

实体会被 normalize 成小写形式：

```text
"Ada Lovelace" -> "ada lovelace"
```

这一版会过滤过短实体和常见停用词。它不完美，但足够支撑第一版 GraphRAG-style baseline。

## 5. 图结构

`GraphIndex` 维护三类关系：

```text
entity -> chunk ids
chunk -> entities
entity -> neighboring entities
```

其中 entity-neighbor edge 来自同一个 chunk 中的实体共现。例如一个 chunk 同时包含：

```text
Ada Lovelace
Analytical Engine
Charles Babbage
```

那么这些实体之间会建立共现边。检索时可以从 query entity 或 seed chunk entity 出发，找到图上的相邻实体和对应 chunks。

## 6. 检索流程

GraphRAG-style 检索流程：

1. 使用 Hybrid RAG 先取 seed results。
2. 从 query 中抽取 query entities。
3. 从 seed chunks 中收集 seed entities。
4. 在 graph index 中根据 query entities 和 seed entities 扩展邻居 chunks。
5. 合并 seed candidates 和 graph-expanded candidates。
6. 计算：

```text
final_score = seed_weight * seed_score + graph_weight * graph_score
```

当前配置：

```text
seed_weight: 0.7
graph_weight: 0.3
expansion_depth: 1
max_neighbors_per_entity: 10
```

## 7. 实验配置

配置文件：

```text
configs/phase3_graph_rag_style.yaml
```

统一设置沿用 Phase 2.5 fair baseline：

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

运行方式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase3_graph_rag_style.ps1 -Python .\.venv\Scripts\python.exe
```

## 8. 结果

与 Phase 2.5 Hybrid RAG fair baseline 对比：

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |
| GraphRAG-style | 0.805 | 0.322 | 0.7718 | 0.9095 | 0.620 | 0.990 | 0.0410s |

Top-10 对比：

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |
| GraphRAG-style | 0.885 | 0.177 | 0.8042 | 0.770 | 1.000 |

图统计：

| Metric | Value |
| --- | ---: |
| Graph entities | 8238 |
| Graph edges | 66592 |
| Index / graph construction time | 22.94s |
| Avg retrieval latency | 0.0410s |

## 9. 结果解释

Phase 3 的轻量 GraphRAG-style baseline 在 top-5 上相比 Hybrid RAG 有小幅提升：

- `Recall@5`: 0.790 -> 0.805
- `Full Evidence Recall@5`: 0.590 -> 0.620
- `NDCG@5`: 0.7582 -> 0.7718
- `MRR@5`: 0.8995 -> 0.9095

这说明图邻居扩展确实能帮助找回一部分 Hybrid 漏掉的多跳证据。但代价也明显：

- 平均检索延迟从 0.0213s 增加到 0.0410s。
- 图构建时间增加到 22.94s。
- Top-10 的 Recall 和 Full Evidence Recall 没有超过 Hybrid，说明朴素图扩展仍然会引入噪声。

## 10. 阶段结论

Phase 3 已经证明：轻量图增强在 top-5 多跳证据召回上有一定价值，但还不是最终方法。

这为 Phase 4 提供了清晰动机：

```text
GraphRAG-style 能提升一部分证据召回，
但需要 Evidence-aware Graph Expansion 和 Coverage-aware Reranking
来减少图扩展噪声、提高完整证据链覆盖。
```

下一步应该进入 Phase 4：Evidence-aware Improved LightRAG。
