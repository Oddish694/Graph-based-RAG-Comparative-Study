# Phase 3 详细说明：GraphRAG-style Baseline

## 1. 阶段目标

Phase 3 的目标是实现一个轻量、可解释、可控的图感知检索 baseline，用来验证图结构是否能帮助多跳问答检索找全 supporting facts。

这一版不是完整复现微软 GraphRAG，也不使用 LLM 做实体关系抽取，而是先实现 retrieval-side 的核心图增强思想：

```text
chunks
  -> entity extraction
  -> sentence-level weighted entity graph construction
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
| `src/graph/entity_extractor.py` | 增强规则实体抽取，识别专有名词、缩写、标题短语和谨慎别名 |
| `src/graph/graph_builder.py` | 从 chunks 构建句子级 entity groups |
| `src/indexing/graph_index.py` | 存储 entity-to-chunk、chunk-to-entity、句子级带权边和边证据 |
| `src/retrieval/graph_rag_style.py` | GraphRAG-style 检索器 |
| `configs/phase3_graph_rag_style.yaml` | Phase 3 fair experiment 配置 |
| `experiments/run_phase3_graph_rag_style.ps1` | Phase 3 运行脚本 |
| `tests/test_entity_extractor.py` | 实体抽取测试 |
| `tests/test_graph_index.py` | 图索引测试 |
| `tests/test_graph_retrieval.py` | 图检索测试 |
| `tests/test_phase3_runner.py` | Phase 3 runner 端到端测试 |

## 4. 实体抽取

当前实体抽取使用 `SimpleEntityExtractor`，基于规则识别英文文本中的专有名词、缩写和标题短语，例如：

```text
Ada Lovelace
Analytical Engine
Charles Babbage
U.S.
"Apollo Guidance Computer"
```

实体会被 normalize 成小写形式：

```text
"Ada Lovelace" -> "ada lovelace"
```

这一版比最初的大写短语匹配更强，新增了：

- 带连接词的实体短语，例如 `University of Oxford`。
- 缩写实体，例如 `U.S.`、`NASA`。
- 引号中的标题或作品名。
- 谨慎别名，例如 `Charles Babbage -> babbage`，以及多词实体首字母缩写。

同时，为了避免图过度膨胀，系统不会把 `city`、`engine`、`computer` 这类泛化尾词单独作为别名。

## 5. 图结构

`GraphIndex` 维护四类关系：

```text
entity -> chunk ids
chunk -> entities
entity -> weighted neighboring entities
edge -> evidence chunks
```

其中 entity-neighbor edge 来自同一句中的实体共现，而不是整个 chunk 的全连接共现。例如：

```text
Sentence 1: Ada Lovelace wrote about the Analytical Engine.
Sentence 2: Charles Babbage designed the Analytical Engine.
```

系统只会连接同一句中的实体，并记录边权和边来源 chunk。这样可以减少“同一 chunk 但不同句”的错误连接。检索时可以从 query entity 或高信息量 seed entity 出发，找到图上的相邻实体、对应 chunks 和完整路径。

## 6. 检索流程

GraphRAG-style 检索流程：

1. 使用 Hybrid RAG 先取 seed results。
2. 从 query 中抽取 query entities。
3. 从 seed chunks 中收集 seed entities。
4. 在 graph index 中根据 query entities 和 seed entities 扩展邻居 chunks。
5. 合并 seed candidates 和 graph-expanded candidates。
6. 计算可解释的图分数：

```text
graph_score =
  query_entity_weight * query_entity_score
  + seed_entity_weight * seed_entity_score
  + expansion_weight * graph_proximity_score

final_score =
  seed_weight * seed_score
  + graph_weight * graph_score
```

其中：

- `query_entity_score`：候选 chunk 是否直接覆盖 query 中的实体。
- `seed_entity_score`：候选 chunk 是否覆盖高信息量 seed 实体。
- `graph_proximity_score`：候选 chunk 实体距离 query / seed 实体有多近，并使用 `distance_decay` 做距离衰减。
- 实体权重按出现频率反比计算，越稀有的实体权重越高。
- `max_seed_entities` 限制从 seed chunks 中进入图扩展的实体数量，避免图扩展过宽。
- `graph_paths` 记录候选结果对应的 query/seed entity path、边权和边证据。

当前配置：

```text
seed_weight: 0.7
graph_weight: 0.3
expansion_depth: 1
max_neighbors_per_entity: 10
max_seed_entities: 25
query_entity_weight: 0.45
seed_entity_weight: 0.25
expansion_weight: 0.30
distance_decay: 0.5
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
| GraphRAG-style | 0.810 | 0.324 | 0.7730 | 0.9117 | 0.630 | 0.990 | 0.1355s |

Top-10 对比：

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |
| GraphRAG-style | 0.920 | 0.184 | 0.8166 | 0.840 | 1.000 |

图统计：

| Metric | Value |
| --- | ---: |
| Graph entities | 12560 |
| Graph edges | 113065 |
| Index / graph construction time | 21.75s |
| Avg retrieval latency | 0.1355s |

## 9. 结果解释

Phase 3 的轻量 GraphRAG-style baseline 在 top-5 和 top-10 上相比 Hybrid RAG 有提升：

- `Recall@5`: 0.790 -> 0.810
- `Full Evidence Recall@5`: 0.590 -> 0.630
- `NDCG@5`: 0.7582 -> 0.7730
- `MRR@5`: 0.8995 -> 0.9117
- `Recall@10`: 0.890 -> 0.920
- `Full Evidence Recall@10`: 0.780 -> 0.840

这说明图邻居扩展确实能帮助找回一部分 Hybrid 漏掉的多跳证据。但代价也明显：

- 平均检索延迟从 0.0213s 增加到 0.1355s。
- 图构建时间增加到 21.75s。
- Top-10 的 Recall 和 Full Evidence Recall 已经超过 Hybrid，但图规模和延迟也更高，说明图增强有效但需要继续做效率和噪声控制。

## 10. 阶段结论

Phase 3 当前最准确的定位是：

```text
带有稀有实体加权、句子级带权边和一跳距离衰减的轻量图感知检索 baseline。
```

它已经证明：轻量图增强在 top-5 和 top-10 多跳证据召回上有价值，但还不是完整 GraphRAG 系统。

这为 Phase 4 提供了清晰动机：

```text
GraphRAG-style 能提升一部分证据召回，
但需要 Evidence-aware Graph Expansion 和 Coverage-aware Reranking
来减少图扩展噪声、提高完整证据链覆盖。
```

下一步应该进入 Phase 4：Evidence-aware Improved LightRAG。
