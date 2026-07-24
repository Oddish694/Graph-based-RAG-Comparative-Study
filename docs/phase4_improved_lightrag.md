# Phase 4 详细说明：Evidence-aware Improved LightRAG

## 1. 阶段目标

Phase 4 的目标是在 Phase 3 GraphRAG-style baseline 基础上，实现一个更面向多跳证据完整召回的 Improved LightRAG-style retriever。

这里的 Improved LightRAG 不是官方 LightRAG 的直接复现，而是本项目自己的轻量改进方法：

```text
Hybrid seed retrieval
  -> graph neighbor expansion
  -> evidence-aware candidate scoring
  -> coverage-aware reranking
  -> final top-k evidence
```

它的核心动机是：Phase 3 证明图邻居扩展可以提高 top-5 的证据召回，但朴素扩展可能引入噪声。Phase 4 通过覆盖感知重排序，让 top-k 更倾向于覆盖不同文档和不同实体线索。

## 2. 已实现文件

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/coverage_reranker.py` | 覆盖感知重排序模块 |
| `src/retrieval/improved_lightrag.py` | Improved LightRAG-style 主检索器 |
| `configs/phase4_improved_lightrag.yaml` | Phase 4 fair experiment 配置 |
| `experiments/run_phase4_improved_lightrag.ps1` | Phase 4 运行脚本 |
| `tests/test_coverage_reranker.py` | coverage reranker 测试 |
| `tests/test_improved_lightrag.py` | Improved LightRAG 检索器测试 |
| `tests/test_phase4_runner.py` | Phase 4 runner 端到端测试 |

## 3. 工作流程

Phase 4 的检索流程如下：

1. 使用 GraphRAG-style 生成候选池。这个候选池本身来自 Hybrid seed retrieval 和 graph neighbor expansion。
2. 每个候选 chunk 保留 `score`、`seed_score`、`graph_score`、`matched_entities` 等字段。
3. Coverage-aware reranker 逐个选择 top-k 结果。
4. 每一步选择时综合考虑：
   - 原始相关性分数。
   - 当前候选是否来自尚未覆盖的 document。
   - 当前候选是否带来新的 matched entities。
5. 最终返回统一检索结果，并标记 `retriever_source = improved_lightrag`。

## 4. Coverage-aware Reranking

重排序模块 `CoverageAwareReranker` 使用贪心选择。

直观 scoring：

```text
rerank_score = relevance_weight * base_score
             + document_coverage_bonus
             + entity_coverage_bonus
```

当前配置：

```text
coverage_weight: 0.05
entity_coverage_weight: 0.03
relevance_weight: 1.0
```

这个权重比较温和，目的是避免为了多样性牺牲太多相关性。系统会优先保留 GraphRAG-style 的强相关候选，同时在分数接近时鼓励覆盖新的文档和实体线索。

## 5. 可消融开关

Phase 4 已经预留两个开关，方便 Phase 5 做消融：

```yaml
use_graph_expansion: true
use_coverage_reranking: true
```

后续可以对比：

```text
Improved LightRAG
w/o graph expansion
w/o coverage reranking
depth=1
depth=2
```

## 6. 实验设置

配置文件：

```text
configs/phase4_improved_lightrag.yaml
```

统一设置沿用 Phase 2.5 / Phase 3 fair baseline：

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
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase4_improved_lightrag.ps1 -Python .\.venv\Scripts\python.exe
```

## 7. 结果

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

## 8. 结果解释

Phase 4 的基础版没有继续提高 Recall@5 和 Full Evidence Recall@5，但改善了排序质量：

- `NDCG@5`: GraphRAG-style 0.7718 -> Improved LightRAG 0.7751
- `NDCG@10`: GraphRAG-style 0.8042 -> Improved LightRAG 0.8090
- `Recall@10`: GraphRAG-style 0.885 -> Improved LightRAG 0.890
- `Full Evidence Recall@10`: GraphRAG-style 0.770 -> Improved LightRAG 0.780

这说明 coverage-aware reranking 的第一版主要收益在候选排序和 top-10 完整覆盖上，而不是直接提升 top-5 召回。

## 9. 阶段结论

Phase 4 建立了 Improved LightRAG 的基础实现，并验证 coverage-aware reranking 有一定排序收益。

下一步应该进入 Phase 5：系统消融。

重点问题：

- coverage 权重是否需要调大或动态化？
- depth=2 是否能提高 Full Evidence Recall，但会不会引入更多噪声？
- 去掉 graph expansion 后，收益是否消失？
- 去掉 coverage reranking 后，NDCG 是否回落？
