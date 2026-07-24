# Phase 5 初步消融分析 / Ablation Analysis

## 1. 目标

本阶段用于回答一个关键问题：

> 当前提升到底来自图路径、实体/别名处理、coverage reranking，还是只是参数调整？

因此本轮消融只使用同一 HotpotQA validation 100 samples、同一 chunking、同一 embedding、同一 top-k 和同一指标。

## 2. 消融设置

| Variant | 目的 |
| --- | --- |
| `phase2_5_fair_hybrid_rag` | 强 baseline，不使用图 |
| `phase3_graph_rag_style` | 句子级带权边 + 稀有实体加权 + 一跳距离衰减 |
| `phase4_improved_lightrag` | GraphRAG-style + coverage-aware reranking |
| `phase5_improved_lightrag_no_aliases` | 关闭别名，测试 alias 对召回和噪声的影响 |
| `phase5_improved_lightrag_no_graph_expansion` | 关闭图扩展，测试收益是否来自图路径 |
| `phase5_improved_lightrag_no_coverage_reranking` | 关闭 coverage reranking，测试重排序贡献 |
| `phase5_improved_lightrag_no_entity_coverage` | 只关闭 entity coverage 奖励，测试实体覆盖重排贡献 |

## 3. 结果

完整 CSV 表：

```text
results/ablation_table.csv
```

核心结果：

| Variant | Recall@5 | Precision@5 | NDCG@5 | Full Evidence Recall@5 | Recall@10 | Full Evidence Recall@10 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.590 | 0.890 | 0.780 | 0.0211s |
| GraphRAG-style | 0.810 | 0.324 | 0.7730 | 0.630 | 0.920 | 0.840 | 0.1355s |
| Improved LightRAG | 0.825 | 0.330 | 0.7834 | 0.660 | 0.920 | 0.840 | 0.1087s |
| w/o aliases | 0.790 | 0.316 | 0.7674 | 0.610 | 0.925 | 0.850 | 0.0782s |
| w/o graph expansion | 0.790 | 0.316 | 0.7584 | 0.590 | 0.895 | 0.790 | 0.0219s |
| w/o coverage reranking | 0.810 | 0.324 | 0.7730 | 0.630 | 0.920 | 0.840 | 0.1102s |
| w/o entity coverage | 0.810 | 0.324 | 0.7759 | 0.630 | 0.925 | 0.850 | 0.1048s |

## 4. 结论

### 4.1 图扩展是主要收益来源

关闭图扩展后，`Recall@5`、`Precision@5`、`Full Evidence Recall@5` 基本回到 Hybrid RAG：

```text
Improved LightRAG: 0.825 Recall@5 / 0.660 Full Evidence Recall@5
w/o graph expansion: 0.790 Recall@5 / 0.590 Full Evidence Recall@5
```

这说明当前提升不是单纯来自参数调整，而是来自图路径带来的候选补充。

### 4.2 Coverage reranking 提升 top-5 完整证据覆盖

关闭 coverage reranking 后，结果退回 GraphRAG-style：

```text
GraphRAG-style: 0.810 Recall@5 / 0.630 Full Evidence Recall@5
w/o coverage reranking: 0.810 Recall@5 / 0.630 Full Evidence Recall@5
Improved LightRAG: 0.825 Recall@5 / 0.660 Full Evidence Recall@5
```

这说明 coverage reranking 的主要贡献是把图候选池中更互补的证据推进 top-5。

### 4.3 别名是双刃剑

关闭别名后，图规模明显下降：

```text
With aliases: 12560 entities / 113065 edges
w/o aliases: 8662 entities / 34226 edges
```

但 top-5 也下降：

```text
Improved LightRAG: 0.825 Recall@5 / 0.660 Full Evidence Recall@5
w/o aliases: 0.790 Recall@5 / 0.610 Full Evidence Recall@5
```

这说明 conservative aliases 对 top-5 有帮助，但也增加了图规模。后续需要继续控制别名噪声，而不是完全关闭别名。

### 4.4 Entity coverage 主要影响 top-5 排序

关闭 entity coverage 后，top-5 回落，但 top-10 略高。这说明 entity coverage 奖励更偏向把实体互补的结果提前，而不是扩大候选池本身。

## 5. 当前定位

当前 Phase 3 最准确的定位是：

> 带有稀有实体加权、句子级带权边和一跳距离衰减的轻量图感知检索 baseline。

它不是 Microsoft GraphRAG，也不是完整 LightRAG。它的价值在于：

- 可复现。
- 可解释。
- 能记录图路径。
- 可以通过消融说明图扩展和 coverage reranking 的贡献。

## 6. 下一步

建议继续做：

1. 对 `graph_paths` 做成功/失败案例筛选。
2. 分析错误路径来自哪些实体或别名。
3. 尝试更严格的 alias whitelist / document-title-aware aliases。
4. 在 500 samples 上重跑主实验和关键消融。
