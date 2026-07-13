# Phase 2.5 详细说明：Fair Baseline + Metrics Upgrade

## 1. 阶段目标

Phase 2.5 的目标是把已经完成的 Vector RAG、BM25 和 Hybrid RAG 变成严格公平、可引用的 baseline。

这个阶段解决两个问题：

1. 历史实验中 Vector RAG 和 BM25 / Hybrid 的 chunk 配置不完全一致，不能作为最终主结果。
2. 原有指标只有 Recall@k、MRR@k 和 Hit Rate@k，还不足以分析多跳证据链是否完整、检索结果是否有噪声、排序质量是否稳定。

因此 Phase 2.5 统一了实验配置，并补充了更完整的 retrieval-side metrics。

## 2. 统一实验配置

三种方法现在使用同一套 fair configs：

| Method | Config |
| --- | --- |
| Vector RAG | `configs/phase2_vector_rag_fair.yaml` |
| BM25 | `configs/phase2_bm25_rag_fair.yaml` |
| Hybrid RAG | `configs/phase2_hybrid_rag_fair.yaml` |

统一设置：

```text
Dataset: HotpotQA validation
Sample size: 100
Seed: 42
Chunk size: 64
Overlap: 8
Top-k: 10
Metric k values: 1, 3, 5, 10
Embedding: sentence-transformers/all-MiniLM-L6-v2
```

Hybrid RAG 额外设置：

```text
candidate_k: 40
bm25_weight: 0.5
dense_weight: 0.5
fusion: weighted
```

## 3. 新增指标

Phase 2.5 在 `src/evaluation/retrieval_metrics.py` 中补充了：

| Metric | 中文说明 | 用途 |
| --- | --- | --- |
| `precision@k` | top-k 中相关证据比例 | 衡量噪声 |
| `ndcg@k` | 相关证据整体排序质量 | 比 MRR 更适合多证据场景 |
| `evidence_hit_count@k` | top-k 中命中的 gold evidence 数量 | 直观看到找回了几个 supporting docs |
| `full_evidence_recall@k` | 是否找全一个问题所需全部 gold evidence | 多跳问答最关键的完整证据链指标 |
| `retrieved_context_tokens@k` | top-k 返回文本的 token-like word 数 | 估算后续 generation 成本 |

原有指标仍然保留：

- `recall@k`
- `mrr@k`
- `hit_rate@k`

## 4. 为什么 Full Evidence Recall 很重要

HotpotQA 是多跳问答，很多问题需要两个或多个 supporting documents。只命中其中一个证据时：

```text
Hit Rate@k = 1
Recall@k = 0.5
Full Evidence Recall@k = 0
```

这说明系统虽然“碰到”了相关文档，但没有找全证据链。后续 GraphRAG-style 和 Improved LightRAG 的核心目标，就是提高这种完整证据链召回能力。

## 5. 运行方式

一键运行三种 fair baseline：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_fair_baselines.ps1 -Python .\.venv\Scripts\python.exe
```

单独运行某个配置：

```powershell
.\.venv\Scripts\python.exe -m src.run_experiment --config configs\phase2_hybrid_rag_fair.yaml
```

## 6. 输出文件

三种方法分别输出：

```text
results/phase2_5_fair_vector_rag/per_query_results.csv
results/phase2_5_fair_vector_rag/aggregate_metrics.json
results/phase2_5_fair_bm25_rag/per_query_results.csv
results/phase2_5_fair_bm25_rag/aggregate_metrics.json
results/phase2_5_fair_hybrid_rag/per_query_results.csv
results/phase2_5_fair_hybrid_rag/aggregate_metrics.json
```

`per_query_results.csv` 包含每个问题的检索结果和所有指标。  
`aggregate_metrics.json` 包含整体平均指标。

## 7. 100 条 HotpotQA Fair Baseline 结果

| Method | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.725 | 0.290 | 0.6901 | 0.8325 | 0.510 | 0.940 | 0.0108s |
| BM25 | 0.710 | 0.284 | 0.6668 | 0.8200 | 0.430 | 0.990 | 0.0053s |
| Hybrid RAG | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0213s |

Top-10 结果：

| Method | Recall@10 | Precision@10 | NDCG@10 | Full Evidence Recall@10 | Hit Rate@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.835 | 0.167 | 0.7339 | 0.670 | 1.000 |
| BM25 | 0.885 | 0.177 | 0.7356 | 0.770 | 1.000 |
| Hybrid RAG | 0.890 | 0.178 | 0.7988 | 0.780 | 1.000 |

## 8. 结果解释

在统一配置下，Hybrid RAG 仍然是当前最强 baseline：

- `Recall@5 = 0.790`，高于 Vector RAG 的 0.725 和 BM25 的 0.710。
- `Full Evidence Recall@5 = 0.590`，说明 59% 的问题在 top-5 中找全了全部 supporting documents。
- `NDCG@5 = 0.7582`，说明 Hybrid 不只是召回更多，排序质量也更好。
- `Precision@5 = 0.316`，也高于 Vector 和 BM25，说明没有单纯靠引入更多噪声换召回。
- 平均检索延迟 `0.0213s`，高于 Vector 和 BM25，因为它同时执行 dense 与 lexical 两路检索。

BM25 的 `Hit Rate@5 = 0.990` 很高，说明它很容易命中至少一个相关文档；但 `Full Evidence Recall@5 = 0.430`，说明它不擅长完整找齐多跳证据链。

## 9. 阶段结论

Phase 2.5 建立了更可信的 baseline：

```text
Vector RAG < BM25 / lexical baseline < Hybrid RAG strong baseline
```

后续 GraphRAG-style 和 Improved LightRAG 不能只超过 Vector RAG，而应该至少和 Hybrid RAG 对比。重点指标也不应只看 Recall@5，而要看：

- Full Evidence Recall@5 是否提升。
- NDCG@5 是否提升。
- Precision@5 是否下降。
- Retrieval latency 增加是否可接受。

## 10. 下一步

下一步进入 Phase 3：GraphRAG-style baseline。

Phase 3 的核心问题是：

> 图结构是否能帮助检索器找到 Hybrid RAG 漏掉的第二跳或多跳 supporting evidence？
