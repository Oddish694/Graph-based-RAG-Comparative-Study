# Phase 4.5 详细说明：LightRAG Controlled Integration

## 1. 阶段定位

Phase 4.5 的目标是把外部 LightRAG 方法纳入本项目的统一评估框架，作为可选对照组。

这个阶段不改变项目主线方法。项目主线仍然是：

```text
Hybrid RAG
  -> GraphRAG-style
  -> Evidence-aware Improved LightRAG
  -> Ablation experiments
```

Phase 4.5 解决的是工程对齐问题：外部 LightRAG 的输出格式、数据输入方式和查询流程通常不一定和本项目一致，所以需要一个 controlled integration layer，把外部结果转换成项目统一的 retrieval result schema，然后才能用同一套 Recall@k、MRR、NDCG、Full Evidence Recall 等指标评估。

## 2. 当前实现状态

当前已经实现 LightRAG controlled integration 的可运行版本：

| 文件 | 作用 |
| --- | --- |
| `src/retrieval/lightrag_runner.py` | LightRAG 适配层、结果标准化器和 controlled retriever |
| `configs/phase4_5_lightrag.yaml` | Phase 4.5 配置 |
| `experiments/run_phase4_5_lightrag.ps1` | Phase 4.5 运行脚本 |
| `tests/test_lightrag_adapter.py` | 适配器、后端和 runner 测试 |

当前默认后端是：

```yaml
backend: local_compat
```

这表示使用项目内部 Hybrid RAG 检索器模拟 LightRAG 接入流程，用来验证：

- 数据是否能进入同一个实验 runner。
- 检索结果是否能转换成统一 schema。
- per-query CSV 和 aggregate metrics 是否能正常输出。
- 后续替换成真实 LightRAG 后是否仍能复用同一套评估代码。

重要说明：`local_compat` 不是官方 LightRAG 成绩，它只是 Phase 4.5 的可控接入与离线 smoke test 后端。

## 3. 为什么不直接把官方 LightRAG 当作主线

直接接入官方 LightRAG 有几个需要控制的问题：

- 外部库通常更偏向完整 RAG 应用，输出可能是答案文本，而不是带 `chunk_id` / `doc_id` 的检索证据列表。
- 本项目评估的是 retrieval-side evidence quality，必须保留每条 evidence 的 metadata。
- 如果外部方法内部使用不同 chunking、embedding、top-k 或索引策略，实验就不再公平。
- 官方 LightRAG 可能需要 LLM、embedding API、工作目录、图存储和缓存配置，工程变量比当前本地实验多。

因此 Phase 4.5 先完成 controlled adapter，再决定是否接入真实外部运行器。

## 4. 统一结果 Schema

本项目所有检索器都需要返回类似下面的结构：

```python
{
    "chunk_id": "...",
    "doc_id": "...",
    "text": "...",
    "score": 0.0,
    "retriever_source": "..."
}
```

`LightRAGResultAdapter` 支持把外部结果中的常见字段转换过来：

| 外部可能字段 | 项目统一字段 |
| --- | --- |
| `chunk_id` / `id` / `source_id` | `chunk_id` |
| `doc_id` / `document_id` / `source_doc_id` / `title` | `doc_id` |
| `text` / `content` / `chunk` / `context` | `text` |
| `score` / `similarity` / `distance` / `rank_score` | `score` |

如果真实 LightRAG 后续能返回 evidence metadata，就可以通过这个 adapter 接入统一评估。

## 5. 当前工作流程

当前 `local_compat` 流程如下：

```text
HotpotQA samples
  -> fixed chunking
  -> LightRAGControlledRetriever
  -> local_compat backend
  -> Hybrid RAG retrieval
  -> LightRAG-compatible result schema
  -> retrieval metrics
  -> CSV / JSON results
```

真实外部 LightRAG 目标流程如下：

```text
HotpotQA samples
  -> fixed chunking with stable chunk_id/doc_id
  -> external LightRAG index
  -> LightRAG query
  -> metadata-preserving retrieved contexts
  -> LightRAGResultAdapter
  -> retrieval metrics
```

## 6. 运行方式

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_phase4_5_lightrag.ps1 -Python .\.venv\Scripts\python.exe
```

配置文件：

```text
configs/phase4_5_lightrag.yaml
```

## 7. 100 Samples Controlled Result

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
Backend: local_compat
```

结果：

| Method | Backend | Recall@5 | Precision@5 | NDCG@5 | MRR@5 | Full Evidence Recall@5 | Hit Rate@5 | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightRAG controlled | local_compat | 0.790 | 0.316 | 0.7582 | 0.8995 | 0.590 | 0.990 | 0.0199s |

这个结果和 Hybrid RAG 接近是预期现象，因为 `local_compat` 使用 Hybrid RAG 作为离线兼容后端。它的意义不是证明 LightRAG 效果，而是证明 LightRAG 对照组已经能进入统一实验框架。

## 8. 阶段结论

Phase 4.5 已经完成 controlled integration layer：

- 已支持 `method: lightrag`。
- 已支持 `backend: local_compat` 离线评估。
- 已预留 `backend: external` 接口。
- 已实现外部结果标准化。
- 已通过单元测试和 100 samples runner。

后续如果要拿官方 LightRAG 作为正式对照，需要再补一个 metadata-preserving external runner。关键要求是：真实 LightRAG 查询结果必须能返回 `chunk_id` 和 `doc_id`，否则无法公平计算 HotpotQA supporting facts retrieval metrics。
