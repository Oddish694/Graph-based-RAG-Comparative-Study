# Phase 2 详细说明：BM25 + Hybrid RAG Baseline

## 1. Phase 2 的目标

Phase 2 的目标是在 Phase 1 Vector RAG baseline 之后，加入 BM25 关键词检索和 Hybrid RAG 混合检索。

它要回答的问题是：

> 在同一批 HotpotQA 多跳问答样本上，BM25 这种关键词检索、Vector RAG 这种语义检索、以及 BM25 + Vector 的 Hybrid RAG，谁更能找回 supporting facts？

Phase 2 并不是最终的图增强方法，而是进入 GraphRAG / LightRAG 之前必须补齐的第二个基础对照组。这样后面如果图增强方法有效，我们可以判断它是否真的超过了一个比较强的 hybrid baseline，而不是只超过了最基础的 Vector RAG。

## 2. 为什么需要 BM25

Vector RAG 把 question 和 chunk 都转成 embedding，然后按向量相似度检索。它擅长语义相似，例如 question 里说 “writer”，chunk 里说 “author”，向量模型可能仍然能判断它们相关。

但 Vector RAG 也有局限：

- 对专有名词、年份、实体名、罕见词，有时不如关键词匹配稳定。
- 如果问题里有非常关键的 exact phrase，embedding 可能把它稀释掉。
- 多跳问答中第二跳证据可能和问题表面语义不强相关，单靠 dense retrieval 可能漏掉。

BM25 是传统 lexical retrieval 方法，核心优势是 exact match。只要 query 中的关键词在 chunk 中出现，尤其是稀有词出现，它就会给较高分数。

所以 Phase 2 加入 BM25 是为了建立 lexical baseline，并测试关键词匹配是否能弥补 dense retrieval 的短板。

## 3. BM25 的输入是什么

BM25 使用的输入和 Vector RAG 一样，都是 Phase 1 产生的 chunks。

一个 chunk 大致是：

```python
{
    "sample_id": "...",
    "doc_id": "Babbage",
    "title": "Babbage",
    "chunk_id": "Babbage::0",
    "chunk_index": 0,
    "text": "Charles Babbage designed the Analytical Engine ..."
}
```

BM25 不直接使用 embedding，而是对 `text` 做 tokenization，统计每个词在当前 chunk 和整个语料中的出现情况。

## 4. BM25Index 做了什么

实现文件：

```text
src/indexing/bm25_index.py
```

它主要做四件事：

1. Tokenization：把文本小写化，并按英文单词、数字、下划线切成 tokens。
2. Term frequency：统计每个 token 在当前 chunk 中出现多少次。
3. Document frequency：统计每个 token 出现在多少个 chunks 中。
4. BM25 scoring：查询时根据 query tokens 给每个 chunk 打分。

BM25 分数直观上由三部分影响：

- query 词是否出现在 chunk 中。
- 这个词在 chunk 中出现得是否足够多。
- 这个词在整个语料中是否足够稀有。

如果一个词在所有 chunk 里都出现，它区分度低，分数贡献较小。如果一个词很少出现，但刚好出现在某个 chunk 中，它更可能是重要线索。

## 5. BM25RAGRetriever 做了什么

实现文件：

```text
src/retrieval/bm25_rag.py
```

它是一个很薄的封装层，作用是把 `BM25Index.search()` 包装成和其他 retriever 一样的接口：

```python
retrieve(query, top_k=5) -> list[retrieval_result]
```

返回结果里会带上：

- `doc_id`
- `chunk_id`
- `text`
- `score`
- `retriever_source = "bm25"`

这样 BM25、Vector、Hybrid 可以复用同一套 evaluation pipeline。

## 6. Hybrid RAG 是什么

Hybrid RAG 指的是同时跑两种检索：

```text
question
  -> BM25 lexical retrieval
  -> dense vector retrieval
  -> fusion
  -> final top-k chunks
```

它不是“什么都不做”的 baseline，而是比 Vector RAG 更强一点的检索 baseline。它试图把 BM25 的 exact match 能力和 dense retrieval 的 semantic matching 能力合起来。

实现文件：

```text
src/retrieval/hybrid_rag.py
```

## 7. 为什么需要 score fusion

BM25 和 dense retrieval 的分数不是同一个尺度：

- BM25 分数来自词频、逆文档频率和长度归一化。
- dense 分数来自向量相似度，通常是 cosine similarity 或 dot product。

如果直接把两个分数相加，不公平。比如 BM25 的原始分数可能是 5.2，dense similarity 可能是 0.43，这两个数不能直接比较。

所以 Hybrid RAG 先做归一化，再融合。

## 8. Weighted score fusion 怎么做

当前默认方式是 weighted score fusion：

```text
hybrid_score = bm25_weight * normalized_bm25_score
             + dense_weight * normalized_dense_score
```

当前配置是：

```yaml
bm25_weight: 0.5
dense_weight: 0.5
fusion: weighted
```

也就是说 BM25 和 dense retrieval 各占一半权重。

归一化后，每一路分数大致压到 0 到 1 之间，这样两路分数可以放到同一尺度上比较。

## 9. Reciprocal Rank Fusion 是什么

Phase 2 还支持 RRF，也就是 reciprocal rank fusion。

RRF 不关心原始分数，只关心每个 chunk 在两个列表中的排名。排名越靠前，贡献越大。

直观公式：

```text
rrf_score = 1 / (rrf_k + rank)
```

如果一个 chunk 在 BM25 和 dense 两个列表里都排得很靠前，它的最终分数就会更高。RRF 的好处是对不同检索器的分数尺度不敏感。

## 10. run_experiment.py 改了什么

Phase 1 时 runner 只支持 Vector RAG。Phase 2 后，`src/run_experiment.py` 被扩展为统一 retrieval runner。

现在它支持：

```yaml
retrieval:
  method: vector
```

```yaml
retrieval:
  method: bm25
```

```yaml
retrieval:
  method: hybrid
```

也就是说，不同方法共用同一套流程：

```text
load samples -> chunk contexts -> build retriever -> retrieve -> evaluate -> write outputs
```

这样保证三种方法的比较是公平的，因为数据、chunk 参数、top-k 和 metrics 都一致。

## 11. Phase 2 输出什么

和 Phase 1 一样，Phase 2 输出两个核心文件：

```text
results/phase2_hybrid_rag*/per_query_results.csv
results/phase2_hybrid_rag*/aggregate_metrics.json
```

BM25-only 输出：

```text
results/phase2_bm25_rag/per_query_results.csv
results/phase2_bm25_rag/aggregate_metrics.json
```

`per_query_results.csv` 保存每个问题的检索结果，包括：

- sample_id
- retriever
- question
- answer
- retrieved_doc_ids
- retrieved_chunk_ids
- retrieved_scores
- retrieval_latency_seconds
- recall@k / mrr@k / hit_rate@k

`aggregate_metrics.json` 保存整体平均指标。

## 12. 当前真实实验配置

BM25 配置：

```text
configs/phase2_bm25_rag.yaml
```

Hybrid 配置：

```text
configs/phase2_hybrid_rag_sentence_transformer.yaml
```

共同设置：

```text
Dataset: HotpotQA validation
Sample size: 100
Chunk size: 64
Overlap: 8
Top-k: 5
Metrics: Recall@1/3/5, MRR@1/3/5, Hit Rate@1/3/5
```

Hybrid 使用：

```text
Embedding: sentence-transformers/all-MiniLM-L6-v2
Fusion: weighted
BM25 weight: 0.5
Dense weight: 0.5
Candidate k: 20
```

## 13. 当前实验结果

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit Rate@5 | Avg Retrieval Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector RAG | 0.375 | 0.650 | 0.740 | 0.8275 | 0.950 | previous Phase 1 run |
| BM25 | 0.355 | 0.595 | 0.710 | 0.8200 | 0.990 | 0.0046s |
| Hybrid RAG | 0.410 | 0.690 | 0.785 | 0.8908 | 0.980 | 0.0158s |

## 14. 结果怎么理解

`Hybrid Recall@5 = 0.785`，高于 Vector RAG 的 0.74，也高于 BM25 的 0.71。这说明混合检索在 top-5 中平均找回了更多 gold supporting documents。

`Hybrid MRR@5 = 0.8908`，高于 Vector RAG 的 0.8275 和 BM25 的 0.82。这说明 Hybrid 不只是找得更多，第一条相关证据也更靠前。

`BM25 Hit Rate@5 = 0.99` 很高，说明 BM25 很容易在 top-5 中至少碰到一个相关文档。但它的 Recall@5 只有 0.71，说明它对多跳证据的完整覆盖不如 Hybrid。

Hybrid 的平均检索延迟是 0.0158s，高于 BM25 的 0.0046s，因为 Hybrid 同时执行 BM25 和 dense retrieval，再做融合。这就是效果和效率之间的 trade-off。

## 15. Phase 2 的价值

Phase 2 的价值在于：

- 补齐 lexical baseline，不再只和 Vector RAG 比。
- 建立一个更强的 Hybrid baseline，避免后续图增强方法只超过弱 baseline。
- 统一了 retriever 接口，让后续 GraphRAG / LightRAG 可以直接接入同一评估框架。
- 记录了检索质量和延迟，开始形成“效果 vs 成本”的对比思路。

## 16. 简历或面试可以怎么说

可以这样表述：

> 第二阶段我在 Vector RAG baseline 的基础上加入 BM25 关键词检索，并实现 Hybrid RAG。BM25 部分会对每个 chunk 统计 term frequency、document frequency 和 document length，用 Okapi BM25 分数返回 lexical top-k。Hybrid 部分同时执行 BM25 和 dense vector retrieval，然后对两路分数做归一化，通过 weighted score fusion 或 reciprocal rank fusion 合并结果。为了保证公平比较，我把 runner 改成统一 retrieval pipeline，让 Vector、BM25、Hybrid 使用同一批 HotpotQA 样本、同一切块参数、同一 top-k 和同一 Recall@k / MRR / Hit Rate 指标。实验显示，在 100 条 HotpotQA validation 样本上，Hybrid RAG 的 Recall@5 达到 0.785，高于 Vector RAG 的 0.74 和 BM25 的 0.71，说明关键词匹配和语义检索在多跳证据召回上具有互补性。
