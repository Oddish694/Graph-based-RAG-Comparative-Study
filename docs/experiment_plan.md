# Experiment Plan

## Main Comparison

The first comparison will evaluate the following systems under the same dataset split and query set:

| System | Retrieval Unit | Retrieval Method | Graph Used | Reranker Used |
| --- | --- | --- | --- | --- |
| Naive RAG | text chunk | dense top-k | no | no |
| Vector RAG | text chunk | dense top-k | no | optional |
| Hybrid RAG | text chunk | BM25 + dense | no | optional |
| GraphRAG-style RAG | entity + relation + chunk | graph-aware retrieval | yes | optional |
| LightRAG | entity + relation + chunk | dual-level retrieval | yes | optional |
| Improved LightRAG | entity + relation + chunk | dual-level retrieval + enhancement | yes | yes |

## Dataset Plan

### HotpotQA Small

Use 500 to 2000 validation samples. Each sample should keep:

- question
- answer
- context documents
- supporting facts

This dataset is suitable because multi-hop questions can test whether graph-based retrieval helps recover evidence across multiple documents.

### 2WikiMultiHopQA Small

Use 500 to 2000 samples after the first HotpotQA loop works. This dataset is useful because it contains clearer reasoning paths.

### BEIR Subset

Use one or two retrieval-oriented subsets after QA evaluation is stable. This is useful for comparing retrievers without mixing too much generation noise.

## Evaluation Metrics

### Retrieval

| Metric | Meaning |
| --- | --- |
| Recall@k | Whether the gold evidence appears in top-k retrieved contexts. |
| Precision@k | How much of top-k retrieved contexts are relevant. |
| MRR | Whether the first relevant context ranks high. |
| NDCG | Whether highly relevant contexts appear earlier. |
| Hit Rate | Whether at least one supporting context is retrieved. |

### Generation

| Metric | Meaning |
| --- | --- |
| Exact Match | Whether generated answer exactly matches gold answer after normalization. |
| F1 | Token-level overlap between generated and gold answer. |
| Faithfulness | Whether generated answer is supported by retrieved contexts. |
| Answer Relevance | Whether generated answer directly addresses the question. |
| Context Precision | Whether retrieved contexts are useful for answering. |
| Context Recall | Whether retrieved contexts cover required evidence. |

### Efficiency

| Metric | Meaning |
| --- | --- |
| Index Time | Time to build vector index and graph index. |
| Query Latency | Average end-to-end query time. |
| Retrieval Latency | Time used before LLM generation. |
| Token Usage | Input and output tokens consumed by LLM calls. |
| Storage Size | Disk usage of vector index, graph index and cache. |

## Ablation Study

### Ablation 1: Chunk Size

Control all other settings and compare:

- 256 tokens
- 512 tokens
- 1024 tokens

Expected analysis:

- Smaller chunks may improve precision but lose cross-sentence context.
- Larger chunks may improve recall but introduce noise.
- Graph-based methods may be less sensitive to chunk size if entity relations are extracted well.

### Ablation 2: Embedding Model

Control chunk size and retriever, then compare:

- bge-small
- bge-base
- bge-m3
- text2vec

Expected analysis:

- Larger embedding models may improve recall at the cost of latency.
- Multilingual or Chinese-capable embedding models may behave differently on translated or Chinese subsets.

### Ablation 3: Retriever

Compare:

- BM25
- dense retriever
- hybrid retriever
- graph-aware retriever

Expected analysis:

- BM25 is strong for exact lexical matching.
- Dense retrieval is stronger for semantic paraphrase.
- Hybrid retrieval can improve robustness.
- Graph-aware retrieval may help multi-hop questions.

### Ablation 4: Top-k

Compare:

- top-3
- top-5
- top-10
- top-20

Expected analysis:

- Higher top-k may improve evidence recall.
- Higher top-k may hurt generation faithfulness by adding irrelevant context.
- Reranker may reduce the negative effect of large top-k.

### Ablation 5: Reranker

Compare:

- no reranker
- bge-reranker
- cross-encoder reranker

Expected analysis:

- Reranker should improve context precision.
- Reranker adds latency, so the project should report both quality gain and runtime cost.

### Ablation 6: Query Enhancement

Compare:

- original query
- query rewrite
- HyDE

Expected analysis:

- Query rewrite may improve clarity and retrieval recall.
- HyDE may help semantic retrieval but can introduce hallucinated assumptions.
- Query enhancement must be evaluated with latency and faithfulness together.

## Result Presentation

The final README should contain:

- main comparison table
- retrieval metric bar charts
- latency comparison chart
- ablation line charts
- representative success cases
- representative failure cases
- short analysis explaining why each method works or fails

