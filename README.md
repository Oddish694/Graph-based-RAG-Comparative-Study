# Graph-based RAG Frameworks: Reproduction, Comparative Study and Ablation Analysis

本项目计划复现并比较多种 RAG 框架，包括 Naive RAG、Vector RAG、GraphRAG-style RAG 与 LightRAG。项目重点不是做一个简单知识库问答 demo，而是构建一个可复现实验框架，用统一数据集、统一指标和统一实验配置分析不同 RAG 组件对检索准确率、回答质量、系统延迟和计算成本的影响。

## Project Goals

- Reproduce core workflows of Naive RAG, Vector RAG, GraphRAG-style RAG and LightRAG.
- Build a unified evaluation pipeline for retrieval quality, answer quality and runtime efficiency.
- Conduct ablation studies on chunking, embedding models, retrievers, top-k, rerankers and query rewriting.
- Add a lightweight improvement such as reranker-based context selection or query rewrite enhanced retrieval.
- Produce reproducible scripts, experiment tables, visual charts and a technical report.

## Research Questions

1. Do graph-based RAG methods improve multi-hop question answering compared with vector-only RAG?
2. How do chunk size and chunking strategy affect retrieval recall and answer faithfulness?
3. Which component has stronger impact: embedding model, retriever type, top-k, reranker or LLM backbone?
4. Can a lightweight improvement such as reranking or query rewriting improve LightRAG-style retrieval without unacceptable latency cost?

## Planned Baselines

| Method | Description |
| --- | --- |
| Naive RAG | Direct chunk retrieval plus LLM generation. |
| Vector RAG | Dense embedding retrieval with FAISS or Chroma. |
| Hybrid RAG | BM25 plus dense retrieval, followed by score fusion. |
| GraphRAG-style RAG | Entity-relation extraction, graph construction and graph-aware retrieval. |
| LightRAG | LightRAG reproduction or integration with controlled configuration. |
| Improved LightRAG | LightRAG plus reranker, query rewrite or HyDE-style retrieval enhancement. |

## Planned Datasets

| Dataset | Reason |
| --- | --- |
| HotpotQA small subset | Multi-hop QA with supporting facts, suitable for graph-based retrieval evaluation. |
| 2WikiMultiHopQA small subset | Multi-hop reasoning path is more explicit, useful for testing graph retrieval. |
| BEIR subset | Standard information retrieval benchmark, useful for retriever comparison. |

The first milestone will use small subsets, such as 500 to 2000 samples, to keep the project lightweight and reproducible on a personal machine.

## Evaluation Metrics

Retrieval metrics:

- Recall@k
- Precision@k
- MRR
- NDCG
- Hit Rate

Generation metrics:

- Exact Match
- F1
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

System metrics:

- Index construction time
- Average query latency
- LLM token usage
- Embedding cost or local embedding runtime
- Storage size

## Ablation Dimensions

| Dimension | Candidate Values |
| --- | --- |
| Chunk size | 256, 512, 1024 tokens |
| Chunk strategy | fixed-size, sentence-based, recursive, semantic |
| Embedding model | bge-small, bge-base, bge-m3, text2vec |
| Retriever | BM25, dense, hybrid, graph-aware |
| Top-k | 3, 5, 10, 20 |
| Reranker | none, bge-reranker, cross-encoder |
| Query enhancement | none, query rewrite, HyDE |
| LLM backbone | Qwen, DeepSeek, GPT-compatible API, Llama-compatible local model |

## Repository Structure

```text
.
├── README.md
├── configs/
│   └── experiment_matrix.yaml
├── docs/
│   ├── experiment_plan.md
│   ├── implementation_plan.md
│   ├── project_brief.md
│   └── resume_notes.md
├── experiments/
├── results/
├── assets/
└── src/
```

## First Milestone

The first milestone targets a minimal but complete experiment loop:

1. Prepare a small HotpotQA subset.
2. Implement Naive RAG and Vector RAG baselines.
3. Run retrieval evaluation with Recall@k, MRR and NDCG.
4. Add LightRAG reproduction or controlled integration.
5. Compare baseline results and write the first experiment report.

