# Graph-based RAG Comparative Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible experiment framework for comparing Naive RAG, Vector RAG, GraphRAG-style RAG and LightRAG with controlled ablation studies.

**Architecture:** The project is organized around independent modules for dataset loading, chunking, indexing, retrieval, generation, evaluation and reporting. Each experiment is driven by YAML configuration so that component comparisons can be repeated without changing code.

**Tech Stack:** Python, Hugging Face datasets, sentence-transformers, FAISS or Chroma, rank-bm25, NetworkX, pandas, matplotlib, seaborn, RAGAS-compatible metrics, LightRAG integration.

---

## File Structure

```text
src/
├── datasets/
│   ├── hotpotqa_loader.py
│   └── schema.py
├── chunking/
│   ├── fixed_chunker.py
│   ├── recursive_chunker.py
│   └── semantic_chunker.py
├── indexing/
│   ├── vector_index.py
│   ├── bm25_index.py
│   └── graph_index.py
├── retrieval/
│   ├── naive_rag.py
│   ├── vector_rag.py
│   ├── hybrid_rag.py
│   ├── graph_rag_style.py
│   └── lightrag_runner.py
├── enhancement/
│   ├── reranker.py
│   ├── query_rewrite.py
│   └── hyde.py
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── answer_metrics.py
│   └── efficiency_metrics.py
├── reporting/
│   ├── tables.py
│   └── plots.py
└── run_experiment.py
```

## Phase 1: Minimal Reproducible Loop

### Task 1: Dataset Schema and HotpotQA Loader

**Files:**

- Create: `src/datasets/schema.py`
- Create: `src/datasets/hotpotqa_loader.py`
- Create: `tests/test_hotpotqa_loader.py`

- [ ] **Step 1: Define a normalized QA sample schema**

Implement a `QASample` dataclass with fields for `sample_id`, `question`, `answer`, `contexts` and `supporting_facts`.

- [ ] **Step 2: Load a small HotpotQA split**

Use Hugging Face datasets to load a fixed number of validation samples and convert them to `QASample`.

- [ ] **Step 3: Add loader tests**

Test that every loaded sample has a non-empty question, answer and context list.

- [ ] **Step 4: Save a cached JSONL subset**

Save the selected subset to `data/processed/hotpotqa_small.jsonl` so later experiments do not depend on repeated downloads.

### Task 2: Chunking Module

**Files:**

- Create: `src/chunking/fixed_chunker.py`
- Create: `src/chunking/recursive_chunker.py`
- Create: `tests/test_chunking.py`

- [ ] **Step 1: Implement fixed-size chunking**

Split context documents into token-like windows with configurable chunk size and overlap.

- [ ] **Step 2: Implement recursive chunking**

Split by paragraph, sentence and fallback character length.

- [ ] **Step 3: Test chunk boundaries**

Verify that chunk IDs are stable and chunk text is never empty.

### Task 3: Vector RAG Baseline

**Files:**

- Create: `src/indexing/vector_index.py`
- Create: `src/retrieval/vector_rag.py`
- Create: `tests/test_vector_retrieval.py`

- [ ] **Step 1: Wrap sentence-transformers embeddings**

Expose an embedding interface that accepts a list of texts and returns vectors.

- [ ] **Step 2: Build a local vector index**

Use FAISS or Chroma to store chunk embeddings and metadata.

- [ ] **Step 3: Retrieve top-k contexts**

Return ranked contexts with scores, document IDs and chunk IDs.

- [ ] **Step 4: Test retrieval output format**

Verify that retrieval always returns at most `k` results and preserves metadata.

### Task 4: Retrieval Metrics

**Files:**

- Create: `src/evaluation/retrieval_metrics.py`
- Create: `tests/test_retrieval_metrics.py`

- [ ] **Step 1: Implement Recall@k**

Compare retrieved chunk metadata with gold supporting facts.

- [ ] **Step 2: Implement MRR**

Compute reciprocal rank of the first relevant retrieved context.

- [ ] **Step 3: Implement Hit Rate**

Return 1 when at least one supporting context appears in top-k.

- [ ] **Step 4: Test metrics with synthetic examples**

Use small hand-written retrieved lists and gold labels to verify exact values.

### Task 5: First Experiment Runner

**Files:**

- Create: `src/run_experiment.py`
- Create: `configs/phase1_vector_rag.yaml`
- Create: `experiments/run_phase1_vector_rag.ps1`

- [ ] **Step 1: Parse YAML experiment config**

Load dataset path, chunk settings, embedding model, retriever settings and output path.

- [ ] **Step 2: Run dataset loading, chunking, indexing and retrieval**

Execute the full retrieval loop without LLM generation.

- [ ] **Step 3: Write results to CSV**

Save per-question retrieval results and aggregate metrics under `results/phase1_vector_rag/`.

- [ ] **Step 4: Run the phase 1 script**

Confirm that one small experiment produces a metrics file and a per-query result file.

## Phase 2: LightRAG Reproduction and Comparison

### Task 6: LightRAG Controlled Runner

**Files:**

- Create: `src/retrieval/lightrag_runner.py`
- Create: `configs/phase2_lightrag.yaml`

- [ ] **Step 1: Define LightRAG input adapter**

Convert normalized dataset contexts into the document format required by LightRAG.

- [ ] **Step 2: Define LightRAG query adapter**

Run each question through LightRAG using fixed query mode and top-k settings.

- [ ] **Step 3: Capture retrieved contexts**

Store retrieved evidence, answer text, latency and token usage when available.

- [ ] **Step 4: Compare LightRAG with Vector RAG**

Run both systems on the same subset and generate a side-by-side metric table.

## Phase 3: Ablation Studies

### Task 7: Chunk and Top-k Ablation

**Files:**

- Create: `configs/ablation_chunk_topk.yaml`
- Create: `experiments/run_ablation_chunk_topk.ps1`

- [ ] **Step 1: Generate experiment matrix**

Run chunk sizes 256, 512 and 1024 with top-k values 3, 5, 10 and 20.

- [ ] **Step 2: Aggregate results**

Save one row per configuration with retrieval metrics and latency.

- [ ] **Step 3: Plot trends**

Generate line charts for Recall@k, MRR and query latency.

### Task 8: Embedding and Retriever Ablation

**Files:**

- Create: `configs/ablation_embedding_retriever.yaml`
- Create: `experiments/run_ablation_embedding_retriever.ps1`

- [ ] **Step 1: Compare embedding models**

Run bge-small, bge-base, bge-m3 and text2vec under the same retriever.

- [ ] **Step 2: Compare retrievers**

Run BM25, dense and hybrid retrieval under the same dataset and chunk settings.

- [ ] **Step 3: Write analysis notes**

Summarize when lexical retrieval beats dense retrieval and when hybrid retrieval helps.

## Phase 4: Lightweight Improvement

### Task 9: Reranker Improvement

**Files:**

- Create: `src/enhancement/reranker.py`
- Create: `configs/improvement_reranker.yaml`

- [ ] **Step 1: Add reranker interface**

Accept query and candidate contexts, then return reranked contexts with new scores.

- [ ] **Step 2: Integrate reranker after initial retrieval**

Retrieve top-20 candidates, rerank them, then pass top-5 to the generator or evaluator.

- [ ] **Step 3: Compare quality and latency**

Report metric gains together with added reranking time.

### Task 10: Query Rewrite Improvement

**Files:**

- Create: `src/enhancement/query_rewrite.py`
- Create: `configs/improvement_query_rewrite.yaml`

- [ ] **Step 1: Add query rewrite prompt**

Rewrite questions into retrieval-friendly search queries while preserving original meaning.

- [ ] **Step 2: Run retrieval with original and rewritten queries**

Compare retrieval metrics and log representative improved and degraded cases.

- [ ] **Step 3: Analyze failure cases**

Identify cases where query rewriting introduces unsupported assumptions.

## Phase 5: Reporting

### Task 11: Tables and Plots

**Files:**

- Create: `src/reporting/tables.py`
- Create: `src/reporting/plots.py`
- Create: `docs/final_report.md`

- [ ] **Step 1: Generate main comparison table**

Include method, Recall@k, MRR, NDCG, latency and storage size.

- [ ] **Step 2: Generate ablation plots**

Create charts for chunk size, top-k, embedding model and reranker impact.

- [ ] **Step 3: Write final report**

Explain setup, results, observations, limitations and future work.

### Task 12: GitHub README Polish

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Add result tables**

Move the most important experiment table into README.

- [ ] **Step 2: Add charts**

Embed saved charts from `assets/`.

- [ ] **Step 3: Add reproduction commands**

Document environment setup, data preparation and experiment commands.

- [ ] **Step 4: Add resume-ready project summary**

Add a concise project summary suitable for recruiters and faculty interviewers.

## Self-review Notes

- The plan covers dataset loading, baselines, metrics, LightRAG comparison, ablations, improvement and reporting.
- The first phase is intentionally small and can produce a working result before graph-heavy components are added.
- GraphRAG-style implementation is kept after Vector RAG and LightRAG comparison to avoid early scope explosion.
- Reranker is prioritized over HyDE because it is easier to evaluate and less likely to introduce unsupported generated assumptions.

