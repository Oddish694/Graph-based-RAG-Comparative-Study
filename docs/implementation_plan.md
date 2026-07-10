# 实施计划 / Implementation Plan

## 总体目标

构建一个可复现的 RAG 对比实验框架，用于比较 Vector RAG、Hybrid RAG、GraphRAG-style RAG、LightRAG 以及改进版 LightRAG 在多跳问答任务上的效果。

框架采用模块化设计：数据加载、文本切块、索引构建、检索、生成、评估和结果报告相互解耦。实验通过 YAML 配置驱动，避免每次修改代码才能改变实验参数。

## 技术栈

- Python
- Hugging Face datasets
- sentence-transformers
- numpy / pandas
- PyYAML
- scikit-learn
- FAISS 或 Chroma，后续扩展
- rank-bm25，后续扩展
- NetworkX，后续用于图结构
- matplotlib / seaborn，后续用于图表
- LightRAG，后续集成

## 目录结构

```text
src/
├── datasets/
│   ├── hotpotqa_loader.py
│   └── schema.py
├── chunking/
│   ├── fixed_chunker.py
│   └── recursive_chunker.py
├── indexing/
│   └── vector_index.py
├── retrieval/
│   ├── naive_rag.py
│   └── vector_rag.py
├── evaluation/
│   └── retrieval_metrics.py
└── run_experiment.py
```

后续会继续扩展：

```text
src/indexing/bm25_index.py
src/indexing/graph_index.py
src/retrieval/hybrid_rag.py
src/retrieval/graph_rag_style.py
src/retrieval/lightrag_runner.py
src/enhancement/reranker.py
src/enhancement/query_rewrite.py
src/reporting/tables.py
src/reporting/plots.py
```

## Phase 1：最小可复现实验闭环

目标：先实现 HotpotQA 小样本上的 Vector RAG baseline，跑通从数据到指标的完整链路。

### Task 1：Dataset Schema and HotpotQA Loader

已完成内容：

- 定义 `QASample` dataclass
- 从 Hugging Face `hotpotqa/hotpot_qa` 加载 validation split
- 支持本地 JSONL 缓存
- 将原始 HotpotQA 格式转换成统一 schema
- 保留 question、answer、contexts、supporting_facts
- 添加 loader 单元测试

意义：统一数据入口，保证后续所有 RAG 方法使用同一批样本和同一套 gold evidence。

### Task 2：Chunking Module

已完成内容：

- 实现 fixed-size chunking
- 支持 chunk size 和 overlap
- 每个 chunk 保留 sample_id、doc_id、title、chunk_id、text 等元数据
- 添加 chunk 边界和非空文本测试

意义：把长 context document 切成适合 embedding 和 retrieval 的短文本块，同时保留评估所需的来源信息。

### Task 3：Vector RAG Baseline

已完成内容：

- 实现 hashing embedding，用于离线 smoke test
- 实现 sentence-transformers embedding adapter
- 实现本地向量索引 `VectorIndex`
- 实现 `VectorRAGRetriever`
- 返回 top-k chunks、score、doc_id、chunk_id 和 text
- 添加检索格式测试

意义：建立普通 dense retrieval baseline，作为后续 GraphRAG / LightRAG 的对照组。

### Task 4：Retrieval Metrics

已完成内容：

- 实现 Recall@k
- 实现 MRR@k
- 实现 Hit Rate@k
- 支持多组 k 值，例如 1、3、5
- 添加 synthetic examples 测试

意义：自动衡量检索器能否找回 HotpotQA 标注的 supporting facts。

### Task 5：Experiment Runner

已完成内容：

- 读取 YAML 实验配置
- 运行 dataset loading、chunking、indexing、retrieval、evaluation
- 输出 per-query CSV
- 输出 aggregate metrics JSON
- 添加 runner 测试
- 提供 PowerShell 入口脚本

意义：把模块组合成一键可复现实验流程。

## Phase 1 当前结果

已完成真实实验：

- Dataset：HotpotQA validation
- Sample size：100
- Embedding：sentence-transformers/all-MiniLM-L6-v2
- Retriever：Vector RAG
- Top-k：5

结果：

| Metric | Value |
| --- | --- |
| Recall@1 | 0.375 |
| Recall@3 | 0.65 |
| Recall@5 | 0.74 |
| MRR@1 | 0.75 |
| MRR@3 | 0.82 |
| MRR@5 | 0.8275 |
| Hit Rate@1 | 0.75 |
| Hit Rate@3 | 0.92 |
| Hit Rate@5 | 0.95 |

解释：Vector RAG 通常可以命中至少一个相关证据，但在多跳问答中还不能完全找齐所有 supporting facts，因此后续图增强方法有明确的改进空间。

## Phase 2：LightRAG 复现与对比

目标：在相同 HotpotQA subset 上运行 LightRAG，并与 Phase 1 的 Vector RAG baseline 对比。

计划任务：

- 定义 LightRAG 输入适配器
- 将 QASample contexts 转成 LightRAG 需要的 document 格式
- 固定 query mode 和 top-k 参数
- 捕获 retrieved contexts、answer、latency 和 token usage
- 与 Vector RAG 生成 side-by-side metric table

## Phase 3：消融实验

目标：系统分析组件对结果的影响。

计划任务：

- chunk size 消融
- top-k 消融
- embedding model 消融
- retriever 类型消融
- reranker 消融
- query rewrite 消融

## Phase 4：轻量改进

目标：在已有 baseline 上加入低成本增强方法。

优先方向：

- reranker-based context selection
- query rewrite enhanced retrieval

评估重点：质量提升是否值得额外延迟。

## Phase 5：结果报告

目标：整理项目成果。

计划产出：

- 主对比表格
- 消融实验图
- 延迟和成本分析
- 成功案例与失败案例
- README polish
- final_report.md
