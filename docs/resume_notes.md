# 简历与面试说明 / Resume Notes

## 中文简历版本

图增强 RAG 框架复现、对比实验与消融分析

- 构建面向 HotpotQA 多跳问答任务的 RAG 对比实验框架，统一数据加载、文本切块、向量检索和评估流程，为 GraphRAG / LightRAG 对比实验提供可复现 baseline。
- 实现 Vector RAG baseline，支持 HotpotQA 数据标准化、JSONL 缓存、fixed-size chunking、sentence-transformers 向量编码、top-k dense retrieval 和 Recall@k / MRR / Hit Rate 指标评估。
- 在 100 条 HotpotQA validation 样本上完成第一阶段实验，取得 Recall@5 0.74、MRR@5 0.8275、Hit Rate@5 0.95，分析普通向量检索在多跳证据覆盖上的局限。
- 使用 YAML 配置驱动实验流程，输出 per-query CSV 和 aggregate metrics JSON，并通过单元测试保证 loader、chunker、retriever 和 metrics 模块稳定可复现。

## English Resume Version

Graph-based RAG Frameworks: Reproduction, Comparative Study and Ablation Analysis

- Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, unifying dataset loading, chunking, dense retrieval and retrieval evaluation for later GraphRAG / LightRAG comparison.
- Implemented a Vector RAG baseline with HotpotQA normalization, JSONL caching, fixed-size chunking, Sentence-Transformers embeddings, top-k dense retrieval and metrics including Recall@k, MRR and Hit Rate.
- Evaluated on 100 HotpotQA validation samples and achieved Recall@5 of 0.74, MRR@5 of 0.8275 and Hit Rate@5 of 0.95, revealing the limitation of vector-only retrieval in covering complete multi-hop evidence.
- Designed YAML-driven experiments with per-query CSV and aggregate JSON outputs, supported by unit tests for dataset loading, chunking, retrieval and metric computation.

## 面试讲法：第一阶段做了什么

第一阶段我先没有直接实现 GraphRAG，而是先做了一个 Vector RAG baseline。原因是如果要证明图增强方法有效，必须先有一个普通向量检索的对照组。

我使用 HotpotQA validation 数据集，因为它是多跳问答数据集，每条样本包含 question、answer、候选 context documents 和 supporting facts。supporting facts 可以告诉我们哪些文档或句子是真正支持答案的证据，因此很适合做 retrieval evaluation。

实现上，我先把 HotpotQA 原始数据统一转换成 QASample 格式，保留 question、answer、contexts 和 supporting_facts。然后把每个 context document 切成固定大小的 chunks，并保留 doc_id、title、chunk_id 等元数据。这样检索时使用 chunk text，评估时可以用 doc_id 和 gold supporting facts 对齐。

接着我用 sentence-transformers 把 chunks 和 question 编码成向量，通过向量相似度返回 top-k chunks。最后我计算 Recall@k、MRR 和 Hit Rate，分析 Vector RAG 在多跳问答中的证据找回能力。

实验结果显示，Vector RAG 在 top-5 下 Hit Rate 达到 0.95，但 Recall@5 是 0.74。这说明它大多数时候能命中至少一个相关证据，但仍然不能完全找齐多跳推理所需的所有证据。这为后续 GraphRAG / LightRAG 的改进提供了比较空间。

## 关键词解释

- Vector RAG：只基于 embedding 相似度做文本 chunk 检索的基础 RAG baseline。
- GraphRAG-style RAG：抽取实体和关系，构建图结构，再利用图关系辅助检索。
- LightRAG：一种轻量图增强 RAG 方法，试图在图检索效果和系统开销之间取得平衡。
- Recall@k：top-k 结果中找回了多少 gold evidence。
- MRR：第一个相关证据排名是否靠前。
- Hit Rate：top-k 里是否至少命中一个相关证据。
