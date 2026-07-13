# 简历与面试说明 / Resume Notes

## 中文简历版本

图增强 RAG 框架复现、对比实验与消融分析

- 构建面向 HotpotQA 多跳问答任务的可复现 RAG 对比实验框架，统一数据加载、JSONL 缓存、文本切块、检索器接口、评估指标和实验输出，为 GraphRAG / LightRAG 对比实验提供公平 baseline。
- 实现 Vector RAG、BM25 lexical retrieval 与 Hybrid RAG baseline，支持 sentence-transformers 向量编码、BM25 关键词检索、weighted score fusion 和 reciprocal rank fusion。
- 设计 fair baseline 实验配置，统一 Vector / BM25 / Hybrid 的数据集、sample size、seed、chunk size、overlap、top-k 和指标设置，避免方法差异被配置差异干扰。
- 扩展 retrieval metrics，实现 Recall@k、Precision@k、MRR@k、NDCG@k、Hit Rate@k、Evidence Hit Count@k、Full Evidence Recall@k 和 Retrieved Context Tokens，用于评估多跳证据覆盖、排序质量、噪声和上下文成本。
- 在 100 条 HotpotQA validation 样本上完成公平对比：Hybrid RAG 取得 Recall@5 0.790、Full Evidence Recall@5 0.590、NDCG@5 0.7582，优于 Vector RAG 和 BM25，为后续 GraphRAG-style / Improved LightRAG 提供强 baseline。
- 使用 YAML 配置驱动实验流程，输出 per-query CSV 和 aggregate metrics JSON，并通过单元测试覆盖 loader、chunker、BM25 index、hybrid retriever、runner 和 retrieval metrics。

## English Resume Version

Graph-based RAG Frameworks: Reproduction, Comparative Study and Ablation Analysis

- Built a reproducible RAG comparison framework for HotpotQA multi-hop QA, unifying dataset loading, JSONL caching, chunking, retriever interfaces, evaluation metrics and experiment outputs for controlled GraphRAG / LightRAG comparison.
- Implemented Vector RAG, BM25 lexical retrieval and Hybrid RAG baselines with Sentence-Transformers embeddings, BM25 scoring, weighted score fusion and reciprocal rank fusion.
- Designed fair baseline configurations that unify dataset, sample size, seed, chunk size, overlap, top-k and metrics across Vector / BM25 / Hybrid retrieval.
- Extended retrieval evaluation with Recall@k, Precision@k, MRR@k, NDCG@k, Hit Rate@k, Evidence Hit Count@k, Full Evidence Recall@k and Retrieved Context Tokens to measure multi-hop evidence coverage, ranking quality, noise and context cost.
- Evaluated on 100 HotpotQA validation samples. Hybrid RAG achieved Recall@5 of 0.790, Full Evidence Recall@5 of 0.590 and NDCG@5 of 0.7582, outperforming Vector RAG and BM25 under the same configuration.
- Designed YAML-driven experiments with per-query CSV and aggregate JSON outputs, backed by unit tests for data loading, chunking, indexing, retrieval and metric computation.

## 面试讲法：项目整体在做什么

这个项目不是做一个简单的知识库问答 demo，而是做一个可复现的 RAG 对比实验框架。核心目标是在同一套多跳问答数据、同一套切块方式、同一套 top-k 设置、同一套评估指标下，比较不同 RAG 检索方法到底谁更能找全证据。

我选择 HotpotQA 是因为它是多跳问答数据集，每条样本除了 question 和 answer，还提供 supporting facts。这些 supporting facts 可以作为 gold evidence，用来自动评估检索结果是否找到了真正支持答案的文档。

整个项目最终会比较 Vector RAG、BM25、Hybrid RAG、GraphRAG-style RAG、LightRAG 和改进版 LightRAG。当前已经完成 Vector RAG baseline、BM25 + Hybrid RAG baseline，以及 Phase 2.5 的公平基线和指标升级。

## 面试讲法：Phase 1 做了什么

第一阶段我先没有直接实现 GraphRAG，而是先做 Vector RAG baseline。原因是如果要证明图增强方法有效，必须先有普通向量检索的对照组。

实现上，我先从 Hugging Face 的 `hotpotqa/hotpot_qa` 加载 validation split，并把原始样本统一转成内部的 `QASample` 格式，保留 question、answer、contexts 和 supporting_facts。然后把每个 context document 切成固定大小 chunks，并保留 `doc_id`、`title`、`chunk_id` 等元数据。这样检索时使用 chunk text，评估时可以用 doc_id 和 gold supporting facts 对齐。

接着我用 sentence-transformers 把 chunks 和 question 编码成向量，通过向量相似度返回 top-k chunks。最后计算 Recall@k、MRR@k 和 Hit Rate@k，分析 Vector RAG 在多跳问答中的证据找回能力。

## 面试讲法：Phase 2 做了什么

第二阶段我加入了 BM25 和 Hybrid RAG。BM25 是关键词检索，它不使用 embedding，而是根据词频、逆文档频率和文档长度归一化计算每个 chunk 和 query 的相关性。它适合捕捉实体名、年份、专有名词和精确短语。

Hybrid RAG 则同时跑 BM25 和 dense vector retrieval。因为 BM25 分数和向量相似度分数不在同一个尺度上，所以我先分别归一化两路分数，再用 weighted score fusion 合并；同时也支持 reciprocal rank fusion，也就是只根据两个检索列表里的排名来融合。

为了公平比较，我把实验 runner 扩展成统一 retrieval pipeline：只要在 YAML 里修改 `retrieval.method`，就可以跑 `vector`、`bm25` 或 `hybrid`，其他数据、chunk、top-k 和 metrics 都保持一致。

## 面试讲法：Phase 2.5 做了什么

Phase 2.5 我主要做了两件事：统一公平基线，以及补齐更适合多跳问答的指标。

首先，我发现历史实验里 Vector RAG 和 BM25 / Hybrid 的 chunk 配置不完全一致，所以不能直接作为最终主结果。于是我新增了三份 fair baseline 配置，让 Vector、BM25 和 Hybrid 使用同一批 HotpotQA validation 样本、同一 seed、同一 `chunk_size=64`、同一 `overlap=8`、同一 `top_k=10` 和同一组 k values。

其次，我补充了 `Precision@k`、`NDCG@k`、`Evidence Hit Count@k`、`Full Evidence Recall@k` 和 `Retrieved Context Tokens`。其中 Full Evidence Recall 很关键，因为多跳问答不只是要命中一个相关文档，而是要找全一个问题所需的全部 supporting documents。

在 100 条 HotpotQA validation 样本上，Hybrid RAG 的 Recall@5 是 0.790，Full Evidence Recall@5 是 0.590，NDCG@5 是 0.7582，均高于 Vector RAG 和 BM25。这说明 Hybrid RAG 是后续 GraphRAG-style 方法需要超过的强 baseline。

## 关键词解释

- Vector RAG：只基于 embedding 相似度做文本 chunk 检索的基础 RAG baseline。
- BM25：基于关键词匹配、词频和逆文档频率的传统 lexical retrieval 方法。
- Hybrid RAG：同时使用 BM25 和 dense retrieval，并通过分数融合或排名融合合并结果。
- GraphRAG-style RAG：抽取实体和关系，构建图结构，再利用图关系辅助多跳证据检索。
- LightRAG：一种轻量图增强 RAG 方法，试图在图检索效果和系统开销之间取得平衡。
- Recall@k：top-k 结果中找回了多少 gold evidence。
- Precision@k：top-k 结果中有多少比例是 gold evidence。
- MRR@k：第一个相关证据在 top-k 中排得是否靠前。
- NDCG@k：相关证据整体排序质量。
- Hit Rate@k：top-k 里是否至少命中一个相关证据。
- Full Evidence Recall@k：top-k 里是否找全当前问题需要的全部 supporting evidence。
