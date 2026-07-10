# 实验计划 / Experiment Plan

## 总体目标

本项目计划在统一数据集、统一 query set、统一评估指标和统一配置文件下，对比不同 RAG 方法在多跳问答任务中的表现。

核心比较对象包括：

| 方法 | 检索单元 | 检索方式 | 是否使用图 | 是否使用 reranker |
| --- | --- | --- | --- | --- |
| Naive RAG | text chunk | dense top-k | 否 | 否 |
| Vector RAG | text chunk | dense top-k | 否 | 可选 |
| Hybrid RAG | text chunk | BM25 + dense fusion | 否 | 可选 |
| GraphRAG-style RAG | entity + relation + chunk | graph-aware retrieval | 是 | 可选 |
| LightRAG | entity + relation + chunk | dual-level retrieval | 是 | 可选 |
| Improved LightRAG | entity + relation + chunk | graph retrieval + enhancement | 是 | 是 |

## 数据集计划

### HotpotQA Small

用途：第一阶段主数据集。

HotpotQA 是英文多跳问答数据集，每条样本包含 question、answer、候选 context documents 和 supporting facts。supporting facts 标注了回答问题所需的证据文档和句子，因此适合评估检索器是否找回了正确证据。

计划使用 validation split 的 500 到 2000 条样本。当前 Phase 1 已经用 100 条 validation 样本跑通 Vector RAG baseline。

### 2WikiMultiHopQA Small

用途：第二个多跳问答数据集。

2WikiMultiHopQA 的多跳推理路径更显式，适合进一步测试 graph-aware retrieval 是否能利用实体关系提升证据召回。

### BEIR Subset

用途：标准检索 benchmark。

BEIR 更偏信息检索任务，可以用于比较 BM25、dense retrieval、hybrid retrieval 等 retriever 本身的能力，减少 generation 过程带来的干扰。

## 指标设计

### 检索指标 / Retrieval Metrics

| 指标 | 含义 | 为什么重要 |
| --- | --- | --- |
| Recall@k | top-k 中找回的 gold evidence 比例 | 衡量证据覆盖率，多跳问答尤其重要 |
| Precision@k | top-k 中相关证据所占比例 | 衡量检索结果是否干净 |
| MRR | 第一个相关证据排名的倒数 | 衡量正确证据是否靠前 |
| NDCG | 相关证据是否排在更高位置 | 衡量排序质量 |
| Hit Rate | top-k 中是否至少命中一个 supporting context | 衡量检索是否至少碰到相关证据 |

### 生成指标 / Generation Metrics

| 指标 | 含义 |
| --- | --- |
| Exact Match | 生成答案是否和标准答案完全一致 |
| F1 | 生成答案和标准答案的 token overlap |
| Faithfulness | 答案是否被检索上下文支持 |
| Answer Relevance | 答案是否直接回答问题 |
| Context Precision | 检索上下文中有多少对回答有用 |
| Context Recall | 检索上下文是否覆盖回答所需证据 |

### 系统指标 / Efficiency Metrics

| 指标 | 含义 |
| --- | --- |
| Index Time | 构建向量索引或图索引所需时间 |
| Retrieval Latency | 检索阶段耗时 |
| Query Latency | 单次 query 的端到端耗时 |
| Token Usage | LLM 输入和输出 token 数 |
| Storage Size | 索引、缓存和结果文件占用空间 |

## 消融实验计划 / Ablation Study

### Ablation 1：Chunk Size

对比：

- 256 tokens
- 512 tokens
- 1024 tokens

预期分析：小 chunk 可能提高精度但丢失上下文，大 chunk 可能提高召回但引入噪声。图增强方法如果实体关系抽取得好，理论上可能对 chunk size 不那么敏感。

### Ablation 2：Embedding Model

对比：

- bge-small
- bge-base
- bge-m3
- text2vec
- all-MiniLM-L6-v2

预期分析：更强的 embedding model 可能提高 Recall@k，但会增加索引和查询成本。

### Ablation 3：Retriever

对比：

- BM25
- dense retriever
- hybrid retriever
- graph-aware retriever

预期分析：BM25 适合精确词匹配，dense retriever 适合语义匹配，hybrid retriever 提高鲁棒性，graph-aware retriever 可能更适合多跳证据发现。

### Ablation 4：Top-k

对比：

- top-3
- top-5
- top-10
- top-20

预期分析：更大的 top-k 通常提高 recall，但也可能引入更多无关上下文，影响 generation faithfulness。

### Ablation 5：Reranker

对比：

- no reranker
- bge-reranker
- cross-encoder reranker

预期分析：reranker 可能提高 context precision，但会增加推理延迟，因此需要同时报告质量收益和时间成本。

### Ablation 6：Query Enhancement

对比：

- original query
- query rewrite
- HyDE

预期分析：query rewrite 可以把自然语言问题改写成更适合检索的查询，但也可能引入原问题没有的假设。HyDE 可能增强语义召回，但有 hallucination 风险。

## 结果展示计划

最终报告和 README 计划包含：

- 主方法对比表
- Recall@k / MRR / Hit Rate 柱状图
- 延迟对比图
- chunk size 和 top-k 消融折线图
- 成功案例分析
- 失败案例分析
- 对 GraphRAG / LightRAG 是否值得使用的结论
