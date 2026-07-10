# Phase 1 详细说明：HotpotQA 小样本 + Vector RAG Baseline

## 1. Phase 1 的目标

Phase 1 的目标是建立一个最小但完整的 Vector RAG 检索 baseline。

它要回答的问题是：

> 如果只使用普通向量检索，不引入图结构、不引入 reranker、不做 query rewrite，在 HotpotQA 多跳问答任务上能找回多少正确证据？

这个结果会作为后续 GraphRAG-style RAG 和 LightRAG 的对照组。只有先知道普通 Vector RAG 的水平，后面才能判断图增强方法是否真的有提升。

## 2. 数据来自哪里

数据来自 Hugging Face：

```text
hotpotqa/hotpot_qa
```

当前使用：

```text
config: distractor
split: validation
sample_size: 100
```

HotpotQA 是一个英文多跳问答数据集。它的特点是一个问题通常需要结合多个文档才能回答，因此适合测试 RAG 系统是否能找全多跳证据。

## 3. 原始 HotpotQA 样本是什么样

一条原始样本大致包含：

```python
{
    "id": "...",
    "question": "Where was the author of X born?",
    "answer": "London",
    "context": [
        ["Document A", ["sentence 0", "sentence 1"]],
        ["Document B", ["sentence 0", "sentence 1"]]
    ],
    "supporting_facts": [
        ["Document A", 0],
        ["Document B", 1]
    ]
}
```

字段含义：

- `question` 是问题。
- `answer` 是标准答案。
- `context` 是候选文档集合，每个文档由标题和句子列表组成。
- `supporting_facts` 是 gold evidence，标注了真正支持答案的文档和句子编号。

## 4. 为什么要做数据标准化

不同数据集的原始字段不一样。为了后续统一比较 Vector RAG、Hybrid RAG、GraphRAG 和 LightRAG，需要先把数据转成统一结构。

项目内部使用 `QASample`：

```python
QASample(
    sample_id="...",
    question="...",
    answer="...",
    contexts=[
        {
            "doc_id": "Document A",
            "title": "Document A",
            "text": "sentence 0 sentence 1"
        }
    ],
    supporting_facts=[
        {
            "doc_id": "Document A",
            "sent_id": 0
        }
    ]
)
```

这样做的好处：

- 后续所有 RAG 方法使用同一输入格式。
- 检索和评估模块不依赖 HotpotQA 原始格式。
- 之后接入 2WikiMultiHopQA 或 BEIR 时，只需要新增 loader，不需要重写检索和评估逻辑。

## 5. 处理后存成什么样

处理后的数据缓存为 JSONL：

```text
data/processed/hotpotqa_small.jsonl
```

JSONL 是一行一个 JSON 样本，适合流式读取，也方便复现实验。

缓存的意义：

- 避免每次实验重新下载数据。
- 固定小样本，减少随机性。
- 后续多个 retriever 可以在同一数据子集上公平比较。

## 6. 为什么要切块

RAG 通常不直接检索整篇文档，而是先把文档切成 chunks。

原因：

- 整篇文档太长时，embedding 会混合太多语义，相关细节可能被稀释。
- 多数问题只需要文档中的几句话。
- chunk 让检索粒度更细，能更准确定位证据。
- 后续传给 LLM 的上下文更短，可以减少 token 成本。
- chunk 保留 doc_id 和 chunk_id，方便和 supporting facts 对齐评估。

## 7. 当前怎么切块

当前实现的是 fixed-size chunking。

核心参数：

```text
chunk_size: 每个 chunk 包含多少 token-like words
overlap: 相邻 chunk 重叠多少 token-like words
```

例如：

```text
chunk_size = 128
overlap = 20
```

overlap 的作用是避免关键信息刚好被切断。如果一句证据跨越两个 chunk，重叠窗口可以提高它被完整保留的概率。

## 8. chunk 处理后是什么样

每个 chunk 大致是：

```python
{
    "sample_id": "...",
    "doc_id": "Document A",
    "title": "Document A",
    "chunk_id": "Document A::0",
    "chunk_index": 0,
    "text": "chunk text ..."
}
```

其中：

- `text` 用于 embedding 和检索。
- `doc_id` 用于和 supporting facts 对齐。
- `chunk_id` 用于定位具体 chunk。
- `sample_id` 用于追踪来源样本。

## 9. 向量化的目的

向量化是把文本转成 dense vector，让计算机可以用数值方式比较语义相似度。

流程：

```text
question text -> question vector
chunk text -> chunk vector
```

如果 question vector 和某个 chunk vector 的相似度高，就说明这个 chunk 可能包含回答问题所需的信息。

## 10. 当前支持的 embedding

当前支持两种 embedding：

1. `HashingEmbeddingModel`

用于 smoke test，不需要下载模型。它主要保证项目在离线环境下也能跑通完整流程。

2. `SentenceTransformerEmbeddingModel`

用于真实实验。当前配置使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

它比 hashing embedding 更能表达语义相似度，因此更适合作为 Vector RAG baseline。

## 11. 向量索引做了什么

向量索引 `VectorIndex` 做两件事。

第一，构建索引：

```text
所有 chunks -> embedding model -> chunk vectors -> 保存 vectors 和 metadata
```

第二，执行检索：

```text
question -> question vector -> 和所有 chunk vectors 计算相似度 -> 排序 -> 返回 top-k chunks
```

返回结果包含：

- `doc_id`
- `chunk_id`
- `title`
- `text`
- `score`

## 12. Vector RAG baseline 的含义

Vector RAG 是最基础的 dense retrieval RAG baseline。

它不使用：

- 图结构
- 实体关系
- reranker
- query rewrite
- HyDE
- graph-aware traversal

它只做：

```text
embedding similarity search
```

因此它适合作为后续增强方法的 baseline。

## 13. 如何评估检索结果

HotpotQA 提供 supporting facts，所以可以自动判断检索结果是否命中正确证据。

当前评估方式是 doc_id level 对齐：

```text
retrieved doc_id vs gold supporting_facts doc_id
```

如果 gold supporting facts 包含：

```text
Document A, Document B
```

而 top-5 检索结果包含：

```text
Document A, Document C, Document B
```

那么两个 gold documents 都被找回，Recall@5 = 1.0。

## 14. Recall@k 的含义

Recall@k 表示 top-k 结果中找回了多少 gold evidence。

公式直观理解：

```text
Recall@k = 找回的 gold evidence 数量 / 全部 gold evidence 数量
```

多跳问答中，Recall 很重要，因为问题往往需要多个证据。只找回一个证据可能不足以正确回答。

## 15. MRR 的含义

MRR 关注第一个相关证据排在第几名。

如果第一个相关证据排第 1，得分是 1.0。

如果排第 2，得分是 0.5。

如果排第 5，得分是 0.2。

MRR 越高，说明正确证据越靠前，后续 LLM 更容易优先看到有用上下文。

## 16. Hit Rate 的含义

Hit Rate@k 表示 top-k 中是否至少命中一个 supporting context。

它比 Recall 更宽松。

例如一个问题需要两个证据，只命中一个时：

```text
Hit Rate@5 = 1
Recall@5 = 0.5
```

所以 Hit Rate 高说明检索器经常能碰到相关证据，但不代表证据完整。

## 17. 当前实验结果

当前真实实验设置：

```text
Dataset: HotpotQA validation
Sample size: 100
Embedding: sentence-transformers/all-MiniLM-L6-v2
Retriever: Vector RAG
Top-k: 5
```

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

## 18. 结果说明了什么

`Hit Rate@5 = 0.95` 表示 95% 的问题在 top-5 中至少命中了一个正确证据。

`Recall@5 = 0.74` 表示 top-5 平均只能覆盖 74% 的 gold evidence，还不能完整找齐所有多跳证据。

`MRR@5 = 0.8275` 表示第一个正确证据通常排得比较靠前。

整体结论：Vector RAG 是一个不错的基础 baseline，但它对多跳证据的完整覆盖仍然有限。后续 GraphRAG / LightRAG 可以尝试通过实体关系和图结构提高证据覆盖率。

## 19. 第一阶段的价值

第一阶段的价值不是追求最高效果，而是建立一个公平、可复现、可量化的 baseline。

它为后续工作提供：

- 统一数据格式
- 可复现实验配置
- 检索 baseline
- 标准评估指标
- 第一组可解释实验结果

## 20. 面试表述

可以这样说：

> 第一阶段我实现了 HotpotQA 多跳问答上的 Vector RAG baseline。首先从 Hugging Face 加载 HotpotQA validation 数据，并把原始 question、answer、contexts 和 supporting facts 统一成 QASample 格式。然后对 context documents 做 fixed-size chunking，保留 doc_id 和 chunk_id 等元数据。接着使用 sentence-transformers 将 question 和 chunks 编码成向量，通过向量相似度返回 top-k chunks。最后将检索结果和 HotpotQA 标注的 supporting facts 对齐，计算 Recall@k、MRR 和 Hit Rate。实验结果显示 Vector RAG 在 top-5 下 Hit Rate 达到 0.95，但 Recall@5 为 0.74，说明普通向量检索通常能命中相关证据，但在多跳证据完整覆盖上仍有不足，因此后续引入 GraphRAG / LightRAG 做对比是有意义的。
