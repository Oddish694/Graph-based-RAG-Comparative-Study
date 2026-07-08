# Resume Notes

## 中文简历表达

图增强 RAG 框架复现与消融实验研究

- 复现并比较 Naive RAG、Vector RAG、GraphRAG-style RAG 与 LightRAG，在 HotpotQA、2WikiMultiHopQA 等公开数据集子集上构建统一评测流程。
- 围绕 chunk 策略、embedding 模型、retriever 类型、top-k、reranker 与 query rewrite 等因素开展消融实验，评估其对 Recall@k、MRR、NDCG、答案质量与查询延迟的影响。
- 引入 reranker/query rewrite 作为轻量改进策略，在复杂多跳问答场景下提升上下文召回率与答案忠实度，并通过实验图表分析性能收益与延迟成本。
- 整理可复现实验脚本、配置文件、结果表格和技术报告，形成面向开源展示的 GitHub 项目。

## English GitHub Summary

This project reproduces and compares Naive RAG, Vector RAG, GraphRAG-style RAG and LightRAG under a unified evaluation framework. It studies how chunking, embedding models, retrievers, top-k, rerankers and query rewriting affect retrieval quality, answer faithfulness, latency and cost on public multi-hop QA and retrieval datasets.

## Interview Talking Points

### Why graph-based RAG?

普通向量 RAG 更依赖 chunk 级别的语义相似度，对多跳问题和跨文档关系的建模较弱。Graph-based RAG 通过实体和关系组织知识，理论上更适合多跳推理和关系查询，但也会带来图构建成本、实体抽取误差和查询延迟问题。

### Why ablation study?

RAG 效果往往不是由单个模块决定的。Chunk 大小、embedding、retriever、top-k、reranker 和 LLM 都会影响最终效果。消融实验可以说明性能提升来自哪里，也能避免只展示一个 demo 而无法解释结果。

### Why reranker as improvement?

Reranker 能在召回较多候选上下文后重新排序，提高传入 LLM 的上下文质量。它的优点是容易插入现有 RAG 流程，评估方式清晰；缺点是会增加延迟，因此需要同时报告质量收益和时间成本。

### Why query rewrite?

Query rewrite 可以把用户问题改写成更适合检索的查询，尤其在问题包含省略、指代或复杂组合条件时可能提升召回。但它也可能引入原问题没有的假设，因此需要做失败案例分析。

## Short Version for Resume

复现并比较 Naive RAG、Vector RAG、GraphRAG-style RAG 与 LightRAG，基于 HotpotQA/2WikiMultiHopQA 构建统一评测流程；围绕 chunk、embedding、retriever、top-k、reranker 与 query rewrite 开展消融实验，分析各模块对检索准确率、答案忠实度和查询延迟的影响，并提出轻量 reranker/query rewrite 改进策略。

