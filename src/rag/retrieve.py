"""Retrieve 链路:问题 → 向量 → 检索 Top-K。

只依赖 Embedder / VectorStore 接口,不碰具体实现(Ollama/Qdrant)。
不设相似度阈值(M2 决策):总是返回 Top-K,靠 generate 阶段的 prompt 兜底拒答。
"""
from rag.interfaces import Embedder, VectorStore, QueryRewriter
from rag.models import RetrievedChunk


def retrieve(question: str, embedder: Embedder, store: VectorStore,
             top_k: int, library: str | None = None,
             rewriter: QueryRewriter | None = None) -> list[RetrievedChunk]:
    """把问题向量化后,去向量库检索最相近的 top_k 块。

    传入 rewriter(M5②)时,先改写查询再向量化,缓解跨语言检索。
    """
    query = rewriter.rewrite(question) if rewriter is not None else question
    query_vector = embedder.embed_one(query)
    return store.search(query_vector, top_k=top_k, library=library)
