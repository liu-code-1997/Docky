"""Retrieve 链路:问题 → 向量 → 检索 Top-K。

只依赖 Embedder / VectorStore 接口,不碰具体实现(Ollama/Qdrant)。
不设相似度阈值(M2 决策):总是返回 Top-K,靠 generate 阶段的 prompt 兜底拒答。
"""
from rag.interfaces import Embedder, VectorStore
from rag.models import RetrievedChunk


def retrieve(question: str, embedder: Embedder, store: VectorStore,
             top_k: int, library: str | None = None) -> list[RetrievedChunk]:
    """把问题向量化后,去向量库检索最相近的 top_k 块。"""
    query_vector = embedder.embed_one(question)
    return store.search(query_vector, top_k=top_k, library=library)
