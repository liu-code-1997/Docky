"""Retrieve 链路:问题 → 向量 → 检索 Top-K。

只依赖 Embedder / VectorStore 接口,不碰具体实现(Ollama/Qdrant)。
不设相似度阈值(M2 决策):总是返回 Top-K,靠 generate 阶段的 prompt 兜底拒答。
"""
from rag.interfaces import Embedder, VectorStore, QueryRewriter, Reranker
from rag.models import RetrievedChunk


def retrieve(question: str, embedder: Embedder, store: VectorStore,
             top_k: int, library: str | None = None,
             rewriter: QueryRewriter | None = None,
             reranker: Reranker | None = None,
             rerank_factor: int = 5) -> list[RetrievedChunk]:
    """把问题向量化后,去向量库检索最相近的 top_k 块。

    - rewriter(M5②):非 None 时先改写查询再向量化,缓解跨语言检索。
    - reranker(M5③):非 None 时先召回 top_k×rerank_factor 个候选,再重排取前 top_k。
    """
    query = rewriter.rewrite(question) if rewriter is not None else question
    query_vector = embedder.embed_one(query)

    # 有重排器时多召回一些候选,交给重排器筛选
    recall_k = top_k * rerank_factor if reranker is not None else top_k
    hits = store.search(query_vector, top_k=recall_k, library=library)

    if reranker is not None:
        return reranker.rerank(question, hits, top_k=top_k)
    return hits
