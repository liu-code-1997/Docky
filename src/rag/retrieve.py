"""Retrieve 链路:问题 → 向量 → 检索 Top-K。

只依赖 Embedder / VectorStore 接口,不碰具体实现(Ollama/Qdrant)。
不设相似度阈值(M2 决策):总是返回 Top-K,靠 generate 阶段的 prompt 兜底拒答。
"""
from collections.abc import Callable

from rag.interfaces import Embedder, VectorStore, QueryRewriter, Reranker
from rag.models import RetrievedChunk
from rag.multi_query import rrf_fuse
from rag.sparse import encode_sparse


def retrieve(question: str, embedder: Embedder, store: VectorStore,
             top_k: int, library: str | None = None,
             rewriter: QueryRewriter | None = None,
             reranker: Reranker | None = None,
             rerank_factor: int = 5,
             query_prefix: str = "",
             hybrid: bool = False,
             hybrid_prefetch_factor: int = 5,
             query_expander: Callable[[str], list[str]] | None = None,
             ) -> list[RetrievedChunk]:
    """把问题向量化后,去向量库检索最相近的 top_k 块。

    - rewriter(M5②):非 None 时先改写查询再向量化,缓解跨语言检索。
    - reranker(M5③):非 None 时先召回 top_k×rerank_factor 个候选,再重排取前 top_k。
    - query_prefix(M8):在向量化前拼接到查询文本,默认空串。
    - hybrid(M9):True 时使用混合检索(稀疏+密集),False 时仅密集检索。
    - hybrid_prefetch_factor(M9):每路 Prefetch 召回 top_k×factor 个候选,默认 5。
    - query_expander(M11):非 None 时对每个变体各自检索后 RRF 融合;None 时单查询。
    """
    recall_k = top_k * rerank_factor if reranker is not None else top_k

    def _retrieve_one(q: str) -> list[RetrievedChunk]:
        query = rewriter.rewrite(q) if rewriter is not None else q
        qv = embedder.embed_one(query_prefix + query)
        if hybrid:
            return store.hybrid_search(qv, encode_sparse(query),
                                       top_k=recall_k, library=library,
                                       prefetch_factor=hybrid_prefetch_factor)
        return store.search(qv, top_k=recall_k, library=library)

    if query_expander is None:
        hits = _retrieve_one(question)
    else:
        lists = [_retrieve_one(q) for q in query_expander(question)]
        hits = rrf_fuse(lists, recall_k) if len(lists) > 1 else (lists[0] if lists else [])

    if reranker is not None:
        return reranker.rerank(question, hits, top_k=top_k)
    return hits
