from rag.interfaces import Embedder
from rag.models import Chunk
from rag.providers.qdrant_store import QdrantStore
from rag.retrieve import retrieve


class FakeEmbedder(Embedder):
    """把问题映射成固定向量,使检索结果可预测。

    "fastapi" -> 偏向第一个向量;其它 -> 偏向第二个。
    """
    def embed_one(self, text: str) -> list[float]:
        if "fastapi" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def _seed_store() -> QdrantStore:
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    chunks = [
        Chunk(id="fastapi/a.md::0", text="FastAPI doc", source="fastapi/a.md",
              library="fastapi", chunk_index=0),
        Chunk(id="qdrant/b.md::0", text="Qdrant doc", source="qdrant/b.md",
              library="qdrant", chunk_index=0),
    ]
    store.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return store


def test_retrieve_embeds_question_and_returns_top_k():
    store = _seed_store()
    results = retrieve("what is fastapi", FakeEmbedder(), store, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.library == "fastapi"  # 问题向量最接近 fastapi 块


def test_retrieve_passes_library_filter_through():
    store = _seed_store()
    # 即便问题向量偏向 fastapi,限定 library=qdrant 也只能拿到 qdrant 的块
    results = retrieve("what is fastapi", FakeEmbedder(), store,
                       top_k=5, library="qdrant")
    assert len(results) == 1
    assert results[0].chunk.library == "qdrant"


class _FakeRewriter:
    """把任何问题改写成含 'fastapi' 的查询。"""
    def __init__(self):
        self.called_with = None

    def rewrite(self, question: str) -> str:
        self.called_with = question
        return "fastapi " + question


def test_retrieve_uses_rewriter_when_provided():
    store = _seed_store()
    rw = _FakeRewriter()
    # 原问题 "tell me about docs" 本会偏向第二个向量(qdrant);
    # 改写后含 fastapi -> 偏向 fastapi 块。证明改写生效。
    results = retrieve("tell me about docs", FakeEmbedder(), store,
                       top_k=1, rewriter=rw)
    assert rw.called_with == "tell me about docs"
    assert results[0].chunk.library == "fastapi"


def test_retrieve_without_rewriter_is_unchanged():
    store = _seed_store()
    # 不传 rewriter,行为与之前一致(baseline)
    results = retrieve("tell me about docs", FakeEmbedder(), store, top_k=1)
    assert results[0].chunk.library == "qdrant"


class _ReverseReranker:
    """把候选顺序反转,用于验证 rerank 确实被调用且作用于结果。"""
    def __init__(self):
        self.got_n = None

    def rerank(self, question, candidates, top_k):
        self.got_n = len(candidates)
        return list(reversed(candidates))[:top_k]


def test_retrieve_applies_reranker_and_recalls_more_candidates():
    store = _seed_store()  # 库里 2 条:fastapi 向量[1,0,0]、qdrant 向量[0,1,0]
    rr = _ReverseReranker()
    # top_k=1,factor=5 -> 先召回 min(5, 全部)=2 个候选,重排(反转)后取1
    results = retrieve("what is fastapi", FakeEmbedder(), store, top_k=1,
                       reranker=rr, rerank_factor=5)
    assert rr.got_n == 2          # 确实召回了多于 top_k 的候选
    assert len(results) == 1      # 最终截到 top_k
