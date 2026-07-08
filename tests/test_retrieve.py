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
