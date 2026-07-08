from rag.models import Chunk
from rag.providers.qdrant_store import QdrantStore


def _chunk(i: int, library: str = "fastapi") -> Chunk:
    return Chunk(
        id=f"{library}/doc.md::{i}",
        text=f"text {i}",
        source=f"{library}/doc.md",
        library=library,
        chunk_index=i,
    )


def test_upsert_and_count():
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    chunks = [_chunk(0), _chunk(1)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, vectors)
    assert store.count() == 2


def test_search_returns_most_similar_first():
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    chunks = [_chunk(0), _chunk(1)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, vectors)

    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert results[0].chunk.chunk_index == 0  # 与第一个向量最相似
    assert results[0].score >= results[1].score


def test_search_can_filter_by_library():
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    chunks = [_chunk(0, "fastapi"), _chunk(1, "qdrant")]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, vectors)

    results = store.search([1.0, 1.0, 0.0], top_k=5, library="qdrant")
    assert len(results) == 1
    assert results[0].chunk.library == "qdrant"


def test_list_libraries_returns_unique_sorted_names():
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    # 3 块:fastapi 两块、qdrant 一块
    chunks = [_chunk(0, "qdrant"), _chunk(1, "fastapi"), _chunk(2, "fastapi")]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.upsert(chunks, vectors)

    libs = store.list_libraries()
    assert libs == ["fastapi", "qdrant"]  # 去重 + 排序


def test_list_libraries_empty_when_no_data():
    store = QdrantStore(location=":memory:", collection_name="test")
    store.ensure_collection(vector_size=3)
    assert store.list_libraries() == []
