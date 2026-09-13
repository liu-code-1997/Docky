from rag.models import Chunk
from rag.providers.qdrant_store import QdrantStore


def _store():
    s = QdrantStore(collection_name="t", location=":memory:")
    s.ensure_collection(vector_size=4)
    return s


def _chunk(cid, lib="l"):
    return Chunk(id=cid, text="t", source=f"{cid}.md", library=lib, chunk_index=0)


def test_named_upsert_and_dense_search():
    s = _store()
    s.upsert([_chunk("a"), _chunk("b")],
             [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([10, 20], [1.0, 2.0]), ([20, 30], [3.0, 1.0])])
    hits = s.search([1, 0, 0, 0], top_k=2)
    assert hits[0].chunk.source == "a.md"        # dense 最近的排第一
    assert s.count() == 2


def test_hybrid_search_rrf_returns_results():
    s = _store()
    s.upsert([_chunk("a"), _chunk("b")],
             [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([10, 20], [1.0, 2.0]), ([20, 30], [3.0, 1.0])])
    hits = s.hybrid_search([1, 0, 0, 0], ([20], [1.0]), top_k=2)
    assert {h.chunk.source for h in hits} == {"a.md", "b.md"}  # 两路融合都覆盖


def test_library_filter_still_applies():
    s = _store()
    s.upsert([_chunk("a", "x"), _chunk("b", "y")], [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([1], [1.0]), ([2], [1.0])])
    hits = s.search([1, 0, 0, 0], top_k=5, library="x")
    assert all(h.chunk.library == "x" for h in hits)


def test_upsert_and_count():
    s = _store()
    s.upsert([_chunk("c"), _chunk("d")],
             [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert s.count() == 2


def test_list_libraries_returns_unique_sorted_names():
    s = _store()
    s.upsert([_chunk("e", "qdrant"), _chunk("f", "fastapi"), _chunk("g", "fastapi")],
             [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    libs = s.list_libraries()
    assert libs == ["fastapi", "qdrant"]


def test_list_libraries_empty_when_no_data():
    s = _store()
    assert s.list_libraries() == []
