from pathlib import Path
from rag.ingest import ingest_directory
from rag.providers.qdrant_store import QdrantStore
from rag.interfaces import Embedder


class FakeEmbedder(Embedder):
    """返回固定 3 维向量,避免依赖真实模型。"""
    def embed_one(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def test_ingest_directory_loads_and_stores(tmp_path: Path):
    lib = tmp_path / "fastapi"
    lib.mkdir()
    (lib / "index.md").write_text("FastAPI " * 300, encoding="utf-8")

    store = QdrantStore(location=":memory:", collection_name="test")
    embedder = FakeEmbedder()

    n = ingest_directory(
        docs_dir=tmp_path,
        embedder=embedder,
        store=store,
        chunk_size=200,
        overlap=50,
        vector_size=3,
    )

    assert n > 1               # 入库的块数
    assert store.count() == n  # Qdrant 里确实有这么多条


def test_ingest_prepends_doc_prefix(tmp_path: Path):
    """verify doc_prefix is prepended to each chunk text before embedding."""
    lib = tmp_path / "docs"
    lib.mkdir()
    (lib / "index.md").write_text("Hello " * 300, encoding="utf-8")

    class _CapEmbedder:
        def __init__(self):
            self.captured_texts = []
        def embed_one(self, text):
            return [float(len(text)), 1.0, 0.0]
        def embed(self, texts):
            self.captured_texts = texts
            return [self.embed_one(t) for t in texts]

    store = QdrantStore(location=":memory:", collection_name="test")
    embedder = _CapEmbedder()

    n = ingest_directory(
        docs_dir=tmp_path,
        embedder=embedder,
        store=store,
        chunk_size=200,
        overlap=50,
        vector_size=3,
        doc_prefix="search_document: "
    )

    assert n > 0
    assert all(t.startswith("search_document: ") for t in embedder.captured_texts)
