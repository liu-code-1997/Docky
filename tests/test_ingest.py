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


def test_ingest_noise_markers_reach_loader(tmp_path: Path):
    """noise_markers passed to ingest_directory are forwarded to load_chunks_from_dir."""
    lib = tmp_path / "lib"
    lib.mkdir()
    # Write content where a section heading matches our noise marker
    (lib / "index.md").write_text(
        "# Good Section\n\n" + "Useful content. " * 50 + "\n\n"
        "# sponsor\n\nNoise content. " * 30 + "\n\n"
        "# Another Good\n\n" + "More useful content. " * 50,
        encoding="utf-8"
    )

    class _CapLoader:
        called_with: list = []

    original_load = __import__("rag.loader", fromlist=["load_chunks_from_dir"]).load_chunks_from_dir

    import rag.ingest as _ingest_mod

    captured = {}

    original = _ingest_mod.load_chunks_from_dir

    def _spy(*args, **kwargs):
        captured["noise_markers"] = kwargs.get("noise_markers")
        return original(*args, **kwargs)

    _ingest_mod.load_chunks_from_dir = _spy
    try:
        store = QdrantStore(location=":memory:", collection_name="test")
        ingest_directory(
            docs_dir=tmp_path,
            embedder=FakeEmbedder(),
            store=store,
            chunk_size=200,
            overlap=50,
            vector_size=3,
            strategy="markdown",
            noise_markers=["sponsor"],
        )
    finally:
        _ingest_mod.load_chunks_from_dir = original

    assert captured.get("noise_markers") == ["sponsor"]


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
