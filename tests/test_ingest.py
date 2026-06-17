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
