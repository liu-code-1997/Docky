"""把 docs/ 灌入 Qdrant。

用法: python scripts/ingest.py
依赖:Ollama 与 Qdrant 已启动(见 M0)。
"""
from pathlib import Path
from rag.config import get_settings
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.qdrant_store import QdrantStore
from rag.ingest import ingest_directory


def main() -> None:
    settings = get_settings()
    docs_dir = Path(__file__).resolve().parent.parent / "docs"

    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)

    # 先探测一次 embedding 维度,用于建集合
    probe = embedder.embed_one("probe")
    vector_size = len(probe)
    print(f"embedding 维度 = {vector_size}")

    store = QdrantStore(collection_name=settings.collection_name, url=settings.qdrant_url)

    n = ingest_directory(
        docs_dir=docs_dir,
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        vector_size=vector_size,
    )
    print(f"✅ 已入库 {n} 个块,Qdrant 当前共 {store.count()} 条。")


if __name__ == "__main__":
    main()
