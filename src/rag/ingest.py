"""Ingest 编排:加载 → 切分 → 向量化 → 入库。"""
from pathlib import Path
from rag.interfaces import Embedder, VectorStore
from rag.loader import load_chunks_from_dir


def ingest_directory(docs_dir: Path, embedder: Embedder, store: VectorStore,
                     chunk_size: int, overlap: int, vector_size: int,
                     strategy: str = "char", doc_prefix: str = "") -> int:
    """把 docs_dir 下的 Markdown 灌入向量库,返回入库的块数。

    - doc_prefix(M8):在向量化前拼接到每个块的文本,默认空串。
    """
    chunks = load_chunks_from_dir(docs_dir, chunk_size=chunk_size, overlap=overlap,
                                  strategy=strategy)
    if not chunks:
        return 0

    store.ensure_collection(vector_size=vector_size)
    vectors = embedder.embed([doc_prefix + c.text for c in chunks])
    store.upsert(chunks, vectors)
    return len(chunks)
