"""加载 docs/ 下的 Markdown,切分并组装成 Chunk 列表。

约定目录结构: docs/<library>/<...>.md
顶层子目录名即 library(用于元数据过滤)。
"""
from pathlib import Path
from rag.models import Chunk
from rag.chunking import chunk_text


def load_chunks_from_dir(docs_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    chunks: list[Chunk] = []

    for md_path in sorted(docs_dir.rglob("*.md")):
        rel = md_path.relative_to(docs_dir)
        # 顶层目录作为 library;若文件直接在 docs_dir 下,则 library 用 "root"
        library = rel.parts[0] if len(rel.parts) > 1 else "root"
        source = rel.as_posix()

        text = md_path.read_text(encoding="utf-8")
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=f"{source}::{i}",
                text=piece,
                source=source,
                library=library,
                chunk_index=i,
            ))
    return chunks
