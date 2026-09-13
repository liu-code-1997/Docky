"""加载 docs/ 下的多格式文件,切分并组装成 Chunk 列表。

约定目录结构: docs/<library>/<...>.<suffix>
顶层子目录名即 library(用于元数据过滤)。
支持的后缀通过 rag.extractors.supported_suffixes() 定义。
"""
from pathlib import Path
from rag.models import Chunk
from rag.chunking import chunk_text, chunk_markdown, _DEFAULT_NOISE_MARKERS
from rag.extractors import extract_text, supported_suffixes


def load_chunks_from_dir(docs_dir: Path, chunk_size: int, overlap: int,
                         strategy: str = "char",
                         noise_markers: tuple[str, ...] | list[str] = _DEFAULT_NOISE_MARKERS) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    supported = supported_suffixes()
    chunks: list[Chunk] = []

    paths = sorted(p for p in docs_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in supported)
    for path in paths:
        rel = path.relative_to(docs_dir)
        # 顶层目录作为 library;若文件直接在 docs_dir 下,则 library 用 "root"
        library = rel.parts[0] if len(rel.parts) > 1 else "root"
        source = rel.as_posix()

        try:
            text = extract_text(path)
        except Exception:
            continue          # 坏/加密/损坏文件:跳过,不中断整批
        if not text.strip():
            continue
        if path.suffix.lower() == ".md" and strategy == "markdown":
            pieces = chunk_markdown(text, chunk_size=chunk_size, overlap=overlap,
                                    noise_markers=noise_markers)
        else:
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
