from pathlib import Path
from rag.loader import load_chunks_from_dir


def test_loads_markdown_and_builds_chunks(tmp_path: Path):
    lib = tmp_path / "fastapi"
    lib.mkdir()
    (lib / "index.md").write_text("FastAPI is great. " * 100, encoding="utf-8")

    chunks = load_chunks_from_dir(tmp_path, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    first = chunks[0]
    assert first.library == "fastapi"
    assert first.source == "fastapi/index.md"
    assert first.chunk_index == 0
    assert first.id == "fastapi/index.md::0"
    # id 全局唯一
    assert len({c.id for c in chunks}) == len(chunks)


def test_ignores_non_markdown_files(tmp_path: Path):
    lib = tmp_path / "fastapi"
    lib.mkdir()
    (lib / "a.md").write_text("hello world", encoding="utf-8")
    (lib / "b.txt").write_text("ignore me", encoding="utf-8")

    chunks = load_chunks_from_dir(tmp_path, chunk_size=200, overlap=50)
    assert all(c.source.endswith(".md") for c in chunks)
    assert len(chunks) == 1
