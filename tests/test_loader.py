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


def test_markdown_strategy_filters_noise_and_keeps_headings(tmp_path: Path):
    lib = tmp_path / "fastapi"
    lib.mkdir()
    (lib / "doc.md").write_text(
        "# FastAPI\n\n简介正文。\n\n"
        "## Path Parameters\n\n路径参数用花括号声明。\n\n"
        "## About FastAPI Cloud\n\nFastAPI Cloud is the primary sponsor.\n",
        encoding="utf-8",
    )

    chunks = load_chunks_from_dir(tmp_path, chunk_size=800, overlap=0,
                                  strategy="markdown")

    texts = [c.text for c in chunks]
    # 噪声段被过滤
    assert not any("primary sponsor" in t for t in texts)
    # 正文块保留(按标题切分)
    assert any("花括号" in t for t in texts)
    # 元数据仍完整
    assert all(c.library == "fastapi" and c.source == "fastapi/doc.md" for c in chunks)
