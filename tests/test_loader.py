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


def test_ignores_unsupported_suffixes(tmp_path: Path):
    lib = tmp_path / "fastapi"
    lib.mkdir()
    (lib / "a.md").write_text("hello world", encoding="utf-8")
    (lib / "b.txt").write_text("text content", encoding="utf-8")
    (lib / "c.unknown").write_text("unsupported", encoding="utf-8")

    chunks = load_chunks_from_dir(tmp_path, chunk_size=200, overlap=50)
    srcs = {c.source for c in chunks}
    # Supported files (.md, .txt) should be loaded
    assert "fastapi/a.md" in srcs
    assert "fastapi/b.txt" in srcs
    # Unsupported files (.unknown) should be ignored
    assert "fastapi/c.unknown" not in srcs
    assert len(chunks) == 2


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


def test_loader_handles_mixed_formats(tmp_path):
    from rag.loader import load_chunks_from_dir
    lib = tmp_path / "lib"; lib.mkdir()
    (lib / "a.md").write_text("# H\nmarkdown body", encoding="utf-8")
    (lib / "b.txt").write_text("plain text body", encoding="utf-8")
    (lib / "c.html").write_text("<body><p>html body content</p></body>", encoding="utf-8")
    chunks = load_chunks_from_dir(tmp_path, chunk_size=800, overlap=0)
    srcs = {c.source for c in chunks}
    assert {"lib/a.md", "lib/b.txt", "lib/c.html"} <= srcs
    assert all(c.library == "lib" for c in chunks)
    assert any("html body content" in c.text for c in chunks)


def test_bad_file_does_not_abort_batch(tmp_path: Path, monkeypatch):
    """Verify that a corrupt/encrypted file is skipped and doesn't abort the entire batch."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.txt").write_text("good content a", encoding="utf-8")
    (lib / "b.txt").write_text("good content b", encoding="utf-8")

    # Monkeypatch extract_text to raise for b.txt but return text for a.txt
    original_extract = None

    def mock_extract_text(path):
        if "b.txt" in str(path):
            raise ValueError("Simulated corrupt/encrypted file")
        return f"content from {path.name}"

    import rag.loader
    monkeypatch.setattr(rag.loader, "extract_text", mock_extract_text)

    # Should not raise; should process a.txt and skip b.txt
    chunks = load_chunks_from_dir(tmp_path, chunk_size=200, overlap=50)
    srcs = {c.source for c in chunks}

    # a.txt should be loaded
    assert "lib/a.txt" in srcs
    # b.txt should be skipped (no chunks from it)
    assert "lib/b.txt" not in srcs
    # At least one chunk from a.txt
    assert len(chunks) >= 1
    assert any("a.txt" in c.source for c in chunks)
