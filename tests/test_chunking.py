from rag.chunking import chunk_text


def test_short_text_yields_single_chunk():
    chunks = chunk_text("hello world", chunk_size=800, overlap=150)
    assert chunks == ["hello world"]


def test_long_text_is_split_into_multiple_chunks():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    # 每块不超过 chunk_size
    assert all(len(c) <= 800 for c in chunks)


def test_chunks_overlap_by_configured_amount():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    # 第二块的开头应与第一块的结尾有 overlap 个字符重叠
    tail = chunks[0][-150:]
    head = chunks[1][:150]
    assert tail == head


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
