from rag.models import Chunk


def test_chunk_holds_text_and_metadata():
    c = Chunk(
        id="fastapi/index.md::0",
        text="FastAPI is a web framework.",
        source="fastapi/index.md",
        library="fastapi",
        chunk_index=0,
    )
    assert c.text.startswith("FastAPI")
    assert c.library == "fastapi"
    assert c.chunk_index == 0
