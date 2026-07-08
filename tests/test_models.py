from rag.models import Chunk, Answer


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


def test_answer_holds_text_and_sources():
    a = Answer(text="FastAPI 是一个 web 框架。", sources=["fastapi/index.md"])
    assert a.text.startswith("FastAPI")
    assert a.sources == ["fastapi/index.md"]


def test_answer_sources_defaults_to_empty_list():
    a = Answer(text="根据现有资料无法回答。")
    assert a.sources == []
