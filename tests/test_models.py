from rag.models import Chunk, Answer, EvalSample


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


def test_eval_sample_multi_source():
    s = EvalSample(question="q", expected_sources=["a.md", "b.md"])
    assert s.expected_sources == ["a.md", "b.md"]


def test_eval_sample_negative_defaults_empty():
    s = EvalSample(question="q")
    assert s.expected_sources == []


def test_eval_sample_compat_old_single_source():
    s = EvalSample(question="q", expected_source="a.md")
    assert s.expected_sources == ["a.md"]


def test_eval_sample_compat_old_null_source():
    s = EvalSample(question="q", expected_source=None)
    assert s.expected_sources == []
