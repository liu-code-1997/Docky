from rag.interfaces import LLM
from rag.models import Chunk, RetrievedChunk
from rag.generate import build_prompt, answer


class FakeLLM(LLM):
    """记录收到的 prompt,返回固定答案。"""
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "这是答案。"


def _rc(source: str, text: str, score: float = 0.9) -> RetrievedChunk:
    lib = source.split("/")[0]
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::0", text=text, source=source,
                    library=lib, chunk_index=0),
        score=score,
    )


def test_build_prompt_includes_constraint_and_sources():
    chunks = [_rc("fastapi/a.md", "FastAPI 用 Depends 做依赖注入。"),
              _rc("fastapi/b.md", "路径参数用花括号声明。")]
    prompt = build_prompt("依赖注入怎么写?", chunks)

    # 含强约束:只依据资料 + 不知道就说不知道
    assert "无法回答" in prompt
    # 含问题
    assert "依赖注入怎么写?" in prompt
    # 含每块资料的原文与来源
    assert "FastAPI 用 Depends" in prompt
    assert "fastapi/a.md" in prompt
    assert "fastapi/b.md" in prompt


def test_build_prompt_handles_empty_chunks():
    prompt = build_prompt("随便问问", [])
    assert "无法回答" in prompt  # 即使没资料,约束仍在
    assert "随便问问" in prompt


def test_answer_calls_llm_and_collects_unique_sources():
    chunks = [_rc("fastapi/a.md", "x"),
              _rc("fastapi/a.md", "y"),   # 同一来源出现两次
              _rc("fastapi/b.md", "z")]
    llm = FakeLLM()
    result = answer("问题", chunks, llm)

    assert result.text == "这是答案。"
    # sources 去重,且保持首次出现顺序
    assert result.sources == ["fastapi/a.md", "fastapi/b.md"]
    # 确实把资料拼进了 prompt
    assert "问题" in llm.last_prompt


def test_answer_with_no_chunks_has_empty_sources():
    llm = FakeLLM()
    result = answer("问题", [], llm)
    assert result.sources == []
    assert result.text == "这是答案。"
