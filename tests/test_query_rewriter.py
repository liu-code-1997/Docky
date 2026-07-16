from rag.interfaces import LLM
from rag.query_rewrite import LlmQueryRewriter


class _FakeLLM(LLM):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def test_rewrite_returns_llm_output_and_includes_original():
    llm = _FakeLLM("path parameter 路径参数 声明")
    rw = LlmQueryRewriter(llm)
    out = rw.rewrite("FastAPI 里路径参数怎么声明?")

    # 原问题被放进 prompt 让 LLM 改写
    assert "路径参数" in llm.last_prompt
    # 改写结果拼上原问题(保底:改写没帮上时原词仍在)
    assert "path parameter" in out
    assert "FastAPI 里路径参数怎么声明?" in out


def test_rewrite_falls_back_to_original_on_empty_reply():
    rw = LlmQueryRewriter(_FakeLLM("   "))
    out = rw.rewrite("查询参数怎么用?")
    # LLM 没给出有用改写时,至少返回原问题
    assert out.strip() == "查询参数怎么用?"


def test_rewrite_strips_whitespace():
    rw = LlmQueryRewriter(_FakeLLM("  query parameter  "))
    out = rw.rewrite("查询参数")
    assert "query parameter" in out
    assert out == out.strip()
