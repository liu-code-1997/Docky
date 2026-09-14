from rag.condense import condense_question
from rag.models import Message


class _FakeLLM:
    def __init__(self, reply): self.reply = reply; self.seen = None
    def generate(self, prompt): self.seen = prompt; return self.reply


def test_no_history_returns_original_without_llm_call():
    llm = _FakeLLM("SHOULD NOT BE USED")
    out = condense_question(llm, "它能限定类型吗?", [])
    assert out == "它能限定类型吗?"
    assert llm.seen is None                      # 无历史不调 LLM


def test_with_history_uses_rewrite():
    hist = [Message(role="user", content="FastAPI 路径参数怎么声明?"),
            Message(role="assistant", content="用花括号 /items/{item_id}")]
    llm = _FakeLLM("路径参数可以限定类型吗?")
    out = condense_question(llm, "它能限定类型吗?", hist)
    assert out == "路径参数可以限定类型吗?"
    assert "路径参数怎么声明" in llm.seen           # 历史进了 prompt


def test_empty_reply_falls_back_to_original():
    hist = [Message(role="user", content="x"), Message(role="assistant", content="y")]
    out = condense_question(_FakeLLM("   "), "它呢?", hist)
    assert out == "它呢?"
