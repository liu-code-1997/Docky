"""Tests for ConversationalRag (M12 多轮编排)."""
from rag.conversation import ConversationalRag
from rag.session_store import InMemorySessionStore
from rag.models import Chunk, RetrievedChunk


def _rc(src, text="资料"):
    return RetrievedChunk(chunk=Chunk(id=src, text=text, source=src, library="l", chunk_index=0), score=1.0)


class _LLM:
    def __init__(self): self.gen_prompts = []
    def generate(self, prompt):
        self.gen_prompts.append(prompt)
        return "路径参数可以限定类型" if "独立问题" in prompt else "答案"


def test_first_turn_no_condense_then_second_turn_condenses():
    llm = _LLM(); store = InMemorySessionStore()
    seen = {}
    def retriever(query, library=None):
        seen["last_query"] = query
        return [_rc("fastapi/tutorial-path-params.md")]
    conv = ConversationalRag(retriever, llm, store, persona="p", refusal_text="拒",
                             history_turns=6, condense=True)

    a1 = conv.chat("s1", "FastAPI 路径参数怎么声明?")
    assert seen["last_query"] == "FastAPI 路径参数怎么声明?"     # 首轮不 condense
    assert a1.text == "答案"

    conv.chat("s1", "它能限定类型吗?")
    assert seen["last_query"] == "路径参数可以限定类型"           # 第二轮:检索用 condense 后的独立问题
    assert len(store.get_history("s1")) == 4                    # 两轮 user+assistant
