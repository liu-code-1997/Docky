from rag.interfaces import LLM
from rag.models import Chunk, RetrievedChunk
from rag.rerank import LlmReranker


class _ScriptLLM(LLM):
    """按候选文本返回预设分数,模拟 LLM 打分。"""
    def __init__(self, scores: dict):
        self.scores = scores

    def generate(self, prompt: str) -> str:
        for key, val in self.scores.items():
            if key in prompt:
                return val
        return "0"


def _rc(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=cid, text=text, source="fastapi/x.md",
                    library="fastapi", chunk_index=0),
        score=score,
    )


def test_rerank_reorders_by_llm_relevance():
    # 向量检索把不相关的排前面;重排应按 LLM 相关性分重新排序
    candidates = [
        _rc("c1", "关于赞助商的内容", 0.9),   # 向量分高但其实不相关
        _rc("c2", "路径参数用花括号声明", 0.5),  # 向量分低但真正相关
    ]
    llm = _ScriptLLM({"赞助商": "1", "花括号": "9"})
    reranker = LlmReranker(llm)

    out = reranker.rerank("路径参数怎么写", candidates, top_k=2)
    assert out[0].chunk.id == "c2"   # 相关的被提到第一
    assert out[1].chunk.id == "c1"


def test_rerank_truncates_to_top_k():
    candidates = [_rc(f"c{i}", f"text {i}", 0.5) for i in range(5)]
    llm = _ScriptLLM({"text 3": "9"})  # c3 最相关
    reranker = LlmReranker(llm)

    out = reranker.rerank("q", candidates, top_k=2)
    assert len(out) == 2
    assert out[0].chunk.id == "c3"


def test_rerank_handles_unparseable_score_as_zero():
    candidates = [_rc("c1", "aaa", 0.5), _rc("c2", "bbb", 0.5)]
    llm = _ScriptLLM({"aaa": "没有数字", "bbb": "7"})
    reranker = LlmReranker(llm)

    out = reranker.rerank("q", candidates, top_k=2)
    assert out[0].chunk.id == "c2"  # 能解析出分数的排前


def test_rerank_empty_candidates():
    reranker = LlmReranker(_ScriptLLM({}))
    assert reranker.rerank("q", [], top_k=4) == []
