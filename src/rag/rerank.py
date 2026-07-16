"""重排(M5 ③):检索后用 LLM 对候选按相关性重排,取前 top_k。

先召回较多候选(top_k × factor),再让 LLM 逐个给"与问题相关性"打分(0–10),
按分数重排。目的:把向量检索排在后面、但其实更相关的块提到前面。

局限:每个候选一次 LLM 调用,慢(top_k×factor 次/问题)。默认关闭。
"""
import re

from rag.interfaces import LLM, Reranker
from rag.models import RetrievedChunk


_SCORE_PROMPT = """判断下面这段资料与问题的相关程度,只输出 0 到 10 的一个整数,不要解释。

问题:{question}
资料:{passage}
相关性(0-10):"""


class LlmReranker(Reranker):
    def __init__(self, llm: LLM):
        self.llm = llm

    def _score(self, question: str, passage: str) -> float:
        reply = self.llm.generate(
            _SCORE_PROMPT.format(question=question, passage=passage[:500]))
        m = re.search(r"\d+(?:\.\d+)?", reply)
        return float(m.group()) if m else 0.0

    def rerank(self, question: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scored = [(self._score(question, rc.chunk.text), rc) for rc in candidates]
        # 按 LLM 相关性分降序;稳定排序保留原相对次序作为平手时的兜底
        scored.sort(key=lambda t: t[0], reverse=True)
        return [rc for _, rc in scored[:top_k]]
