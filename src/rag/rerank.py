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


_LISTWISE_PROMPT = """下面是若干候选资料,请按与【问题】的相关性从高到低排序。
只输出候选编号,用逗号分隔(如 3,1,4,2),不要解释、不要输出其它内容。

【问题】{question}

{candidates}
排序:"""


def _parse_order(reply: str, n: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for tok in re.findall(r"\d+", reply):
        i = int(tok)
        if 1 <= i <= n and i not in seen:
            seen.add(i)
            order.append(i)
    return order


class ListwiseLlmReranker(Reranker):
    """一次 LLM 调用对全部候选排序(比逐条打分快,零新依赖)。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def rerank(self, question: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        blocks = "\n".join(f"[{i}] {rc.chunk.text[:300]}"
                           for i, rc in enumerate(candidates, start=1))
        reply = self.llm.generate(
            _LISTWISE_PROMPT.format(question=question, candidates=blocks))
        order = _parse_order(reply, len(candidates))
        seen = set(order)
        order += [i for i in range(1, len(candidates) + 1) if i not in seen]  # 未提及者补末尾
        ranked = [candidates[i - 1] for i in order]
        return ranked[:top_k]


class CrossEncoderReranker(Reranker):
    """bge 类 cross-encoder 重排。torch/sentence_transformers 惰性加载(首次 rerank 时)。"""

    def __init__(self, model_name: str, scorer=None):
        self.model_name = model_name
        self._scorer = scorer          # callable(list[tuple[str,str]])->list[float];测试注入
        self._model = None

    def _score(self, pairs):
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model.predict(pairs)

    def rerank(self, question: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(question, rc.chunk.text) for rc in candidates]
        scores = self._score(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda t: t[0], reverse=True)
        return [RetrievedChunk(chunk=rc.chunk, score=float(s)) for s, rc in ranked[:top_k]]


def build_reranker(provider: str, llm: LLM, cross_encoder_model: str = "BAAI/bge-reranker-v2-m3") -> Reranker:
    """按名字构造重排器。将来接 bge cross-encoder 只需加一分支。"""
    if provider == "llm_listwise":
        return ListwiseLlmReranker(llm)
    if provider == "llm_pointwise":
        return LlmReranker(llm)
    if provider == "cross_encoder":
        return CrossEncoderReranker(cross_encoder_model)
    raise ValueError(f"未知 rerank_provider: {provider}(可选 llm_listwise | llm_pointwise | cross_encoder)")
