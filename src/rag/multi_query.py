"""多查询检索:生成查询变体 + 客户端 RRF 融合(M11)。零依赖。"""
from rag.interfaces import LLM
from rag.models import RetrievedChunk

_EXPAND_PROMPT = """针对下面的问题,生成 {k} 个措辞或角度不同、但意图相同的检索查询,便于多路检索。\
每行一个,不要编号、不要解释。

问题:{question}
查询:"""


def expand_queries(llm: LLM, question: str, n: int = 3) -> list[str]:
    if n <= 1:
        return [question]
    try:
        reply = llm.generate(_EXPAND_PROMPT.format(k=n - 1, question=question))
    except Exception:
        return [question]
    variants = [ln.strip(" -*\t") for ln in reply.splitlines() if ln.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for q in [question, *variants]:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:n]


def rrf_fuse(result_lists: list[list[RetrievedChunk]], top_k: int,
             k_rrf: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, rc in enumerate(results):
            cid = rc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
            best.setdefault(cid, rc)
    ranked = sorted(best.values(), key=lambda rc: scores[rc.chunk.id], reverse=True)
    return [RetrievedChunk(chunk=rc.chunk, score=scores[rc.chunk.id])
            for rc in ranked[:top_k]]
