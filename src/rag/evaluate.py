"""评估逻辑(M4;M7 升多来源)。检索层指标 + 生成层评分 + 聚合,全为纯函数。"""
import math

from rag.interfaces import AnswerScorer
from rag.models import Answer, EvalSample, RetrievedChunk

_REFUSAL_MARKER = "无法回答"


def _relevance_flags(expected_sources: list[str],
                     retrieved: list[RetrievedChunk]) -> list[bool]:
    """逐个检索块是否相关(source ∈ expected_sources)。"""
    relevant = set(expected_sources)
    return [rc.chunk.source in relevant for rc in retrieved]


def hit_at_k(expected_sources: list[str],
             retrieved: list[RetrievedChunk]) -> bool:
    """top-k 中出现任一相关来源即命中。"""
    if not expected_sources:
        return False
    return any(_relevance_flags(expected_sources, retrieved))


def reciprocal_rank(expected_sources: list[str],
                    retrieved: list[RetrievedChunk]) -> float:
    """第一个相关块的倒数排名;无则 0。"""
    if not expected_sources:
        return 0.0
    for i, rel in enumerate(_relevance_flags(expected_sources, retrieved),
                            start=1):
        if rel:
            return 1.0 / i
    return 0.0


def recall_at_k(expected_sources: list[str],
                retrieved: list[RetrievedChunk]) -> float:
    """top-k 命中的不同相关来源数 / 相关来源总数。"""
    if not expected_sources:
        return 0.0
    retrieved_sources = {rc.chunk.source for rc in retrieved}
    hit_sources = retrieved_sources & set(expected_sources)
    return len(hit_sources) / len(set(expected_sources))


def precision_at_k(expected_sources: list[str],
                   retrieved: list[RetrievedChunk]) -> float:
    """top-k 中相关块数 / 返回块数(k)。"""
    if not expected_sources or not retrieved:
        return 0.0
    flags = _relevance_flags(expected_sources, retrieved)
    return sum(flags) / len(retrieved)


def ndcg_at_k(expected_sources: list[str],
              retrieved: list[RetrievedChunk]) -> float:
    """二元相关性 nDCG@k;IDCG 用 top-k 内相关数为理想上界(见 spec 诚实说明)。"""
    if not expected_sources or not retrieved:
        return 0.0
    flags = _relevance_flags(expected_sources, retrieved)
    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2)
              for i, rel in enumerate(flags))
    n_rel = sum(flags)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg


def is_refusal(answer_text: str) -> bool:
    """答案是否为拒答。"""
    return _REFUSAL_MARKER in answer_text


def evaluate_sample(sample: EvalSample, retrieved: list[RetrievedChunk],
                    answer: Answer, scorer: AnswerScorer) -> dict:
    """评估一条样本,返回该条的各项指标。"""
    return {
        "question": sample.question,
        "hit": hit_at_k(sample.expected_source, retrieved),
        "mrr": reciprocal_rank(sample.expected_source, retrieved),
        "gen_score": scorer.score(answer.text, sample),
        "refusal": is_refusal(answer.text),
    }


def aggregate(rows: list[dict]) -> dict:
    """把逐条结果聚合成总报告。"""
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit_rate": 0.0, "avg_mrr": 0.0,
                "avg_gen_score": 0.0, "refusals": 0}
    return {
        "n": n,
        "hit_rate": sum(1 for r in rows if r["hit"]) / n,
        "avg_mrr": sum(r["mrr"] for r in rows) / n,
        "avg_gen_score": sum(r["gen_score"] for r in rows) / n,
        "refusals": sum(1 for r in rows if r["refusal"]),
    }
