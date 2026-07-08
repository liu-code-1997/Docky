"""评估逻辑(M4):检索层指标 + 生成层评分 + 聚合。

检索层(与评分方法无关,始终算):
- hit@k:期望来源是否出现在检索到的 Top-K 里
- MRR:期望来源的倒数排名(越靠前越高)

生成层:调注入的 AnswerScorer 打分 + 检测拒答。

全部纯函数,便于单测;真实 pipeline 由 CLI 装配后传入结果。
"""
from rag.interfaces import AnswerScorer
from rag.models import Answer, EvalSample, RetrievedChunk

_REFUSAL_MARKER = "无法回答"


def hit_at_k(expected_source: str | None,
             retrieved: list[RetrievedChunk]) -> bool:
    """期望来源是否在检索结果里。"""
    if not expected_source:
        return False
    return any(rc.chunk.source == expected_source for rc in retrieved)


def reciprocal_rank(expected_source: str | None,
                    retrieved: list[RetrievedChunk]) -> float:
    """期望来源的倒数排名:第1位=1.0,第2位=0.5,...;不在则 0。"""
    if not expected_source:
        return 0.0
    for i, rc in enumerate(retrieved, start=1):
        if rc.chunk.source == expected_source:
            return 1.0 / i
    return 0.0


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
