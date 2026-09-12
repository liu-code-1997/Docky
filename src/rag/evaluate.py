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
    """评估一条:正例算排序指标+生成分,负例只算拒答正确性。"""
    negative = not sample.expected_sources
    refusal = is_refusal(answer.text)
    row: dict = {
        "question": sample.question,
        "negative": negative,
        "refusal": refusal,
    }
    if negative:
        row["refusal_correct"] = refusal          # 负例应拒答
    else:
        es = sample.expected_sources
        row["hit"] = hit_at_k(es, retrieved)
        row["mrr"] = reciprocal_rank(es, retrieved)
        row["recall"] = recall_at_k(es, retrieved)
        row["precision"] = precision_at_k(es, retrieved)
        row["ndcg"] = ndcg_at_k(es, retrieved)
        row["gen_score"] = scorer.score(answer.text, sample)
        row["refusal_correct"] = not refusal      # 正例不应误拒
    return row


def aggregate(rows: list[dict]) -> dict:
    """正例算排序指标+生成分,拒答正确率跨全体;两组分开、互不污染。"""
    positives = [r for r in rows if not r["negative"]]
    negatives = [r for r in rows if r["negative"]]

    def avg(items: list[dict], key: str) -> float:
        return sum(i[key] for i in items) / len(items) if items else 0.0

    return {
        "n": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "hit_rate": avg(positives, "hit"),
        "avg_mrr": avg(positives, "mrr"),
        "avg_recall": avg(positives, "recall"),
        "avg_precision": avg(positives, "precision"),
        "avg_ndcg": avg(positives, "ndcg"),
        "avg_gen_score": avg(positives, "gen_score"),
        "refusal_accuracy": avg(rows, "refusal_correct"),
    }
