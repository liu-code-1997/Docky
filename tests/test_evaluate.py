import math

from rag.evaluate import (
    hit_at_k, reciprocal_rank, recall_at_k, precision_at_k, ndcg_at_k,
    is_refusal, evaluate_sample, aggregate,
)
from rag.models import Chunk, RetrievedChunk, Answer, EvalSample


def _rc(source: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::0", text="t", source=source,
                    library="lib", chunk_index=0),
        score=score,
    )


def test_hit_at_k_any_relevant():
    retrieved = [_rc("x.md"), _rc("a.md")]
    assert hit_at_k(["a.md", "b.md"], retrieved) is True
    assert hit_at_k(["z.md"], retrieved) is False


def test_reciprocal_rank_first_relevant_position():
    retrieved = [_rc("x.md"), _rc("a.md"), _rc("b.md")]
    assert reciprocal_rank(["b.md"], retrieved) == 1.0 / 3
    assert reciprocal_rank(["a.md"], retrieved) == 1.0 / 2
    assert reciprocal_rank(["z.md"], retrieved) == 0.0


def test_recall_at_k_distinct_sources():
    retrieved = [_rc("a.md"), _rc("a.md"), _rc("x.md")]
    # 相关源 {a,b},召回命中 {a} → 1/2
    assert recall_at_k(["a.md", "b.md"], retrieved) == 0.5
    assert recall_at_k(["a.md"], retrieved) == 1.0


def test_precision_at_k_relevant_chunks_over_k():
    retrieved = [_rc("a.md"), _rc("x.md"), _rc("b.md"), _rc("y.md")]
    # 4 个里 2 个相关 → 0.5
    assert precision_at_k(["a.md", "b.md"], retrieved) == 0.5


def test_ndcg_at_k_rewards_higher_rank():
    top = [_rc("a.md"), _rc("x.md")]     # 相关在第1位
    low = [_rc("x.md"), _rc("a.md")]     # 相关在第2位
    assert ndcg_at_k(["a.md"], top) == 1.0
    assert ndcg_at_k(["a.md"], low) == math.log2(2) / math.log2(3)


def test_metrics_empty_and_no_hit_edges():
    assert recall_at_k(["a.md"], []) == 0.0
    assert precision_at_k(["a.md"], []) == 0.0
    assert ndcg_at_k(["a.md"], [_rc("x.md")]) == 0.0   # n_rel=0
    assert hit_at_k([], [_rc("a.md")]) is False        # 负例无排序指标


def test_is_refusal_detects_cannot_answer():
    assert is_refusal("根据现有资料无法回答。") is True
    assert is_refusal("路径参数用花括号声明。") is False


def test_is_refusal_uses_injected_marker():
    assert is_refusal("这条资料没有", marker="资料没有") is True
    assert is_refusal("这条资料没有", marker="无法回答") is False


class _FixedScorer:
    def score(self, answer_text, sample):
        return 1.0


def test_evaluate_sample_positive():
    sample = EvalSample(question="q", expected_sources=["a.md"])
    retrieved = [_rc("a.md"), _rc("x.md")]
    row = evaluate_sample(sample, retrieved, Answer(text="答案"), _FixedScorer())
    assert row["negative"] is False
    assert row["hit"] is True
    assert row["gen_score"] == 1.0
    assert row["refusal_correct"] is True    # 正例未拒答 = 正确


def test_evaluate_sample_positive_wrong_refusal():
    sample = EvalSample(question="q", expected_sources=["a.md"])
    row = evaluate_sample(sample, [_rc("a.md")],
                          Answer(text="根据现有资料无法回答"), _FixedScorer())
    assert row["refusal_correct"] is False   # 正例误拒 = 错


def test_evaluate_sample_negative_refusal_correct():
    sample = EvalSample(question="q")         # 负例
    row = evaluate_sample(sample, [],
                          Answer(text="根据现有资料无法回答"), _FixedScorer())
    assert row["negative"] is True
    assert row["refusal_correct"] is True     # 负例拒答 = 正确
    assert "hit" not in row                    # 负例不含排序指标


def test_aggregate_splits_positive_negative():
    rows = [
        {"negative": False, "hit": True, "mrr": 1.0, "recall": 1.0,
         "precision": 0.5, "ndcg": 1.0, "gen_score": 0.8,
         "refusal": False, "refusal_correct": True},
        {"negative": True, "refusal": True, "refusal_correct": True},
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["n_positive"] == 1
    assert agg["n_negative"] == 1
    assert agg["hit_rate"] == 1.0
    assert agg["avg_gen_score"] == 0.8
    assert agg["refusal_accuracy"] == 1.0
