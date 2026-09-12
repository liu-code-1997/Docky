import math

from rag.evaluate import (
    hit_at_k, reciprocal_rank, recall_at_k, precision_at_k, ndcg_at_k,
    is_refusal,
)
from rag.models import Chunk, RetrievedChunk


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
