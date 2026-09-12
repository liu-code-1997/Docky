from rag.models import Chunk, RetrievedChunk
from rag.evaluate import (
    hit_at_k, reciprocal_rank, is_refusal,
)


def _rc(source: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::0", text="t", source=source,
                    library=source.split("/")[0], chunk_index=0),
        score=score,
    )


def test_hit_at_k_true_when_source_present():
    retrieved = [_rc("fastapi/a.md"), _rc("fastapi/b.md")]
    assert hit_at_k("fastapi/b.md", retrieved) is True


def test_hit_at_k_false_when_absent():
    retrieved = [_rc("fastapi/a.md")]
    assert hit_at_k("fastapi/z.md", retrieved) is False


def test_reciprocal_rank_rewards_earlier_position():
    retrieved = [_rc("fastapi/a.md"), _rc("fastapi/b.md"), _rc("fastapi/c.md")]
    assert reciprocal_rank("fastapi/a.md", retrieved) == 1.0     # 第1位
    assert reciprocal_rank("fastapi/b.md", retrieved) == 0.5     # 第2位
    assert reciprocal_rank("fastapi/c.md", retrieved) == 1 / 3   # 第3位


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank("fastapi/z.md", [_rc("fastapi/a.md")]) == 0.0


def test_is_refusal_detects_cannot_answer():
    assert is_refusal("根据现有资料无法回答。") is True
    assert is_refusal("路径参数用花括号声明。") is False
