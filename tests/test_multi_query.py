from rag.multi_query import expand_queries, rrf_fuse
from rag.models import Chunk, RetrievedChunk


def _rc(cid, score=1.0):
    return RetrievedChunk(chunk=Chunk(id=cid, text="t", source=cid, library="l", chunk_index=0), score=score)


class _FakeLLM:
    def __init__(self, reply): self.reply = reply
    def generate(self, prompt): return self.reply


def test_expand_includes_original_and_dedupes():
    out = expand_queries(_FakeLLM("变体一\n变体二\n变体一"), "原问题", n=3)
    assert out[0] == "原问题"
    assert "变体一" in out and "变体二" in out
    assert len(out) == len(set(out)) <= 3


def test_expand_n1_returns_only_original():
    assert expand_queries(_FakeLLM("x"), "q", n=1) == ["q"]


def test_expand_empty_reply_falls_back():
    assert expand_queries(_FakeLLM("   "), "q", n=3) == ["q"]


def test_rrf_fuse_ranks_consensus_first():
    # a 在两个列表都靠前 → 融合应排第一
    l1 = [_rc("a"), _rc("b"), _rc("c")]
    l2 = [_rc("a"), _rc("d")]
    out = rrf_fuse([l1, l2], top_k=3)
    assert out[0].chunk.id == "a"
    assert out[0].score >= out[1].score            # 分数降序


def test_rrf_fuse_dedupes_by_id():
    l1 = [_rc("a")]; l2 = [_rc("a")]
    out = rrf_fuse([l1, l2], top_k=5)
    assert len(out) == 1
