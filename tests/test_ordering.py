from rag.ordering import reorder_for_long_context
from rag.models import Chunk, RetrievedChunk


def _rc(cid):
    return RetrievedChunk(chunk=Chunk(id=cid, text="t", source=cid, library="l", chunk_index=0), score=1.0)


def test_most_relevant_at_ends():
    out = reorder_for_long_context([_rc("r0"), _rc("r1"), _rc("r2"), _rc("r3")])
    ids = [c.chunk.id for c in out]
    assert ids[0] == "r0"        # 最相关在首
    assert ids[-1] == "r1"       # 次相关在尾
    assert set(ids) == {"r0", "r1", "r2", "r3"}   # 不丢不重


def test_short_lists_unchanged():
    assert [c.chunk.id for c in reorder_for_long_context([_rc("a")])] == ["a"]
    assert reorder_for_long_context([]) == []
