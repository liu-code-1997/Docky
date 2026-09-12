from rag.gen_eval import (
    parse_candidate, generate_candidate, sample_chunks, generate_candidates,
)
from rag.models import Chunk


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt):
        return self.reply


def _chunk(source, i=0):
    return Chunk(id=f"{source}::{i}", text="正文", source=source,
                 library="lib", chunk_index=i)


def test_parse_candidate_extracts_json():
    reply = '前言 {"question": "Q?", "expected_keywords": ["k1"], "expected_answer": "A"} 结尾'
    got = parse_candidate(reply)
    assert got == {"question": "Q?", "expected_keywords": ["k1"],
                   "expected_answer": "A"}


def test_parse_candidate_bad_json_returns_none():
    assert parse_candidate("没有 json") is None


def test_generate_candidate_attaches_source():
    llm = _FakeLLM('{"question": "Q?", "expected_keywords": [], "expected_answer": "A"}')
    cand = generate_candidate(_chunk("a.md"), llm)
    assert cand["expected_sources"] == ["a.md"]
    assert cand["question"] == "Q?"


def test_sample_chunks_stratified_per_source():
    chunks = [_chunk("a.md", 0), _chunk("a.md", 1),
              _chunk("b.md", 0), _chunk("b.md", 1)]
    picked = sample_chunks(chunks, per_source=1, seed=0)
    assert {c.source for c in picked} == {"a.md", "b.md"}
    assert len(picked) == 2


def test_generate_candidates_skips_unparseable():
    chunks = [_chunk("a.md")]
    ok = generate_candidates(chunks, _FakeLLM('{"question":"Q?","expected_keywords":[],"expected_answer":"A"}'), per_source=1)
    bad = generate_candidates(chunks, _FakeLLM("垃圾"), per_source=1)
    assert len(ok) == 1 and bad == []
