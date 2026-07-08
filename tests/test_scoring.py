import pytest
from rag.interfaces import LLM, Embedder
from rag.models import EvalSample
from rag.scoring import (
    KeywordScorer, LlmJudgeScorer, SemanticScorer, get_scorer,
)


def _sample(keywords, expected_answer="标准答案") -> EvalSample:
    return EvalSample(
        question="q",
        expected_source="fastapi/a.md",
        expected_keywords=keywords,
        expected_answer=expected_answer,
    )


def test_keyword_scorer_all_hit():
    scorer = KeywordScorer()
    s = _sample(["路径参数", "path"])
    score = scorer.score("路径参数在 path 里用花括号声明", s)
    assert score == 1.0


def test_keyword_scorer_partial_hit():
    scorer = KeywordScorer()
    s = _sample(["路径参数", "path", "花括号", "占位符"])
    # 命中 3/4
    score = scorer.score("路径参数用花括号写在 path 中", s)
    assert score == 0.75


def test_keyword_scorer_no_hit():
    scorer = KeywordScorer()
    s = _sample(["依赖注入", "Depends"])
    assert scorer.score("完全无关的答案", s) == 0.0


def test_keyword_scorer_is_case_insensitive():
    scorer = KeywordScorer()
    s = _sample(["Path", "FastAPI"])
    assert scorer.score("path works in fastapi", s) == 1.0


def test_keyword_scorer_empty_keywords_returns_zero():
    scorer = KeywordScorer()
    s = _sample([])
    # 没有关键词可评时约定返回 0(避免除零,也表示"无法据此判定为对")
    assert scorer.score("任何答案", s) == 0.0


class _FakeLLM(LLM):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def test_llm_judge_parses_score_and_builds_prompt():
    llm = _FakeLLM("1")
    scorer = LlmJudgeScorer(llm)
    s = _sample(["x"], expected_answer="路径参数用花括号声明")
    score = scorer.score("模型的答案", s)

    assert score == 1.0
    # prompt 里应带上标准答案、模型答案、问题
    assert "路径参数用花括号声明" in llm.last_prompt
    assert "模型的答案" in llm.last_prompt


def test_llm_judge_maps_half_and_zero():
    assert LlmJudgeScorer(_FakeLLM("0.5")).score("a", _sample(["x"])) == 0.5
    assert LlmJudgeScorer(_FakeLLM("0")).score("a", _sample(["x"])) == 0.0


def test_llm_judge_extracts_score_from_verbose_reply():
    # LLM 可能话痨,要能从文本里抽出数字
    llm = _FakeLLM("我认为这个答案基本正确,评分:1")
    assert LlmJudgeScorer(llm).score("a", _sample(["x"])) == 1.0


def test_llm_judge_unparseable_reply_returns_zero():
    llm = _FakeLLM("完全没有数字的胡言乱语")
    assert LlmJudgeScorer(llm).score("a", _sample(["x"])) == 0.0


class _FakeEmbedder(Embedder):
    """按文本内容返回预设向量,让余弦可预测。"""
    def __init__(self, mapping):
        self.mapping = mapping

    def embed_one(self, text: str) -> list[float]:
        return self.mapping[text]

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def test_semantic_identical_vectors_score_one():
    emb = _FakeEmbedder({"答案": [1.0, 0.0], "标准": [1.0, 0.0]})
    scorer = SemanticScorer(emb)
    s = _sample(["x"], expected_answer="标准")
    assert scorer.score("答案", s) == pytest.approx(1.0)


def test_semantic_orthogonal_vectors_score_zero():
    emb = _FakeEmbedder({"答案": [1.0, 0.0], "标准": [0.0, 1.0]})
    scorer = SemanticScorer(emb)
    s = _sample(["x"], expected_answer="标准")
    # 正交 -> 余弦 0(clamp 到 [0,1])
    assert scorer.score("答案", s) == pytest.approx(0.0)


def test_semantic_opposite_vectors_clamped_to_zero():
    emb = _FakeEmbedder({"答案": [1.0, 0.0], "标准": [-1.0, 0.0]})
    scorer = SemanticScorer(emb)
    s = _sample(["x"], expected_answer="标准")
    # 余弦 -1,负分无意义,clamp 到 0
    assert scorer.score("答案", s) == pytest.approx(0.0)


def test_get_scorer_returns_right_impl():
    llm = _FakeLLM("1")
    emb = _FakeEmbedder({})
    assert isinstance(get_scorer("keyword"), KeywordScorer)
    assert isinstance(get_scorer("llm_judge", llm=llm), LlmJudgeScorer)
    assert isinstance(get_scorer("semantic", embedder=emb), SemanticScorer)


def test_get_scorer_unknown_raises():
    with pytest.raises(ValueError):
        get_scorer("nonsense")
