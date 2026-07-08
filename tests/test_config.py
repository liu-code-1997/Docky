from rag.config import Settings


def test_settings_have_sensible_defaults():
    s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.llm_model == "qwen2.5:7b"
    assert s.embedding_model == "nomic-embed-text"
    assert s.collection_name == "rag_docs"
    # 切分默认参数(M1 会用到)
    assert s.chunk_size == 800
    assert s.chunk_overlap == 150
    assert s.top_k == 4
    # 评估默认用关键词评分(M4)
    assert s.eval_scorer == "keyword"


def test_settings_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    s = Settings()
    assert s.llm_model == "qwen2.5:14b"


def test_eval_scorer_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("EVAL_SCORER", "semantic")
    s = Settings()
    assert s.eval_scorer == "semantic"
