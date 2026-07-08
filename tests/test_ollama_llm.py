import httpx
from rag.providers.ollama_llm import OllamaLLM


def test_generate_calls_ollama_and_returns_text(monkeypatch):
    def fake_post(url, json, timeout):
        assert "/api/generate" in url
        assert json["model"] == "qwen2.5:7b"
        assert json["prompt"] == "say hi"
        assert json["stream"] is False  # 非流式,一次拿完整答案
        return httpx.Response(200, json={"response": "hi there"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    out = llm.generate("say hi")
    assert out == "hi there"


def test_generate_raises_on_http_error(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(500, json={"error": "boom"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    import pytest
    with pytest.raises(httpx.HTTPStatusError):
        llm.generate("anything")
