import httpx
from rag.providers.ollama_embedder import OllamaEmbedder


def test_embed_one_calls_ollama_and_returns_vector(monkeypatch):
    def fake_post(url, json, timeout):
        assert "/api/embeddings" in url
        assert json["model"] == "nomic-embed-text"
        assert json["prompt"] == "hello"
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")
    vec = emb.embed_one("hello")
    assert vec == [0.1, 0.2, 0.3]


def test_embed_batch_returns_one_vector_per_text(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(200, json={"embedding": [1.0, 2.0]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")
    vecs = emb.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(v == [1.0, 2.0] for v in vecs)
