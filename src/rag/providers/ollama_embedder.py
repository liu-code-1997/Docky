"""Ollama embedding 实现。

坑(notes/02):embedding 模型有最大输入长度,过长会被静默截断。
本实现逐条请求,简单直观;真实生产可考虑批量/并发优化(YAGNI,暂不做)。
"""
import httpx
from rag.interfaces import Embedder


class OllamaEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_one(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
