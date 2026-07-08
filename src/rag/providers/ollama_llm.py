"""Ollama LLM 实现。

与 OllamaEmbedder 同构:逐次请求,非流式(stream=False),一次拿完整答案。
坑(notes/02):LLM 也有上下文长度上限,prompt 太长会被截断;M2 先不处理,留到评估阶段。
"""
import httpx
from rag.interfaces import LLM


class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

