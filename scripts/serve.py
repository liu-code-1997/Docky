"""启动 RAG 问答 HTTP 服务。

用法:
    python scripts/serve.py                 # 默认 0.0.0.0:8000
    python scripts/serve.py --port 9000
    然后访问 http://localhost:8000/docs 看交互式 API 文档。

依赖:Ollama 与 Qdrant 已启动,且已先跑过 scripts/ingest.py 灌库。
"""
import argparse
import uvicorn

from rag.config import get_settings
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline
from rag.api import create_app


def build_app():
    """装配真实 provider 并返回 FastAPI app —— 选用哪个实现的决定只在这里。"""
    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    pipeline = RagPipeline(embedder, store, llm, top_k=settings.top_k)
    return create_app(pipeline, store)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 RAG 问答 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
