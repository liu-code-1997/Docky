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
from rag.profile import load_profile
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline
from rag.query_rewrite import LlmQueryRewriter
from rag.rerank import build_reranker
from rag.retrieve import retrieve
from rag.multi_query import expand_queries
from rag.agent import RagAgent
from rag.providers.chat_factory import build_chat_llm
from rag.api import create_app


def build_app():
    """装配真实 provider 并返回 FastAPI app —— 选用哪个实现的决定只在这里。"""
    settings = get_settings()
    profile = load_profile(settings.profile)
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.llm_temperature)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    rewriter = LlmQueryRewriter(llm, profile.rewrite_prompt) if (settings.query_rewrite and profile.rewrite_prompt) else None
    reranker = build_reranker(settings.rerank_provider, llm, settings.rerank_cross_encoder_model) if settings.rerank else None
    # M11:多查询扩展器(multi_query 开时)
    query_expander = (
        (lambda q: expand_queries(llm, q, settings.multi_query_n))
        if settings.multi_query else None
    )
    pipeline = RagPipeline(embedder, store, llm, top_k=settings.top_k,
                           rewriter=rewriter, reranker=reranker,
                           rerank_factor=settings.rerank_factor,
                           persona=profile.persona, refusal_text=profile.refusal_text,
                           query_prefix=profile.embed_query_prefix,
                           hybrid=settings.hybrid,
                           hybrid_prefetch_factor=settings.hybrid_prefetch_factor,
                           multi_query=settings.multi_query,
                           multi_query_n=settings.multi_query_n,
                           reorder_context=settings.reorder_context,
                           inline_citations=settings.inline_citations)

    # M6:装配 agent —— retriever 回调复用现有 retrieve 链路(含改写/重排)
    def retriever(query, library=None, top_k=settings.top_k):
        return retrieve(query, embedder, store, top_k=top_k, library=library,
                        rewriter=rewriter, reranker=reranker,
                        rerank_factor=settings.rerank_factor,
                        query_prefix=profile.embed_query_prefix,
                        hybrid=settings.hybrid,
                        hybrid_prefetch_factor=settings.hybrid_prefetch_factor,
                        query_expander=query_expander)

    agent = RagAgent(llm=build_chat_llm(settings), retriever=retriever,
                     top_k=settings.top_k, max_steps=settings.agent_max_steps,
                     persona=profile.persona, refusal_text=profile.refusal_text)

    return create_app(pipeline, store, agent=agent)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 RAG 问答 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
