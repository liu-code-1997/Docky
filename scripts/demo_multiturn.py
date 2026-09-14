"""M12 多轮会话演示:两轮对话,验证 condense 把第 2 轮的指代补全。

跑法:.venv/bin/python scripts/demo_multiturn.py
依赖:Ollama + Qdrant 已启动,rag_docs 已灌库。
"""
from rag.config import get_settings
from rag.profile import load_profile
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.query_rewrite import LlmQueryRewriter
from rag.rerank import build_reranker
from rag.retrieve import retrieve
from rag.multi_query import expand_queries
from rag.session_store import InMemorySessionStore
from rag.condense import condense_question
from rag.conversation import ConversationalRag


def main() -> None:
    s = get_settings()
    profile = load_profile(s.profile)
    embedder = OllamaEmbedder(s.ollama_base_url, s.embedding_model)
    llm = OllamaLLM(s.ollama_base_url, s.llm_model, temperature=s.llm_temperature)
    store = QdrantStore(collection_name=s.collection_name, url=s.qdrant_url)
    rewriter = LlmQueryRewriter(llm, profile.rewrite_prompt) if (s.query_rewrite and profile.rewrite_prompt) else None
    reranker = build_reranker(s.rerank_provider, llm, s.rerank_cross_encoder_model) if s.rerank else None
    query_expander = (lambda q: expand_queries(llm, q, s.multi_query_n)) if s.multi_query else None

    def retriever(query, library=None, top_k=s.top_k):
        return retrieve(query, embedder, store, top_k=top_k, library=library,
                        rewriter=rewriter, reranker=reranker,
                        rerank_factor=s.rerank_factor,
                        query_prefix=profile.embed_query_prefix,
                        hybrid=s.hybrid, hybrid_prefetch_factor=s.hybrid_prefetch_factor,
                        query_expander=query_expander)

    sess = InMemorySessionStore()
    conv = ConversationalRag(retriever, llm, sess, profile.persona, profile.refusal_text,
                             history_turns=s.history_turns, condense=s.condense)

    sid = "demo-1"
    turns = ["FastAPI 路径参数怎么声明?", "它能限定类型吗?"]
    for i, q in enumerate(turns, 1):
        history = sess.get_history(sid)
        standalone = condense_question(llm, q, history) if (s.condense and history) else q
        print(f"\n{'='*60}\n第 {i} 轮 用户问:{q}")
        if standalone != q:
            print(f"  ↳ condense 改写后的独立检索问句:{standalone}")
        ans = conv.chat(sid, q)
        print(f"  ↳ 命中来源:{ans.sources}")
        print(f"  ↳ 回答:{ans.text}")


if __name__ == "__main__":
    main()
