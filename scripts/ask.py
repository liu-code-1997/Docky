"""命令行问答:对已灌库的文档提问。

用法:
    python scripts/ask.py "FastAPI 的依赖注入怎么写?"
    python scripts/ask.py "路径参数怎么声明?" --library fastapi

依赖:Ollama 与 Qdrant 已启动,且已先跑过 scripts/ingest.py 灌库。
"""
import argparse
from rag.config import get_settings
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline
from rag.query_rewrite import LlmQueryRewriter


def main() -> None:
    parser = argparse.ArgumentParser(description="对已灌库的文档提问")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--library", default=None,
                        help="只在某个库内检索,如 fastapi(默认全部)")
    args = parser.parse_args()

    settings = get_settings()

    # 装配真实 provider —— 选用哪个实现的决定只发生在这里
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.llm_temperature)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    rewriter = LlmQueryRewriter(llm) if settings.query_rewrite else None
    pipe = RagPipeline(embedder, store, llm, top_k=settings.top_k,
                       rewriter=rewriter)

    ans = pipe.ask(args.question, library=args.library)

    print("\n=== 答案 ===")
    print(ans.text)
    if ans.sources:
        print("\n=== 出处 ===")
        for s in ans.sources:
            print(f"- {s}")
    else:
        print("\n(无出处)")


if __name__ == "__main__":
    main()
