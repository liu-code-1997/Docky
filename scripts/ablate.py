"""一键消融:扫 query_rewrite × rerank 四组,跑评估集,输出 markdown 对比表。

用法:
    python scripts/ablate.py [--dataset eval/dataset.json] [--scorer keyword]

依赖:Ollama 与 Qdrant 已启动,且已灌库、已迁移 dataset.json。
诚实边界:chunk_strategy 是灌库期参数(切换需重灌),不在此进程内扫描。
"""
import argparse
import json
from pathlib import Path

from rag.config import get_settings
from rag.profile import load_profile
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.retrieve import retrieve
from rag.generate import answer as generate_answer
from rag.scoring import get_scorer
from rag.evaluate import evaluate_sample, aggregate
from rag.query_rewrite import LlmQueryRewriter
from rag.rerank import build_reranker
from rag.multi_query import expand_queries
from rag.ordering import reorder_for_long_context
from rag.ablate import format_comparison_table
from rag.models import EvalSample


def main() -> None:
    parser = argparse.ArgumentParser(description="消融扫描 rewrite×rerank")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--scorer", default=None)
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.profile)
    scorer_name = args.scorer or settings.eval_scorer
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.eval_temperature)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    scorer = get_scorer(scorer_name, llm=llm, embedder=embedder)
    samples = [EvalSample(**d) for d in
               json.loads(Path(args.dataset).read_text(encoding="utf-8"))]

    # M11:多查询扩展器(从 settings 构造)
    query_expander = (
        (lambda q: expand_queries(llm, q, settings.multi_query_n))
        if settings.multi_query else None
    )

    results = []
    for use_rewrite in (False, True):
        for use_rerank in (False, True):
            rewriter = LlmQueryRewriter(llm, profile.rewrite_prompt) if (use_rewrite and profile.rewrite_prompt) else None
            reranker = build_reranker(settings.rerank_provider, llm, settings.rerank_cross_encoder_model) if use_rerank else None
            rows = []
            for s in samples:
                retrieved = retrieve(s.question, embedder, store,
                                     top_k=settings.top_k, library=None,
                                     rewriter=rewriter, reranker=reranker,
                                     rerank_factor=settings.rerank_factor,
                                     query_prefix=profile.embed_query_prefix,
                                     hybrid=settings.hybrid,
                                     hybrid_prefetch_factor=settings.hybrid_prefetch_factor,
                                     query_expander=query_expander)
                ctx = reorder_for_long_context(retrieved) if settings.reorder_context else retrieved
                ans = generate_answer(s.question, ctx, llm,
                                      persona=profile.persona,
                                      refusal_text=profile.refusal_text,
                                      inline_citations=settings.inline_citations)
                rows.append(evaluate_sample(s, retrieved, ans, scorer,
                                            refusal_marker=profile.refusal_marker))
            results.append({
                "config": f"rewrite={'on' if use_rewrite else 'off'} "
                          f"rerank={'on' if use_rerank else 'off'}",
                "agg": aggregate(rows),
            })

    print(f"\n评分器={scorer_name} top_k={settings.top_k} "
          f"样本={len(samples)}(chunk 策略需手动重灌后单独跑)\n")
    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
