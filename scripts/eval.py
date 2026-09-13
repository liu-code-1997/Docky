"""跑评估集,量化 RAG 系统好坏(M4)。

用法:
    python scripts/eval.py                      # 用默认 keyword 评分
    EVAL_SCORER=semantic python scripts/eval.py # 换语义相似度
    python scripts/eval.py --scorer llm_judge   # 或命令行指定

依赖:Ollama 与 Qdrant 已启动,且已先跑过 scripts/ingest.py 灌库。
"""
import argparse
import json
from pathlib import Path

from rag.config import get_settings
from rag.profile import load_profile
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline
from rag.retrieve import retrieve
from rag.generate import answer as generate_answer
from rag.scoring import get_scorer
from rag.evaluate import evaluate_sample, aggregate
from rag.query_rewrite import LlmQueryRewriter
from rag.rerank import build_reranker
from rag.multi_query import expand_queries
from rag.ordering import reorder_for_long_context
from rag.agent import RagAgent
from rag.providers.chat_factory import build_chat_llm
from rag.models import EvalSample


def main() -> None:
    parser = argparse.ArgumentParser(description="跑 RAG 评估集")
    parser.add_argument("--scorer", default=None,
                        help="覆盖 EVAL_SCORER:keyword | llm_judge | semantic")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--query-rewrite", dest="query_rewrite",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖 QUERY_REWRITE:开启检索前查询改写(M5②)")
    parser.add_argument("--rerank", dest="rerank",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖 RERANK:开启检索后重排(M5③)")
    parser.add_argument("--agent", action="store_true",
                        help="用 M6 agent(自主检索循环)而非单轮 RAG")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.profile)
    scorer_name = args.scorer or settings.eval_scorer

    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    # 评估恒用 eval_temperature(默认 0):被评的答案生成 + llm_judge 裁判都可复现
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.eval_temperature)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    scorer = get_scorer(scorer_name, llm=llm, embedder=embedder)

    # M5②:查询改写(命令行 --query-rewrite 可覆盖 config)
    use_rewrite = args.query_rewrite if args.query_rewrite is not None else settings.query_rewrite
    rewriter = LlmQueryRewriter(llm, profile.rewrite_prompt) if (use_rewrite and profile.rewrite_prompt) else None

    # M5③:重排(命令行 --rerank 可覆盖 config)
    use_rerank = args.rerank if args.rerank is not None else settings.rerank
    reranker = build_reranker(settings.rerank_provider, llm) if use_rerank else None

    # M11:多查询扩展器(multi_query 开时)
    query_expander = (
        (lambda q: expand_queries(llm, q, settings.multi_query_n))
        if settings.multi_query else None
    )

    # M6:agent 模式 —— 用自主检索循环替代单轮 RAG
    agent = None
    if args.agent:
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

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = [EvalSample(**d) for d in data]

    mode = "agent" if args.agent else "single-turn"
    print(f"模式: {mode} | 评分: {scorer_name} | top_k={settings.top_k} | "
          f"query_rewrite={use_rewrite} | rerank={use_rerank} | {len(samples)} 条\n")

    rows = []
    for s in samples:
        if agent is not None:
            # agent 内部自主检索;评估看生成分与拒答(检索命中在循环内,不单独计)
            ans = agent.ask(s.question, library=None)
            gen = scorer.score(ans.text, s)
            refusal = profile.refusal_marker in ans.text
            negative = not s.expected_sources
            row = {"question": s.question, "negative": negative,
                   "refusal": refusal,
                   "refusal_correct": refusal if negative else not refusal}
            if not negative:
                row.update({"hit": False, "mrr": 0.0, "recall": 0.0,
                            "precision": 0.0, "ndcg": 0.0, "gen_score": gen})
            rows.append(row)
            print(f"gen={gen:>4.2f} {'拒答' if refusal else '答  '}  {s.question}")
        else:
            retrieved = retrieve(s.question, embedder, store,
                                 top_k=settings.top_k, library=None,
                                 rewriter=rewriter, reranker=reranker,
                                 rerank_factor=settings.rerank_factor,
                                 query_prefix=profile.embed_query_prefix,
                                 hybrid=settings.hybrid,
                                 hybrid_prefetch_factor=settings.hybrid_prefetch_factor,
                                 query_expander=query_expander)
            if settings.reorder_context:
                retrieved = reorder_for_long_context(retrieved)
            ans = generate_answer(s.question, retrieved, llm,
                                  persona=profile.persona,
                                  refusal_text=profile.refusal_text,
                                  inline_citations=settings.inline_citations)
            r = evaluate_sample(s, retrieved, ans, scorer,
                                refusal_marker=profile.refusal_marker)
            rows.append(r)
            if r["negative"]:
                print(f"[neg] {'拒答✓' if r['refusal_correct'] else '误答✗'}  {s.question}")
            else:
                print(f"hit={'✓' if r['hit'] else '✗'} "
                      f"recall={r['recall']:.2f} ndcg={r['ndcg']:.2f} "
                      f"gen={r['gen_score']:.2f}  {s.question}")

    agg = aggregate(rows)
    print("-" * 72)
    print(f"\n=== 汇总(正例 {agg['n_positive']} / 负例 {agg['n_negative']})===")
    if not args.agent:
        print(f"hit@{settings.top_k}:        {agg['hit_rate']:.1%}")
        print(f"recall@{settings.top_k}:     {agg['avg_recall']:.3f}")
        print(f"precision@{settings.top_k}:  {agg['avg_precision']:.3f}")
        print(f"nDCG@{settings.top_k}:       {agg['avg_ndcg']:.3f}")
        print(f"MRR:            {agg['avg_mrr']:.3f}")
    print(f"生成分({scorer_name}): {agg['avg_gen_score']:.3f}")
    print(f"拒答正确率:      {agg['refusal_accuracy']:.1%}")


if __name__ == "__main__":
    main()
