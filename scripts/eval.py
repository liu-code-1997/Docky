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
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline
from rag.retrieve import retrieve
from rag.generate import answer as generate_answer
from rag.scoring import get_scorer
from rag.evaluate import evaluate_sample, aggregate
from rag.query_rewrite import LlmQueryRewriter
from rag.models import EvalSample


def main() -> None:
    parser = argparse.ArgumentParser(description="跑 RAG 评估集")
    parser.add_argument("--scorer", default=None,
                        help="覆盖 EVAL_SCORER:keyword | llm_judge | semantic")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--query-rewrite", dest="query_rewrite",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖 QUERY_REWRITE:开启检索前查询改写(M5②)")
    args = parser.parse_args()

    settings = get_settings()
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
    rewriter = LlmQueryRewriter(llm) if use_rewrite else None

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = [EvalSample(**d) for d in data]

    print(f"评分方法: {scorer_name} | top_k={settings.top_k} | "
          f"query_rewrite={use_rewrite} | {len(samples)} 条样本\n")
    header = f"{'hit':>4} {'mrr':>5} {'gen':>5} {'拒答':>4}  问题"
    print(header)
    print("-" * 72)

    rows = []
    for s in samples:
        retrieved = retrieve(s.question, embedder, store,
                             top_k=settings.top_k, library=None,
                             rewriter=rewriter)
        ans = generate_answer(s.question, retrieved, llm)
        r = evaluate_sample(s, retrieved, ans, scorer)
        rows.append(r)
        print(f"{'✓' if r['hit'] else '✗':>4} "
              f"{r['mrr']:>5.2f} {r['gen_score']:>5.2f} "
              f"{'是' if r['refusal'] else '否':>4}  {s.question}")

    agg = aggregate(rows)
    print("-" * 72)
    print(f"\n=== 汇总({agg['n']} 条)===")
    print(f"检索命中率 hit@{settings.top_k}: {agg['hit_rate']:.1%}")
    print(f"平均 MRR:              {agg['avg_mrr']:.3f}")
    print(f"平均生成分({scorer_name}): {agg['avg_gen_score']:.3f}")
    print(f"拒答条数:              {agg['refusals']}/{agg['n']}")


if __name__ == "__main__":
    main()
