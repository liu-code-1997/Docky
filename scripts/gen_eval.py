"""从 docs 生成评估候选题,写入 eval/candidates.json(供人工复核)。

用法:
    python scripts/gen_eval.py --docs docs --per-source 2 --out eval/candidates.json

依赖:Ollama 已启动(仅需 LLM,不需 Qdrant)。
"""
import argparse
import json
from pathlib import Path

from rag.config import get_settings
from rag.profile import load_profile
from rag.providers.ollama_llm import OllamaLLM
from rag.loader import load_chunks_from_dir
from rag.gen_eval import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="生成评估候选题")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval/candidates.json")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.profile)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.eval_temperature)
    chunks = load_chunks_from_dir(Path(args.docs),
                                  chunk_size=settings.chunk_size,
                                  overlap=settings.chunk_overlap,
                                  strategy=settings.chunk_strategy,
                                  noise_markers=profile.noise_markers)
    candidates = generate_candidates(chunks, llm,
                                     per_source=args.per_source, seed=args.seed)
    Path(args.out).write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"生成候选 {len(candidates)} 条 → {args.out}(请人工复核后并入 dataset.json)")


if __name__ == "__main__":
    main()
