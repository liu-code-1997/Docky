"""从已有 chunk 反向生成评估候选题(M7)。领域无关:复用 loader 从 docs 出题。

只产候选写入 eval/candidates.json,绝不覆盖正式集;人工复核后再合并。
负例(拒答样本)不由本工具生成——LLM 不擅长造"库里没有"的题,手工加。
"""
import json
import random
import re

from rag.interfaces import LLM
from rag.models import Chunk

_GEN_PROMPT = """根据下面这段资料,出一道能被它回答的问题,并给出答案要点。
严格只输出 JSON,不要解释,格式:
{{"question": "问题", "expected_keywords": ["关键词1", "关键词2"], "expected_answer": "标准答案"}}

资料:
{passage}
JSON:"""


def build_gen_prompt(passage: str) -> str:
    return _GEN_PROMPT.format(passage=passage[:1000])


def parse_candidate(reply: str) -> dict | None:
    """从 LLM 回复里抽出 JSON 候选;抽不到或缺 question 则返回 None。"""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    q = obj.get("question")
    if not q or not isinstance(q, str):
        return None
    return {
        "question": q.strip(),
        "expected_keywords": obj.get("expected_keywords") or [],
        "expected_answer": obj.get("expected_answer") or "",
    }


def generate_candidate(chunk: Chunk, llm: LLM) -> dict | None:
    parsed = parse_candidate(llm.generate(build_gen_prompt(chunk.text)))
    if parsed is None:
        return None
    return {
        "question": parsed["question"],
        "expected_sources": [chunk.source],
        "expected_keywords": parsed["expected_keywords"],
        "expected_answer": parsed["expected_answer"],
    }


def sample_chunks(chunks: list[Chunk], per_source: int,
                  seed: int = 0) -> list[Chunk]:
    """按 source 分层抽样,保证覆盖不同文件;可复现(seed)。"""
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    rng = random.Random(seed)
    selected: list[Chunk] = []
    for source in sorted(by_source):
        group = list(by_source[source])
        rng.shuffle(group)
        selected.extend(group[:per_source])
    return selected


def generate_candidates(chunks: list[Chunk], llm: LLM,
                        per_source: int = 2, seed: int = 0) -> list[dict]:
    out: list[dict] = []
    for c in sample_chunks(chunks, per_source, seed):
        cand = generate_candidate(c, llm)
        if cand:
            out.append(cand)
    return out
