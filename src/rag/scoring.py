"""生成层评分器(M4):三种可插拔实现。

A keyword    —— 关键词命中比例;确定、零成本,但换个说法会误判。
B llm_judge  —— 让 LLM 打分;最懂语义,但有噪声、慢。
C semantic   —— 答案与标准答案的余弦相似度;懂语义、确定,但依赖阈值。

三种分数不同等可信,报告里如实标注(见 notes/06)。
"""
import math
import re

from rag.interfaces import AnswerScorer, LLM, Embedder
from rag.models import EvalSample


class KeywordScorer(AnswerScorer):
    """方法A:期望关键词在答案中命中的比例(大小写不敏感)。"""

    def score(self, answer_text: str, sample: EvalSample) -> float:
        keywords = sample.expected_keywords
        if not keywords:
            return 0.0  # 无关键词可评:约定返回 0,避免除零
        text = answer_text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return hits / len(keywords)


_JUDGE_PROMPT = """你是评分员。请判断【学生答案】相对【标准答案】是否正确回答了【问题】。
只输出一个分数,不要解释:
- 1   完全正确
- 0.5 部分正确
- 0   错误或答非所问

【问题】{question}
【标准答案】{expected}
【学生答案】{actual}
【分数】"""


class LlmJudgeScorer(AnswerScorer):
    """方法B:让 LLM 打分 0 / 0.5 / 1。复用现有 LLM 接口。

    局限:LLM 打分有随机噪声,同一答案两次可能不同;裁判本身也会错。
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def score(self, answer_text: str, sample: EvalSample) -> float:
        prompt = _JUDGE_PROMPT.format(
            question=sample.question,
            expected=sample.expected_answer,
            actual=answer_text,
        )
        reply = self.llm.generate(prompt)
        # 从回复里抽第一个数字(容忍话痨的裁判)。取 1 / 0.5 / 0 三档最近值。
        m = re.search(r"\d+(?:\.\d+)?", reply)
        if not m:
            return 0.0
        val = float(m.group())
        return min([0.0, 0.5, 1.0], key=lambda x: abs(x - val))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticScorer(AnswerScorer):
    """方法C:答案与标准答案的余弦相似度(clamp 到 [0,1])。复用现有 Embedder。

    局限:"语义相近"不等于"事实正确"(话题像但结论相反也会高分);
    实际用作判定时还需一个阈值,本 scorer 只给出连续相似度,阈值留给看数字的人。
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def score(self, answer_text: str, sample: EvalSample) -> float:
        va = self.embedder.embed_one(answer_text)
        vb = self.embedder.embed_one(sample.expected_answer)
        return max(0.0, _cosine(va, vb))  # 负相似度无意义,clamp 到 0


def get_scorer(name: str, llm: LLM | None = None,
               embedder: Embedder | None = None) -> AnswerScorer:
    """按名字返回评分器实现:keyword | llm_judge | semantic。"""
    if name == "keyword":
        return KeywordScorer()
    if name == "llm_judge":
        if llm is None:
            raise ValueError("llm_judge 评分器需要传入 llm")
        return LlmJudgeScorer(llm)
    if name == "semantic":
        if embedder is None:
            raise ValueError("semantic 评分器需要传入 embedder")
        return SemanticScorer(embedder)
    raise ValueError(f"未知评分器: {name}(可选 keyword | llm_judge | semantic)")
