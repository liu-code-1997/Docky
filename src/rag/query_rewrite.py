"""查询改写(M5 ②):检索前把问题扩展出可能的英文术语,缓解跨语言检索。

场景:问题是中文、文档是英文,embedding 跨语言匹配吃亏。让 LLM 给出问题里
关键概念的英文术语,拼到原问题后一起检索,提高召回。复用现有 LLM 接口。

局限:多一次 LLM 调用(变慢);改写本身也可能跑偏。默认关闭,按开关启用。
"""
from rag.interfaces import LLM, QueryRewriter


_REWRITE_PROMPT = """你在为技术文档检索改写查询。文档是英文的。
请针对下面的问题,只输出最相关的英文关键词/术语(空格分隔,不要解释、不要标点):

问题:{question}
英文关键词:"""


class LlmQueryRewriter(QueryRewriter):
    def __init__(self, llm: LLM):
        self.llm = llm

    def rewrite(self, question: str) -> str:
        expansion = self.llm.generate(_REWRITE_PROMPT.format(question=question)).strip()
        if not expansion:
            return question.strip()  # 改写没产出时,退回原问题
        # 拼上原问题:即便改写跑偏,原词仍在,不至于比不改写更差
        return f"{expansion} {question}".strip()
