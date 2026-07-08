"""Generate 链路:资料 + 问题 → Prompt → LLM → 带出处的答案。

防幻觉是这里的核心(notes/05):
- 强约束 system 段:只能依据下面的资料回答。
- 显式拒答:资料里没有答案就回答"根据现有资料无法回答",不要编造。
- 带出处:答案附上去重后的来源文件列表。
"""
from rag.interfaces import LLM
from rag.models import Answer, RetrievedChunk


_SYSTEM = """你是一个严谨的技术文档问答助手。请严格遵守以下规则:
1. 只能依据下面【资料】中的内容回答问题。
2. 如果【资料】中没有足够信息回答,必须回答"根据现有资料无法回答",不要编造、不要凭常识补充。
3. 回答尽量简洁、准确,可引用资料中的术语。"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """把 Top-K 资料与问题组装成给 LLM 的完整 prompt。"""
    if chunks:
        blocks = []
        for i, rc in enumerate(chunks, start=1):
            blocks.append(f"[资料{i} | 来源:{rc.chunk.source}]\n{rc.chunk.text}")
        context = "\n\n".join(blocks)
    else:
        context = "(无相关资料)"

    return (
        f"{_SYSTEM}\n\n"
        f"【资料】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        f"【回答】"
    )


def answer(question: str, chunks: list[RetrievedChunk], llm: LLM) -> Answer:
    """生成答案,并附上去重(保序)后的来源列表。"""
    prompt = build_prompt(question, chunks)
    text = llm.generate(prompt)

    sources: list[str] = []
    for rc in chunks:
        if rc.chunk.source not in sources:
            sources.append(rc.chunk.source)

    return Answer(text=text, sources=sources)
