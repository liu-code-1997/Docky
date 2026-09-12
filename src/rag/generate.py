"""Generate 链路:资料 + 问题 → Prompt → LLM → 带出处的答案。

防幻觉是这里的核心(notes/05):
- 强约束 system 段:只能依据下面的资料回答。
- 显式拒答:资料里没有答案就回答"根据现有资料无法回答",不要编造。
- 带出处:答案附上去重后的来源文件列表。
"""
from rag.interfaces import LLM
from rag.models import Answer, RetrievedChunk


_DEFAULT_PERSONA = "你是一个严谨的技术文档问答助手"
_DEFAULT_REFUSAL = "根据现有资料无法回答"


def build_prompt(question: str, chunks: list[RetrievedChunk],
                 persona: str = _DEFAULT_PERSONA,
                 refusal_text: str = _DEFAULT_REFUSAL) -> str:
    """把 Top-K 资料与问题组装成给 LLM 的完整 prompt。"""
    system = (
        f"{persona}。请严格遵守以下规则:\n"
        f"1. 只能依据下面【资料】中的内容回答问题。\n"
        f"2. 如果【资料】中没有足够信息回答,必须回答\"{refusal_text}\","
        f"不要编造、不要凭常识补充。\n"
        f"3. 回答尽量简洁、准确,可引用资料中的术语。"
    )
    if chunks:
        blocks = [f"[资料{i} | 来源:{rc.chunk.source}]\n{rc.chunk.text}"
                  for i, rc in enumerate(chunks, start=1)]
        context = "\n\n".join(blocks)
    else:
        context = "(无相关资料)"
    return (f"{system}\n\n【资料】\n{context}\n\n【问题】\n{question}\n\n【回答】")


def answer(question: str, chunks: list[RetrievedChunk], llm: LLM,
           persona: str = _DEFAULT_PERSONA,
           refusal_text: str = _DEFAULT_REFUSAL) -> Answer:
    """生成答案,并附上去重(保序)后的来源列表。"""
    prompt = build_prompt(question, chunks, persona, refusal_text)
    text = llm.generate(prompt)

    sources: list[str] = []
    for rc in chunks:
        if rc.chunk.source not in sources:
            sources.append(rc.chunk.source)

    return Answer(text=text, sources=sources)
