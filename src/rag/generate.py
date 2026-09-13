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
        f"{persona}。请遵守以下规则:\n"
        f"1. 依据下面【资料】中的内容回答问题;只要资料里有相关信息,就据此作答,"
        f"不必因为信息不够完整就拒答。\n"
        f"2. 仅当【资料】与问题完全无关、找不到任何可用信息时,才回答\"{refusal_text}\";"
        f"任何情况下都不要编造、不要用资料之外的常识补充。\n"
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
