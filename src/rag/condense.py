"""把依赖上文的追问改写成可独立检索的问题(M12)。"""
from rag.interfaces import LLM
from rag.models import Message

_CONDENSE_PROMPT = """给定下面的【对话历史】和用户的【追问】,把追问改写成一个脱离历史也能\
独立理解与检索的完整问题。只输出改写后的问题,不要解释、不要加引号。

【对话历史】
{history}

【追问】{question}
【独立问题】"""


def _render(history: list[Message]) -> str:
    role = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role.get(m.role, m.role)}:{m.content}" for m in history)


def condense_question(llm: LLM, question: str, history: list[Message]) -> str:
    if not history:
        return question
    reply = llm.generate(_CONDENSE_PROMPT.format(
        history=_render(history), question=question)).strip()
    return reply or question
