"""RagAgent(M6):把"检索一次→生成一次"升级为 LLM 自主决策的循环。

LLM 每回合自己决定:要不要检索、检索够不够、要不要换 query 再查一轮,
最后据资料作答。检索作为一个工具 search_docs 提供,内部就是现有 retrieve 链路。

依赖注入:agent 只依赖一个 `retriever` 可调用(query, library, top_k) -> chunks,
而非直接持有 embedder/store。这样测试可注入替身,生产注入真实 retrieve 的偏函数,
agent 循环对底层检索实现无感知。
"""
import json
import re

from rag.interfaces import ChatLLM
from rag.models import Answer, Message, ToolSpec, ToolCall, RetrievedChunk


_DEFAULT_PERSONA = "你是一个严谨的技术文档问答助手"
_DEFAULT_REFUSAL = "根据现有资料无法回答"


def _build_system(persona: str, refusal_text: str) -> str:
    """构造系统提示词。

    Args:
        persona: 角色描述,如"你是一个严谨的技术文档问答助手"
        refusal_text: 拒答文本,如"根据现有资料无法回答"

    Returns:
        完整的系统提示词字符串
    """
    return (
        f"{persona},可以使用 search_docs 工具检索文档。\n\n"
        f"工作方式:\n"
        f"- 需要资料时,**用结构化的工具调用**触发 search_docs,不要把工具调用当文本写进正文。\n"
        f"- 若结果不够好,可以换个查询词再检索一轮。\n"
        f"- 收集到足够资料后,只依据检索到的资料回答。\n"
        f"- 如果检索不到相关资料,必须回答\"{refusal_text}\",不要编造。\n"
        f"- 回答尽量简洁、准确。"
    )


SEARCH_DOCS = ToolSpec(
    name="search_docs",
    description="在技术文档知识库中检索与查询最相关的片段。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索用的查询词,可用英文术语提高命中"},
            "library": {"type": "string",
                        "description": "限定库,如 fastapi/qdrant/mysql;不限定则省略"},
        },
        "required": ["query"],
    },
)


def _recover_tool_calls(text: str) -> list[ToolCall]:
    """兜底:从正文里捞出被当成文本写出来的工具调用。

    本地小模型有时不走结构化 tool_calls,而是把 {"name":..,"arguments":..}
    (可能包在 <tool_call> 标签里)直接写进 content。这里用正则找出这类 JSON。
    """
    calls: list[ToolCall] = []
    # 匹配含 "name" 和 "arguments" 的 JSON 对象
    for m in re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if name and isinstance(args, dict):
            calls.append(ToolCall(id=f"recovered_{len(calls)}", name=name,
                                  arguments=args))
    return calls


class RagAgent:
    def __init__(self, llm: ChatLLM, retriever, top_k: int = 4, max_steps: int = 5,
                 persona: str = _DEFAULT_PERSONA,
                 refusal_text: str = _DEFAULT_REFUSAL):
        self.llm = llm
        self.retriever = retriever      # callable(query, library=None, top_k=int) -> list[RetrievedChunk]
        self.top_k = top_k
        self.max_steps = max_steps
        self._system = _build_system(persona, refusal_text)

    def ask(self, question: str, library: str | None = None) -> Answer:
        messages: list[Message] = [
            Message(role="system", content=self._system),
            Message(role="user", content=question),
        ]
        sources: list[str] = []         # 累积、去重、保序

        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, [SEARCH_DOCS])

            # 兜底:本地小模型有时把工具调用写进正文而非结构化字段,尝试捞出来
            tool_calls = resp.tool_calls or _recover_tool_calls(resp.text)

            if not tool_calls:
                return Answer(text=resp.text, sources=sources)

            # 记录助手这一回合的工具调用
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=tool_calls))
            # 执行每个工具调用,把结果回填为 tool 消息
            for tc in tool_calls:
                chunks = self._run_tool(tc.name, tc.arguments)
                for rc in chunks:
                    if rc.chunk.source not in sources:
                        sources.append(rc.chunk.source)
                messages.append(Message(
                    role="tool", tool_call_id=tc.id,
                    content=self._format_chunks(chunks)))

        # 兜底:到 max_steps 仍未收敛(防死循环),用已检索资料诚实收场
        return Answer(
            text="根据现有资料无法回答(超出最大检索轮数)。",
            sources=sources)

    def _run_tool(self, name: str, args: dict) -> list[RetrievedChunk]:
        if name == "search_docs":
            query = args.get("query", "")
            lib = args.get("library")
            return self.retriever(query, library=lib, top_k=self.top_k)
        return []                       # 未知工具:返回空,不崩溃

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(无相关资料)"
        return "\n\n".join(
            f"[来源:{rc.chunk.source}]\n{rc.chunk.text}" for rc in chunks)
