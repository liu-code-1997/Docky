"""RagAgent 循环(M6):LLM 自主决定是否检索、检索几轮,再作答。"""
from rag.interfaces import ChatLLM
from rag.models import Chunk, RetrievedChunk, Message, ToolSpec, ChatResponse, ToolCall
from rag.agent import RagAgent


class FakeChatLLM(ChatLLM):
    """按预设脚本依次返回 ChatResponse,并记录每次收到的 messages。"""
    def __init__(self, script: list[ChatResponse]):
        self.script = list(script)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[ToolSpec] = []

    def chat(self, messages, tools):
        self.calls.append(list(messages))
        self.tools_seen = tools
        return self.script.pop(0)


class FakeStore:
    """search 返回预设的 RetrievedChunk,按 query 里是否含关键词决定给什么。"""
    def __init__(self, mapping):
        self.mapping = mapping           # query 子串 -> list[RetrievedChunk]
        self.searched = []

    def search(self, query_vector, top_k, library=None):
        return []                        # 不直接用;retrieve 走下面的 embedder+store

    # RagAgent 用 retrieve(),这里用一个直接可控的检索桩替代


class FakeRetriever:
    """替身:记录调用并按 query 返回结果,绕过真实 embedder/store。"""
    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def __call__(self, query, library=None, top_k=4):
        self.queries.append((query, library))
        for key, chunks in self.mapping.items():
            if key in query:
                return chunks
        return []


def _chunk(source, text):
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::0", text=text, source=source,
                    library=source.split("/")[0], chunk_index=0),
        score=0.9,
    )


def _tool_call(query, library=None):
    return ToolCall(id="c1", name="search_docs",
                    arguments={"query": query, "library": library})


def _make_fake_chat_llm_returning_text(text):
    """Helper: 构造一个返回给定文本的 FakeChatLLM。"""
    return FakeChatLLM([ChatResponse(text=text, tool_calls=[])])


def _agent(llm, retriever):
    # 用 retriever 替身注入,隔离真实检索
    return RagAgent(llm=llm, retriever=retriever, max_steps=5)


def test_single_turn_no_tool_call_returns_text_directly():
    # LLM 第一回合就给最终文本(不检索)
    llm = FakeChatLLM([ChatResponse(text="直接回答", tool_calls=[])])
    agent = _agent(llm, FakeRetriever({}))
    ans = agent.ask("你好")
    assert ans.text == "直接回答"
    assert ans.sources == []


def test_multi_turn_searches_then_answers():
    # 回合1:LLM 要求检索;回合2:据结果作答
    retriever = FakeRetriever({"path": [_chunk("fastapi/path.md", "路径参数用花括号")]})
    llm = FakeChatLLM([
        ChatResponse(text="", tool_calls=[_tool_call("path parameter", "fastapi")]),
        ChatResponse(text="路径参数用花括号声明", tool_calls=[]),
    ])
    agent = _agent(llm, retriever)
    ans = agent.ask("路径参数怎么写?")

    assert ans.text == "路径参数用花括号声明"
    assert ans.sources == ["fastapi/path.md"]         # 出处累积
    assert retriever.queries == [("path parameter", "fastapi")]  # 用了 LLM 给的 query


def test_sources_accumulate_and_dedupe_across_rounds():
    retriever = FakeRetriever({
        "q1": [_chunk("fastapi/a.md", "A")],
        "q2": [_chunk("fastapi/a.md", "A again"), _chunk("fastapi/b.md", "B")],
    })
    llm = FakeChatLLM([
        ChatResponse(text="", tool_calls=[_tool_call("q1")]),
        ChatResponse(text="", tool_calls=[_tool_call("q2")]),
        ChatResponse(text="综合答案", tool_calls=[]),
    ])
    agent = _agent(llm, retriever)
    ans = agent.ask("多轮问题")
    assert ans.text == "综合答案"
    assert ans.sources == ["fastapi/a.md", "fastapi/b.md"]   # 去重保序


def test_max_steps_stops_infinite_tool_loop():
    # LLM 每回合都要检索,永不收敛;应在 max_steps 处兜底停止
    retriever = FakeRetriever({"x": [_chunk("fastapi/a.md", "A")]})
    llm = FakeChatLLM([
        ChatResponse(text="", tool_calls=[_tool_call("x")]) for _ in range(10)
    ])
    agent = RagAgent(llm=llm, retriever=retriever, max_steps=3)
    ans = agent.ask("停不下来")
    # 兜底:调用 chat 次数不超过 max_steps
    assert len(llm.calls) <= 3
    # 仍返回一个 Answer(不崩溃),出处是已检索到的
    assert ans.sources == ["fastapi/a.md"]


def test_tools_are_passed_to_llm():
    llm = FakeChatLLM([ChatResponse(text="hi", tool_calls=[])])
    agent = _agent(llm, FakeRetriever({}))
    agent.ask("q")
    assert any(t.name == "search_docs" for t in llm.tools_seen)


def test_system_prompt_present_in_first_call():
    llm = FakeChatLLM([ChatResponse(text="hi", tool_calls=[])])
    agent = _agent(llm, FakeRetriever({}))
    agent.ask("q")
    first = llm.calls[0]
    assert first[0].role == "system"
    assert "无法回答" in first[0].content        # 拒答约束在


def test_recovers_tool_call_leaked_into_text():
    # 兜底(坑:本地 7B 有时把工具调用写进正文而非结构化字段)
    # 回合1:content 里混入 JSON 形式的工具调用,tool_calls 为空;
    # agent 应能识别并执行检索,而不是把它当最终答案。
    retriever = FakeRetriever({"path parameter": [_chunk("fastapi/path.md", "花括号")]})
    leaked = '好的,我来查一下。\n{"name": "search_docs", "arguments": {"query": "path parameter"}}'
    llm = FakeChatLLM([
        ChatResponse(text=leaked, tool_calls=[]),
        ChatResponse(text="路径参数用花括号", tool_calls=[]),
    ])
    agent = _agent(llm, retriever)
    ans = agent.ask("路径参数?")
    assert retriever.queries == [("path parameter", None)]   # 泄漏的调用被捞出并执行
    assert ans.sources == ["fastapi/path.md"]
    assert ans.text == "路径参数用花括号"


def test_agent_system_prompt_uses_injected_persona():
    from rag.agent import RagAgent
    agent = RagAgent(llm=_make_fake_chat_llm_returning_text("答"),
                     retriever=lambda q, library=None, top_k=4: [],
                     persona="你是保险顾问", refusal_text="资料没有")
    assert "你是保险顾问" in agent._system
    assert "资料没有" in agent._system
