"""M6:对话/工具调用的数据模型 + ChatLLM 接口契约。"""
import pytest
from rag.models import Message, ToolSpec, ToolCall, ChatResponse
from rag.interfaces import ChatLLM


def test_message_basic():
    m = Message(role="user", content="你好")
    assert m.role == "user"
    assert m.content == "你好"
    assert m.tool_calls == []          # 默认无工具调用
    assert m.tool_call_id is None


def test_message_assistant_with_tool_calls():
    tc = ToolCall(id="call_1", name="search_docs",
                  arguments={"query": "path params", "library": "fastapi"})
    m = Message(role="assistant", content="", tool_calls=[tc])
    assert m.tool_calls[0].name == "search_docs"
    assert m.tool_calls[0].arguments["query"] == "path params"


def test_tool_message_carries_call_id():
    # 工具执行结果回填时,要带上对应的 tool_call_id
    m = Message(role="tool", content="[结果]", tool_call_id="call_1")
    assert m.role == "tool"
    assert m.tool_call_id == "call_1"


def test_tool_spec_shape():
    spec = ToolSpec(
        name="search_docs",
        description="检索技术文档",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    assert spec.name == "search_docs"
    assert "query" in spec.parameters["properties"]


def test_chat_response_final_text():
    r = ChatResponse(text="最终答案", tool_calls=[])
    assert r.text == "最终答案"
    assert r.tool_calls == []
    assert r.is_final is True           # 无 tool_calls => 终态


def test_chat_response_with_tool_calls_is_not_final():
    tc = ToolCall(id="c1", name="search_docs", arguments={"query": "x"})
    r = ChatResponse(text="", tool_calls=[tc])
    assert r.is_final is False          # 有 tool_calls => 还要继续循环


def test_chatllm_is_abstract():
    # 接口不能直接实例化,必须有实现
    with pytest.raises(TypeError):
        ChatLLM()
