"""OpenAI 兼容 ChatLLM(覆盖 OpenAI/DeepSeek/Groq/vLLM/Ollama 的 /v1 端点)。"""
import json
import httpx
from rag.models import Message, ToolSpec
from rag.providers.openai_chat import OpenAICompatChatLLM


TOOLS = [ToolSpec(name="search_docs", description="检索文档",
                  parameters={"type": "object", "properties": {}})]


def test_chat_parses_final_text(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        assert "/v1/chat/completions" in url
        assert json["model"] == "qwen2.5:7b"
        # tools 以 OpenAI 格式传递
        assert json["tools"][0]["function"]["name"] == "search_docs"
        assert headers["Authorization"] == "Bearer sk-test"
        body = {"choices": [{"message": {"role": "assistant",
                                         "content": "最终答案", "tool_calls": None}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OpenAICompatChatLLM(base_url="http://x/v1", model="qwen2.5:7b",
                              api_key="sk-test")
    resp = llm.chat([Message(role="user", content="hi")], TOOLS)
    assert resp.text == "最终答案"
    assert resp.is_final is True


def test_chat_parses_tool_calls(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        body = {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "search_docs",
                             "arguments": json_dumps({"query": "path params",
                                                      "library": "fastapi"})},
            }],
        }}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    # 用真实 json.dumps 造 arguments 字符串(OpenAI 的 arguments 是 JSON 字符串)
    def json_dumps(d):
        return json.dumps(d)

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OpenAICompatChatLLM(base_url="http://x/v1", model="m", api_key="k")
    resp = llm.chat([Message(role="user", content="q")], TOOLS)

    assert resp.is_final is False
    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].name == "search_docs"
    assert resp.tool_calls[0].arguments == {"query": "path params",
                                            "library": "fastapi"}


def test_chat_serializes_conversation_including_tool_messages(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["messages"] = json["messages"]
        body = {"choices": [{"message": {"content": "ok", "tool_calls": None}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OpenAICompatChatLLM(base_url="http://x/v1", model="m", api_key="k")
    msgs = [
        Message(role="system", content="rules"),
        Message(role="user", content="q"),
        Message(role="tool", content="[结果]", tool_call_id="call_1"),
    ]
    llm.chat(msgs, TOOLS)

    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "tool"]
    # tool 消息带上 tool_call_id
    assert seen["messages"][2]["tool_call_id"] == "call_1"


def test_chat_raises_on_http_error(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return httpx.Response(500, json={"error": "boom"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OpenAICompatChatLLM(base_url="http://x/v1", model="m", api_key="k")
    import pytest
    with pytest.raises(httpx.HTTPStatusError):
        llm.chat([Message(role="user", content="q")], TOOLS)
