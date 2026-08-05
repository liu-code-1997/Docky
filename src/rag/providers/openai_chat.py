"""OpenAI 兼容的 ChatLLM 实现(M6)。

一个类覆盖所有走 /v1/chat/completions 的服务:OpenAI、DeepSeek、Groq、
本地 vLLM,以及 Ollama(它也暴露 /v1 兼容端点)。换厂商只改 base_url/model/key。

职责:在"厂商中立的 Message/ToolSpec/ChatResponse"与"OpenAI 线格式"之间来回转换,
把协议差异全挡在这里,让 agent 循环对厂商无感知。
"""
import json

import httpx

from rag.interfaces import ChatLLM
from rag.models import Message, ToolSpec, ToolCall, ChatResponse


def _message_to_wire(m: Message) -> dict:
    """把内部 Message 转成 OpenAI messages 元素。"""
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    return d


def _tool_to_wire(t: ToolSpec) -> dict:
    return {"type": "function",
            "function": {"name": t.name, "description": t.description,
                         "parameters": t.parameters}}


class OpenAICompatChatLLM(ChatLLM):
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 120.0, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def chat(self, messages: list[Message],
             tools: list[ToolSpec]) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [_tool_to_wire(t) for t in tools]

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = httpx.post(f"{self.base_url}/chat/completions",
                          json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return _parse_message(msg)


def _parse_message(msg: dict) -> ChatResponse:
    """把 OpenAI 响应里的 assistant message 解析成 ChatResponse。"""
    raw_calls = msg.get("tool_calls") or []
    tool_calls = []
    for c in raw_calls:
        fn = c["function"]
        args = fn.get("arguments") or "{}"
        # OpenAI 的 arguments 是 JSON 字符串;解析失败则退回空参数
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed = {}
        tool_calls.append(ToolCall(id=c.get("id", ""), name=fn["name"],
                                   arguments=parsed))
    return ChatResponse(text=msg.get("content") or "", tool_calls=tool_calls)
