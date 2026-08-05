"""build_chat_llm 工厂:按配置选 provider,是"选哪家只发生在一处"的装配点。"""
import pytest
from rag.config import Settings
from rag.providers.chat_factory import build_chat_llm
from rag.providers.openai_chat import OpenAICompatChatLLM


def test_builds_openai_compat_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    s = Settings(chat_provider="openai_compat", chat_base_url="http://x/v1",
                 chat_model="qwen2.5:7b", chat_api_key_env="OPENAI_API_KEY")
    llm = build_chat_llm(s)
    assert isinstance(llm, OpenAICompatChatLLM)
    assert llm.base_url == "http://x/v1"
    assert llm.model == "qwen2.5:7b"
    assert llm.api_key == "sk-abc"        # 从环境变量读到


def test_openai_compat_missing_key_is_ok_for_local(monkeypatch):
    # 本地 Ollama 无需 key:环境变量不存在时,api_key 为空字符串而非报错
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(chat_provider="openai_compat", chat_api_key_env="OPENAI_API_KEY")
    llm = build_chat_llm(s)
    assert llm.api_key == ""


def test_unknown_provider_raises():
    s = Settings(chat_provider="nonsense")
    with pytest.raises(ValueError):
        build_chat_llm(s)
