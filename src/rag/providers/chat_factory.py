"""ChatLLM 工厂(M6):按配置选具体 provider。

这是"选哪家 provider"的唯一决策点(composition root)。agent 循环只依赖
ChatLLM 接口,对具体厂商无感知——换厂商只改配置,不改循环代码。
"""
import os

from rag.config import Settings
from rag.interfaces import ChatLLM
from rag.providers.openai_chat import OpenAICompatChatLLM


def build_chat_llm(settings: Settings) -> ChatLLM:
    provider = settings.chat_provider

    if provider == "openai_compat":
        # 密钥从环境变量读(名字由 chat_api_key_env 指定);本地 Ollama 无需 key
        api_key = os.environ.get(settings.chat_api_key_env, "")
        return OpenAICompatChatLLM(
            base_url=settings.chat_base_url,
            model=settings.chat_model,
            api_key=api_key,
        )

    if provider == "claude":
        # 延迟导入:只有真用 claude 时才需要 anthropic SDK
        from rag.providers.claude_chat import ClaudeChatLLM
        return ClaudeChatLLM(
            model=settings.claude_model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    raise ValueError(
        f"未知 chat_provider: {provider}(可选 openai_compat | claude)")
