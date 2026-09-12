"""集中式配置。所有模型名、服务地址、切分参数都在这里,便于后续切换。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 服务地址
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"

    # 模型
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"

    # Qdrant
    collection_name: str = "rag_docs"

    # 切分参数(M1 使用)
    chunk_size: int = 800
    chunk_overlap: int = 150

    # 切分策略(M5 ①):char(按字符,baseline) | markdown(按标题+噪声过滤)
    chunk_strategy: str = "char"

    # M8:领域 profile(profiles/<name>.toml),换领域=换名字
    profile: str = "tech"

    # 查询改写(M5 ②):检索前用 LLM 扩展英文术语,缓解跨语言检索。
    # 经消融验证是最有效手段(hit@4 39%→94%),故默认开启。
    query_rewrite: bool = True

    # 重排(M5 ③):开启后先召回 top_k×rerank_factor 候选,用 LLM 重排取前 top_k
    rerank: bool = False
    rerank_factor: int = 5

    # 检索参数(M2 使用)
    top_k: int = 4

    # 评估参数(M4 使用):生成层评分方法 keyword | llm_judge | semantic
    eval_scorer: str = "keyword"

    # 生成温度:日常问答略高更自然;评估恒为 0,同 prompt 恒定输出、结果可复现
    llm_temperature: float = 0.7
    eval_temperature: float = 0.0

    # ---- M6 agent 的对话 LLM(function calling)----
    # provider: openai_compat(覆盖 OpenAI/DeepSeek/Groq/vLLM/Ollama)| claude
    chat_provider: str = "openai_compat"
    # openai_compat 三旋钮:换厂商/模型/地址只改这几行(密钥从环境变量读)
    chat_base_url: str = "http://localhost:11434/v1"   # 默认本地 Ollama 的 /v1 端点
    chat_model: str = "qwen2.5:7b"
    chat_api_key_env: str = "OPENAI_API_KEY"           # 从哪个环境变量读 key(本地可空)
    # claude(chat_provider=claude 时生效;ANTHROPIC_API_KEY 从环境变量读)
    claude_model: str = "claude-opus-5"
    # agent 循环兜底上限
    agent_max_steps: int = 5


def get_settings() -> Settings:
    """返回一个 Settings 实例。集中在此,方便测试时替换。"""
    return Settings()
