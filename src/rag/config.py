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

    # 检索参数(M2 使用)
    top_k: int = 4

    # 评估参数(M4 使用):生成层评分方法 keyword | llm_judge | semantic
    eval_scorer: str = "keyword"

    # 生成温度:日常问答略高更自然;评估恒为 0,同 prompt 恒定输出、结果可复现
    llm_temperature: float = 0.7
    eval_temperature: float = 0.0


def get_settings() -> Settings:
    """返回一个 Settings 实例。集中在此,方便测试时替换。"""
    return Settings()
