"""核心数据模型:跨各模块传递的最小单位。"""
from pydantic import BaseModel


class Chunk(BaseModel):
    """一个文档块:存储与检索的基本单位。"""
    id: str           # 全局唯一,形如 "fastapi/index.md::0"
    text: str         # 块的原文
    source: str       # 来源文件相对路径
    library: str      # 所属库,如 "fastapi" / "qdrant"(用于元数据过滤)
    chunk_index: int  # 在原文件中的第几块


class RetrievedChunk(BaseModel):
    """检索返回的块,附带相似度分数。"""
    chunk: Chunk
    score: float
