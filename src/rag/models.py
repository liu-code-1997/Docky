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


class Answer(BaseModel):
    """问答的最终产物:答案文本 + 去重后的来源列表。"""
    text: str                          # LLM 生成的答案
    sources: list[str] = []            # 去重后的来源 source(无资料时为空)


class EvalSample(BaseModel):
    """评估集的一条样本(M4)。"""
    question: str
    expected_source: str | None = None       # 期望命中的来源文件;无答案样本为 None
    expected_keywords: list[str] = []         # 方法A:答案里应出现的关键词
    expected_answer: str = ""                 # 方法B/C:标准答案(A 用不到)
