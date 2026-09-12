"""核心数据模型:跨各模块传递的最小单位。"""
from pydantic import BaseModel, model_validator


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
    """评估集的一条样本(M4;M7 升多来源)。"""
    question: str
    expected_sources: list[str] = []          # 相关来源文件;负例(拒答样本)为 []
    expected_keywords: list[str] = []         # 方法A:答案里应出现的关键词
    expected_answer: str = ""                 # 方法B/C:标准答案(A 用不到)

    @model_validator(mode="before")
    @classmethod
    def _compat_expected_source(cls, data):
        """兼容旧字段 expected_source(str|None)→ expected_sources。
        迁移完 dataset.json 后仍保留,以兜住任何遗留旧格式输入。"""
        if isinstance(data, dict) and "expected_source" in data:
            data = dict(data)
            src = data.pop("expected_source")
            data.setdefault("expected_sources", [src] if src else [])
        return data


# ---- M6:对话 / 工具调用抽象 ----

class ToolCall(BaseModel):
    """LLM 发起的一次工具调用请求。"""
    id: str                        # 调用 id,回填结果时用它对应
    name: str                      # 工具名,如 "search_docs"
    arguments: dict = {}           # 工具参数


class ToolSpec(BaseModel):
    """告诉 LLM 有哪个工具可用及其参数(JSON Schema)。"""
    name: str
    description: str
    parameters: dict               # JSON Schema 描述参数


class Message(BaseModel):
    """一条对话消息。role: system | user | assistant | tool。"""
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = []       # assistant 回合可能带一组工具调用
    tool_call_id: str | None = None       # role=tool 时,对应哪个 ToolCall


class ChatResponse(BaseModel):
    """ChatLLM 一个回合的产物:要么最终文本,要么一组待执行的工具调用。"""
    text: str = ""
    tool_calls: list[ToolCall] = []

    @property
    def is_final(self) -> bool:
        """无工具调用即为终态(该文本就是最终答案)。"""
        return not self.tool_calls
