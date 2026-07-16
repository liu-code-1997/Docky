"""可插拔接口:本地↔API、Qdrant↔其他库切换时,只换实现,不动核心逻辑。"""
from abc import ABC, abstractmethod
from rag.models import Chunk, RetrievedChunk, EvalSample


class Embedder(ABC):
    """把文本转成向量。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量把文本转成向量,返回顺序与输入一致。"""

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        """把单条文本转成向量。"""


class VectorStore(ABC):
    """向量存储与检索。"""

    @abstractmethod
    def ensure_collection(self, vector_size: int) -> None:
        """确保集合存在(不存在则创建)。"""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """写入块及其向量。"""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int,
               library: str | None = None) -> list[RetrievedChunk]:
        """按向量相似度检索,可选按 library 过滤。"""

    @abstractmethod
    def count(self) -> int:
        """返回集合内向量条数。"""

    @abstractmethod
    def list_libraries(self) -> list[str]:
        """返回库中所有不同的 library 名(去重、排序)。"""


class LLM(ABC):
    """大语言模型:根据 prompt 生成文本。"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """生成回答。"""


class AnswerScorer(ABC):
    """给一条答案打分(M4 评估)。返回 0–1,越高越接近期望答案。"""

    @abstractmethod
    def score(self, answer_text: str, sample: EvalSample) -> float:
        """对 answer_text 相对 sample 的期望打分。"""


class QueryRewriter(ABC):
    """检索前改写查询(M5 ②):如把中文问题扩展出英文术语,缓解跨语言检索。"""

    @abstractmethod
    def rewrite(self, question: str) -> str:
        """返回用于检索的查询文本(通常含原问题 + 扩展词)。"""


class Reranker(ABC):
    """检索后重排(M5 ③):对候选块按与问题的相关性重新排序,取前 top_k。"""

    @abstractmethod
    def rerank(self, question: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        """返回重排后的前 top_k 个候选。"""
