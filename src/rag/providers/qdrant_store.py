"""Qdrant 向量存储实现。

支持两种构造:
- location=":memory:" 用于测试(进程内,无需 Docker)。
- url="http://localhost:6333" 连真实 Qdrant 服务。

id 用确定性整数(对 chunk.id 做哈希)作为 Qdrant point id,
原始字符串 id 与全部元数据存进 payload。
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from rag.interfaces import VectorStore
from rag.models import Chunk, RetrievedChunk


def _point_id(chunk_id: str) -> str:
    # 用确定性 UUID5,保证同一 chunk.id 重复 ingest 时覆盖而非重复插入
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore(VectorStore):
    def __init__(self, collection_name: str,
                 url: str | None = None, location: str | None = None):
        if location is not None:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int,
               library: str | None = None) -> list[RetrievedChunk]:
        query_filter = None
        if library is not None:
            query_filter = Filter(must=[
                FieldCondition(key="library", match=MatchValue(value=library))
            ])
        # 新版 qdrant-client(1.10+)用 query_points 取代已废弃的 search。
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit.payload), score=hit.score)
            for hit in response.points
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count
