"""Qdrant 向量存储实现。

支持两种构造:
- location=":memory:" 用于测试(进程内,无需 Docker)。
- url="http://localhost:6333" 连真实 Qdrant 服务。

id 用确定性整数(对 chunk.id 做哈希)作为 Qdrant point id,
原始字符串 id 与全部元数据存进 payload。

M9: 使用命名向量 dense + sparse,支持 RRF 混合检索。
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseVector, Modifier,
    Prefetch, FusionQuery, Fusion,
)
from rag.interfaces import VectorStore
from rag.models import Chunk, RetrievedChunk

DENSE = "dense"
SPARSE = "sparse"


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
                vectors_config={DENSE: VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               sparse_vectors: list[tuple[list[int], list[float]]] | None = None) -> None:
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            vector: dict = {DENSE: vec}
            if sparse_vectors is not None:
                idx, val = sparse_vectors[i]
                vector[SPARSE] = SparseVector(indices=idx, values=val)
            points.append(PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload=chunk.model_dump(),
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _filter(self, library: str | None) -> Filter | None:
        if library is None:
            return None
        return Filter(must=[FieldCondition(key="library", match=MatchValue(value=library))])

    def search(self, query_vector: list[float], top_k: int,
               library: str | None = None) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=DENSE,
            limit=top_k,
            query_filter=self._filter(library),
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit.payload), score=hit.score)
            for hit in response.points
        ]

    def hybrid_search(self, query_vector: list[float],
                      sparse_query: tuple[list[int], list[float]],
                      top_k: int, library: str | None = None) -> list[RetrievedChunk]:
        idx, val = sparse_query
        qf = self._filter(library)
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=query_vector, using=DENSE, limit=top_k * 5, filter=qf),
                Prefetch(query=SparseVector(indices=idx, values=val), using=SPARSE,
                         limit=top_k * 5, filter=qf),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=qf,
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit.payload), score=hit.score)
            for hit in response.points
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def list_libraries(self) -> list[str]:
        # scroll 分页遍历所有点,只取 payload 里的 library 字段(不要向量,省带宽)。
        libraries: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["library"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                lib = (p.payload or {}).get("library")
                if lib:
                    libraries.add(lib)
            if offset is None:  # 没有下一页了
                break
        return sorted(libraries)
