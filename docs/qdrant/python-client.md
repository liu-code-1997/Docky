# Python Client

`qdrant-client` is the official Python library for talking to Qdrant. It wraps
the REST and gRPC APIs with a typed interface.

## Install

```bash
pip install qdrant-client
```

## Connect

```python
from qdrant_client import QdrantClient

# connect to a running server
client = QdrantClient(url="http://localhost:6333")
```

## In-memory mode for tests

For unit tests you can run Qdrant entirely in memory, with no server or Docker:

```python
client = QdrantClient(location=":memory:")
```

This is invaluable for fast, isolated tests — the whole store lives in the
process and disappears when it ends. Your ingestion and search code can run
against it exactly as against a real server.

## Typical workflow

```python
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(location=":memory:")

# 1. create a collection
client.create_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=3, distance=Distance.COSINE),
)

# 2. upsert points
client.upsert(
    collection_name="demo",
    points=[PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"tag": "a"})],
)

# 3. search
res = client.query_points(collection_name="demo", query=[0.1, 0.2, 0.25], limit=1)
print(res.points[0].payload)
```

## Version notes

APIs evolve. Two changes that commonly trip people up:

- `search()` was replaced by `query_points()` in 1.10+.
- Results are read from `response.points`.

When a method is missing, check the installed version and inspect the actual
signature rather than copying an old tutorial.

## Recap

- `QdrantClient(url=...)` for a server; `location=":memory:"` for tests.
- Workflow: create collection → upsert points → `query_points`.
- Prefer in-memory mode for fast, dependency-free tests.
