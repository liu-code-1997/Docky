# Collections

A **collection** is a named set of points (vectors + payload) that all share
the same vector size and distance metric. You must create a collection before
inserting any points into it.

## Create a collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
```

- `size` is the dimensionality of your embeddings — it must match your
  embedding model's output (e.g. 768 for `nomic-embed-text`).
- `distance` is the similarity metric: `COSINE`, `DOT`, or `EUCLID`.

Both `size` and `distance` are **fixed at creation** and cannot be changed
later. To change them you must recreate the collection.

## Check if a collection exists

```python
if not client.collection_exists("my_docs"):
    client.create_collection(
        collection_name="my_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
```

This is the safe pattern for ingestion scripts: create only if missing, so
re-running doesn't error.

## Choosing a distance metric

- **Cosine**: compares direction, ignores magnitude. The most common choice for
  text embeddings.
- **Dot product**: fast; suitable when vectors are already normalized.
- **Euclidean**: straight-line distance; used in some image use cases.

If unsure, use Cosine for text.

## Delete or recreate

```python
client.delete_collection("my_docs")
```

Deleting removes all points. Because vector size is immutable, changing your
embedding model means deleting and recreating the collection, then re-ingesting.

## Recap

- A collection fixes vector size and distance metric at creation.
- Use `collection_exists` to create idempotently.
- Cosine is the usual metric for text embeddings.
