# Points and Upserting

A **point** is the basic record in Qdrant: an id, a vector, and an optional
payload. You add or update points with the `upsert` operation.

## Upsert points

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="my_docs",
    points=[
        PointStruct(
            id=1,
            vector=[0.05, 0.61, 0.76, ...],   # length must equal the collection's size
            payload={"source": "fastapi/index.md", "library": "fastapi"},
        ),
        PointStruct(
            id=2,
            vector=[0.19, 0.81, 0.75, ...],
            payload={"source": "qdrant/index.md", "library": "qdrant"},
        ),
    ],
)
```

`upsert` means insert-or-update: if a point with that id already exists it is
overwritten, otherwise it is created.

## Point ids

Ids can be **unsigned integers** or **UUIDs**. A useful trick for idempotent
ingestion is to derive a deterministic UUID from a stable string key, so that
re-ingesting the same document overwrites the same point instead of creating
duplicates:

```python
import uuid

def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
```

## Payload

The payload is arbitrary JSON attached to a point. Store whatever metadata you
need for filtering or for displaying results — source path, category, tags,
timestamps. Payload fields can later be used as search filters.

## Retrieve and delete

```python
# fetch specific points by id
client.retrieve(collection_name="my_docs", ids=[1, 2])

# delete by id
client.delete(collection_name="my_docs", points_selector=[1, 2])

# count points
client.count(collection_name="my_docs").count
```

## Scroll through all points

To iterate over every point (e.g. to list all distinct payload values), use
`scroll`, which paginates:

```python
points, next_page = client.scroll(
    collection_name="my_docs",
    with_payload=True,
    with_vectors=False,
    limit=256,
)
```

## Recap

- A point = id + vector + payload; add them with `upsert`.
- Use deterministic UUIDs for idempotent ingestion.
- Payload holds filterable metadata.
- Use `scroll` to walk the whole collection.
