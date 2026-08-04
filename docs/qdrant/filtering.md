# Filtering

Qdrant can combine vector similarity search with **payload filters**, so you
only search among points whose metadata matches certain conditions. This is
called filtered search.

## Filter by a payload field

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

response = client.query_points(
    collection_name="my_docs",
    query=query_vector,
    limit=4,
    query_filter=Filter(
        must=[FieldCondition(key="library", match=MatchValue(value="fastapi"))]
    ),
)
```

This returns the 4 most similar points **among those where `library ==
"fastapi"`**. Points from other libraries are excluded before ranking.

## must, should, must_not

A `Filter` combines conditions:

- `must`: all conditions must match (logical AND).
- `should`: at least one should match (logical OR, boosts matching).
- `must_not`: none of these may match (logical NOT).

```python
Filter(
    must=[FieldCondition(key="library", match=MatchValue(value="fastapi"))],
    must_not=[FieldCondition(key="draft", match=MatchValue(value=True))],
)
```

## Range conditions

For numeric payload you can filter by range:

```python
from qdrant_client.models import Range

Filter(
    must=[FieldCondition(key="year", range=Range(gte=2020, lte=2024))]
)
```

## Why filtering matters for RAG

In a multi-source knowledge base, filtering lets a query stay within one
document set — e.g. only search the `fastapi` docs when the user asks a FastAPI
question. This improves precision and avoids cross-topic noise.

## Recap

- Combine similarity search with `query_filter`.
- Use `must` / `should` / `must_not` to build conditions.
- `Range` filters numeric fields.
- Filtering keeps a query within the relevant subset of the collection.
