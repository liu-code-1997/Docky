# Similarity Search

The core operation in Qdrant is finding the points whose vectors are most
similar to a query vector. In current versions of `qdrant-client` this is done
with `query_points`.

## Basic search

```python
response = client.query_points(
    collection_name="my_docs",
    query=[0.2, 0.1, 0.9, ...],   # the query vector
    limit=4,                       # return top 4 most similar
)

for point in response.points:
    print(point.id, point.score, point.payload)
```

`response.points` is the list of hits, ordered from most to least similar.
Each hit has an `id`, a `score` (the similarity), and the `payload`.

## query_points replaces the old search()

Older tutorials use `client.search(...)`. That method was deprecated and
removed; `qdrant-client` 1.10+ uses `query_points(...)`, and the results are
under `response.points` rather than being returned directly. If you see
`AttributeError: 'QdrantClient' object has no attribute 'search'`, switch to
`query_points`.

## Getting a query vector

You don't search with text directly — you first turn your query text into a
vector with the **same embedding model** used at ingestion time, then pass that
vector to `query_points`. Using a different model would put the query in a
different vector space and make scores meaningless.

## How many results

The `limit` parameter is your top-k. Larger k improves recall (more chance the
right document is included) but adds noise. A common range for RAG is 4–8;
tune it against an evaluation set rather than guessing.

## Scores

The `score` meaning depends on the collection's distance metric. For Cosine,
higher is more similar (max 1.0). Scores are only comparable within the same
collection and metric.

## Recap

- Search with `query_points`; read hits from `response.points`.
- Embed the query with the same model used for ingestion.
- `limit` is top-k; tune it with evaluation.
