# Qdrant

Qdrant is an open-source **vector database** and vector similarity search
engine. It stores high-dimensional vectors (embeddings) together with their
metadata (called payload), and lets you find the vectors most similar to a
query vector.

## What Qdrant is for

Modern AI applications turn text, images, or audio into embeddings — lists of
floating point numbers that capture meaning. To find "similar" items you
compare these vectors. Qdrant is built to store millions of such vectors and
search them by similarity, fast.

Typical uses:

- **Semantic search** over documents (the retrieval step in a RAG system).
- **Recommendations** ("items similar to this one").
- **Deduplication** and near-duplicate detection.
- **Classification** by nearest neighbours.

## Key concepts

- **Collection**: a named set of points that share the same vector size and
  distance metric. Similar to a table in a relational database.
- **Point**: one record — an id, a vector, and an optional payload.
- **Vector**: the embedding, a fixed-length list of floats.
- **Payload**: arbitrary JSON metadata attached to a point (source file,
  category, timestamp), usable for filtering.
- **Distance**: how similarity is measured — Cosine, Dot, or Euclidean.

## Running Qdrant

The simplest way is Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Port 6333 serves the REST and web API; 6334 serves gRPC. Once running, open
`http://localhost:6333/dashboard` for the web UI.

## Recap

- Qdrant stores vectors + payload and searches by similarity.
- Data is organized into collections of points.
- Distance metric and vector size are fixed per collection.
