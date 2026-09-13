# M9 多路召回 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 加稀疏(BM25 式)召回 + Qdrant 原生 RRF 融合,补稠密向量在精确术语上的短板,抬升 recall@4(基线 0.936)。

**Architecture:** 纯 Python 稀疏编码器(零依赖,crc32 稳定哈希)。QdrantStore 迁移到命名向量 dense+sparse;新增 hybrid_search(Prefetch 双路 + Fusion.RRF)。retrieve/ingest/pipeline/scripts 加 hybrid 开关(默认关=dense-only,行为等价现状)。

**Tech Stack:** Python 3.11+、qdrant-client 1.18(已装,原生 sparse+RRF)、pytest。**零新依赖。**

**Spec:** `specs/2026-09-13-m9-hybrid-recall-design.md`

## Global Constraints
- 测试用 `.venv/bin/python -m pytest`(系统 python3=3.9 会挂)。零新依赖。
- 稀疏哈希**必须稳定**(`zlib.crc32`),禁用内置 `hash()`。
- hybrid 默认 False;dense-only 走命名 "dense",行为等价现状。
- 命名向量是 schema 变更 → 代码落地后 live 库须重灌(Task 6);单测用 `:memory:`(spike 已证支持 sparse+RRF)。
- conventional commits;可测逻辑在 src/rag,scripts 仅装配。预存 StarletteDeprecationWarning 忽略。

---

### Task 1: 稀疏编码器 src/rag/sparse.py

**Files:** Create `src/rag/sparse.py`, `tests/test_sparse.py`

**Interfaces:** Produces `tokenize(text)->list[str]`;`encode_sparse(text)->tuple[list[int],list[float]]`(crc32 稳定哈希到 2**20,值=词频)。

- [ ] **Step 1: 写失败测试** `tests/test_sparse.py`:

```python
from rag.sparse import tokenize, encode_sparse


def test_tokenize_keeps_identifiers_lowercases():
    assert tokenize("Use item_id and LEFT JOIN") == ["use", "item_id", "and", "left", "join"]


def test_encode_sparse_counts_term_frequency():
    idx, val = encode_sparse("join join index")
    assert len(idx) == len(val) == 2            # 两个不同词
    assert sorted(val) == [1.0, 2.0]            # join=2, index=1


def test_encode_sparse_is_deterministic_across_calls():
    a = encode_sparse("query_points limit filter")
    b = encode_sparse("query_points limit filter")
    assert a == b                                # 稳定哈希:两次完全一致


def test_encode_sparse_empty():
    assert encode_sparse("") == ([], [])
```

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_sparse.py -v` → FAIL
- [ ] **Step 3: 实现** `src/rag/sparse.py`:

```python
"""稀疏(BM25 式词形)向量编码 —— 纯 Python 零依赖。

用 zlib.crc32 稳定哈希(禁用内置 hash():它每进程随机,会让灌库与检索对不上)。
值=词频,IDF 交给 Qdrant 服务端的 Modifier.IDF。
"""
import re
import zlib
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_SPACE = 2 ** 20


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def encode_sparse(text: str) -> tuple[list[int], list[float]]:
    counts: Counter[int] = Counter()
    for tok in tokenize(text):
        counts[zlib.crc32(tok.encode("utf-8")) % _SPACE] += 1
    indices = list(counts.keys())
    values = [float(counts[i]) for i in indices]
    return indices, values
```

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_sparse.py -v` → PASS(4)
- [ ] **Step 5: 提交** `git add src/rag/sparse.py tests/test_sparse.py && git commit -m "feat: 纯 Python 稀疏编码器(crc32 稳定哈希,零依赖)"`

---

### Task 2: QdrantStore 命名向量 + hybrid_search

**Files:** Modify `src/rag/interfaces.py`, `src/rag/providers/qdrant_store.py`, `tests/test_qdrant_store.py`

**Interfaces:**
- `VectorStore.upsert(chunks, vectors, sparse_vectors=None)`(sparse_vectors: `list[tuple[list[int],list[float]]] | None`)
- `VectorStore.search(query_vector, top_k, library=None)`(查命名 "dense",行为等价)
- 新增 `VectorStore.hybrid_search(query_vector, sparse_query, top_k, library=None)`(sparse_query: `tuple[list[int],list[float]]`)
- `ensure_collection(vector_size)` 建命名 dense + sparse

- [ ] **Step 1: 写失败测试** 把 `tests/test_qdrant_store.py` 改造为命名向量,并新增 hybrid 测试(用 `:memory:`):

```python
from qdrant_client import QdrantClient  # noqa (若已 import 则复用)
from rag.providers.qdrant_store import QdrantStore
from rag.models import Chunk


def _store():
    s = QdrantStore(collection_name="t", location=":memory:")
    s.ensure_collection(vector_size=4)
    return s


def _chunk(cid, lib="l"):
    return Chunk(id=cid, text="t", source=f"{cid}.md", library=lib, chunk_index=0)


def test_named_upsert_and_dense_search():
    s = _store()
    s.upsert([_chunk("a"), _chunk("b")],
             [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([10, 20], [1.0, 2.0]), ([20, 30], [3.0, 1.0])])
    hits = s.search([1, 0, 0, 0], top_k=2)
    assert hits[0].chunk.source == "a.md"        # dense 最近的排第一
    assert s.count() == 2


def test_hybrid_search_rrf_returns_results():
    s = _store()
    s.upsert([_chunk("a"), _chunk("b")],
             [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([10, 20], [1.0, 2.0]), ([20, 30], [3.0, 1.0])])
    hits = s.hybrid_search([1, 0, 0, 0], ([20], [1.0]), top_k=2)
    assert {h.chunk.source for h in hits} == {"a.md", "b.md"}  # 两路融合都覆盖


def test_library_filter_still_applies():
    s = _store()
    s.upsert([_chunk("a", "x"), _chunk("b", "y")], [[1, 0, 0, 0], [0, 1, 0, 0]],
             sparse_vectors=[([1], [1.0]), ([2], [1.0])])
    hits = s.search([1, 0, 0, 0], top_k=5, library="x")
    assert all(h.chunk.library == "x" for h in hits)
```

(删除/改写旧的未命名向量测试。)

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_qdrant_store.py -v` → FAIL

- [ ] **Step 3: 实现**

`interfaces.py` `VectorStore`:改 `upsert` 签名加 `sparse_vectors=None`;新增抽象 `hybrid_search`:

```python
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               sparse_vectors: list[tuple[list[int], list[float]]] | None = None) -> None:
        """写入块及其稠密向量(可选稀疏向量)。"""

    @abstractmethod
    def hybrid_search(self, query_vector: list[float],
                      sparse_query: tuple[list[int], list[float]],
                      top_k: int, library: str | None = None) -> list[RetrievedChunk]:
        """稠密+稀疏双路召回,RRF 融合取前 top_k。"""
```

`qdrant_store.py`:

```python
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseVector, Modifier, Prefetch, FusionQuery, Fusion,
)

DENSE = "dense"
SPARSE = "sparse"


def ensure_collection(self, vector_size: int) -> None:
    if not self.client.collection_exists(self.collection_name):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={DENSE: VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
        )


def upsert(self, chunks, vectors, sparse_vectors=None):
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        vector = {DENSE: vec}
        if sparse_vectors is not None:
            idx, val = sparse_vectors[i]
            vector[SPARSE] = SparseVector(indices=idx, values=val)
        points.append(PointStruct(id=_point_id(chunk.id), vector=vector,
                                   payload=chunk.model_dump()))
    self.client.upsert(collection_name=self.collection_name, points=points)


def _filter(self, library):
    if library is None:
        return None
    return Filter(must=[FieldCondition(key="library", match=MatchValue(value=library))])


def search(self, query_vector, top_k, library=None):
    resp = self.client.query_points(
        collection_name=self.collection_name, query=query_vector, using=DENSE,
        limit=top_k, query_filter=self._filter(library))
    return [RetrievedChunk(chunk=Chunk(**h.payload), score=h.score) for h in resp.points]


def hybrid_search(self, query_vector, sparse_query, top_k, library=None):
    idx, val = sparse_query
    qf = self._filter(library)
    resp = self.client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            Prefetch(query=query_vector, using=DENSE, limit=top_k * 5, filter=qf),
            Prefetch(query=SparseVector(indices=idx, values=val), using=SPARSE,
                     limit=top_k * 5, filter=qf),
        ],
        query=FusionQuery(fusion=Fusion.RRF), limit=top_k, query_filter=qf)
    return [RetrievedChunk(chunk=Chunk(**h.payload), score=h.score) for h in resp.points]
```

> 注:`_point_id` / `count` / `list_libraries` 保持不变。`Prefetch` 的过滤参数名以安装版 qdrant-client 为准(1.18 为 `filter=`);若报参数名错,改成该版本接受的名字(实现时以 `:memory:` 测试跑通为准)。

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_qdrant_store.py -v` → PASS;再跑全量 `.venv/bin/python -m pytest -q`(可能有依赖旧 upsert 的测试需顺带适配)。
- [ ] **Step 5: 提交** `git add src/rag/interfaces.py src/rag/providers/qdrant_store.py tests/test_qdrant_store.py && git commit -m "feat: QdrantStore 命名向量 dense+sparse + hybrid_search(RRF)"`

---

### Task 3: retrieve.py 混合路径

**Files:** Modify `src/rag/retrieve.py`; Test `tests/test_retrieve.py`

**Interfaces:** `retrieve(..., query_prefix="", hybrid=False)`。hybrid 时用 `encode_sparse(改写后 query)` 调 `store.hybrid_search`。

- [ ] **Step 1: 写失败测试** 在 `tests/test_retrieve.py` 追加(捕获 fake store 记录调用):

```python
def test_retrieve_hybrid_uses_hybrid_search_with_sparse_from_query():
    from rag.retrieve import retrieve

    class _Emb:
        def embed_one(self, t): return [0.0]
    class _Store:
        def __init__(self): self.called = None
        def search(self, *a, **k): self.called = ("dense", a, k); return []
        def hybrid_search(self, qv, sparse, top_k, library=None):
            self.called = ("hybrid", sparse); return []

    st = _Store()
    retrieve("LEFT JOIN 慢", _Emb(), st, top_k=4, hybrid=True)
    assert st.called[0] == "hybrid"
    idx, val = st.called[1]
    assert len(idx) == len(val) > 0              # 稀疏向量来自 query 文本
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `retrieve.py`:

```python
from rag.sparse import encode_sparse

def retrieve(question, embedder, store, top_k, library=None, rewriter=None,
             reranker=None, rerank_factor=5, query_prefix="", hybrid=False):
    query = rewriter.rewrite(question) if rewriter is not None else question
    query_vector = embedder.embed_one(query_prefix + query)
    recall_k = top_k * rerank_factor if reranker is not None else top_k
    if hybrid:
        hits = store.hybrid_search(query_vector, encode_sparse(query),
                                   top_k=recall_k, library=library)
    else:
        hits = store.search(query_vector, top_k=recall_k, library=library)
    if reranker is not None:
        return reranker.rerank(question, hits, top_k=top_k)
    return hits
```

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_retrieve.py -v` → PASS
- [ ] **Step 5: 提交** `git add src/rag/retrieve.py tests/test_retrieve.py && git commit -m "feat: retrieve 混合检索路径(hybrid 开关)"`

---

### Task 4: ingest.py 计算并存稀疏向量

**Files:** Modify `src/rag/ingest.py`; Test `tests/test_ingest.py`

**Interfaces:** `ingest_directory` 内对每个 chunk 算 `encode_sparse(c.text)`(原文,不加 doc_prefix),传给 `store.upsert(..., sparse_vectors=...)`。

- [ ] **Step 1: 写失败测试** 在 `tests/test_ingest.py` 追加(捕获 upsert 收到的 sparse_vectors):

```python
def test_ingest_passes_sparse_vectors_per_chunk():
    from rag.ingest import ingest_directory
    from pathlib import Path
    import tempfile
    # 用记录 upsert 入参的假 store + 假 embedder(复用文件里已有风格)
    captured = {}
    class _Emb:
        def embed(self, texts): return [[0.0] for _ in texts]
        def embed_one(self, t): return [0.0]
    class _Store:
        def ensure_collection(self, vector_size): pass
        def upsert(self, chunks, vectors, sparse_vectors=None):
            captured["sparse"] = sparse_vectors; captured["n"] = len(chunks)
    with tempfile.TemporaryDirectory() as d:
        Path(d, "a.md").write_text("# T\nLEFT JOIN index\n", encoding="utf-8")
        ingest_directory(Path(d), _Emb(), _Store(), chunk_size=800, overlap=0,
                         vector_size=1)
    assert captured["sparse"] is not None
    assert len(captured["sparse"]) == captured["n"]     # 每 chunk 一个稀疏向量
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `ingest.py`:

```python
from rag.sparse import encode_sparse

def ingest_directory(docs_dir, embedder, store, chunk_size, overlap, vector_size,
                     strategy="char", doc_prefix="", noise_markers=_DEFAULT_NOISE_MARKERS):
    chunks = load_chunks_from_dir(docs_dir, chunk_size=chunk_size, overlap=overlap,
                                  strategy=strategy, noise_markers=noise_markers)
    if not chunks:
        return 0
    store.ensure_collection(vector_size=vector_size)
    vectors = embedder.embed([doc_prefix + c.text for c in chunks])
    sparse_vectors = [encode_sparse(c.text) for c in chunks]
    store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)
    return len(chunks)
```

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_ingest.py -v` → PASS
- [ ] **Step 5: 提交** `git add src/rag/ingest.py tests/test_ingest.py && git commit -m "feat: ingest 计算并存稀疏向量"`

---

### Task 5: config + pipeline + 装配层接线 hybrid

**Files:** Modify `src/rag/config.py`, `src/rag/pipeline.py`, `scripts/{serve,eval,ingest,ask,ablate,gen_eval}.py`; Test `tests/test_pipeline.py`

**Interfaces:** `Settings.hybrid: bool=False`、`hybrid_prefetch_factor: int=5`;`RagPipeline(..., hybrid=False)` 透传给 retrieve;各 script 从 settings 注入 hybrid;retrieve 调用点传 hybrid;ingest 装配无需变(sparse 自动)。

- [ ] **Step 1: 写失败测试** `tests/test_pipeline.py` 追加:

```python
def test_pipeline_hybrid_flag_routes_to_hybrid_search():
    from rag.pipeline import RagPipeline
    class _Emb:
        def embed_one(self, t): return [0.0]
    class _Store:
        def __init__(self): self.mode = None
        def search(self, *a, **k): self.mode = "dense"; return []
        def hybrid_search(self, *a, **k): self.mode = "hybrid"; return []
    class _LLM:
        def generate(self, p): return "答"
    st = _Store()
    RagPipeline(_Emb(), st, _LLM(), top_k=4, hybrid=True).ask("q")
    assert st.mode == "hybrid"
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现**
  - `config.py`:加 `hybrid: bool = False` 和 `hybrid_prefetch_factor: int = 5`。
  - `pipeline.py`:`__init__(..., hybrid: bool = False)` 存 `self.hybrid`;`ask` 里 `retrieve(..., hybrid=self.hybrid)`。
  - 各 script:`retrieve(...)` 调用点传 `hybrid=settings.hybrid`;`RagPipeline(...)` 传 `hybrid=settings.hybrid`;`RagAgent` 的 retriever 闭包传 `hybrid=settings.hybrid`。(ingest.py 装配不变——稀疏在 ingest_directory 内部自动算。)
- [ ] **Step 4: 确认通过 + 全量 + 脚本语法**
```bash
.venv/bin/python -m pytest -q
for f in serve eval ingest ask ablate gen_eval; do .venv/bin/python -c "import ast; ast.parse(open('scripts/$f.py').read())" && echo "$f ok"; done
```
- [ ] **Step 5: 提交** `git add -A && git commit -m "feat: hybrid 开关接线(config/pipeline/scripts)"`

---

### Task 6: 重灌 + hybrid on/off 度量(实机,控制器执行)

> 需 Ollama + Qdrant 在线;非 TDD 子代理任务。

- [ ] **Step 1: 删旧库 + 重灌**(现在会存 dense+sparse):
```bash
curl -s -X DELETE http://localhost:6333/collections/rag_docs >/dev/null
.venv/bin/python scripts/ingest.py
```
- [ ] **Step 2: eval hybrid 关**(基线复现)`HYBRID=false`:`.venv/bin/python scripts/eval.py`
- [ ] **Step 3: eval hybrid 开**:`HYBRID=true .venv/bin/python scripts/eval.py`
- [ ] **Step 4: 记录判定** 在 `notes/07` 追加 hybrid off/on 对比表 + 结论。recall@4 升则 `HYBRID=true` 设推荐(config 默认或 .env),否则默认关(机制保留)。提交 `notes/07`(+ 若改默认则 config)。

---

## Self-Review
- Spec §1 稀疏编码(crc32)→T1 ✅;§2 命名向量+hybrid_search→T2 ✅;§3 retrieve→T3 ✅;§4 config/ingest/装配→T4+T5 ✅;§5 度量→T6 ✅;§6 文件全覆盖;§7 测试分散各任务;§8 非目标未越界;§9 诚实提醒(crc32/重灌/eval 为准)贯穿。
- Placeholder:唯一软处是 Prefetch 过滤参数名(T2 注明"以 :memory: 测试跑通为准"),给了明确落地判据,非占位符。
- Type 一致:`sparse` 统一为 `tuple[list[int],list[float]]`(sparse.py 产出 → ingest 传 → store.upsert/hybrid_search 收 → retrieve 传);`hybrid: bool` 默认 False 贯穿 retrieve/pipeline/scripts。
