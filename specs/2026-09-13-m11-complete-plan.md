# M11 补齐 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** 补齐 M11:多查询 + 上下文排序 + 行内引用。三开关默认关,零依赖,eval 可量。

**Spec:** `specs/2026-09-13-m11-complete-design.md`

## Global Constraints
- `.venv/bin/python -m pytest`(venv=3.12)。零新依赖。三开关默认关=逐字节现状(回归安全)。检索/重排/embedding 算法不动。conventional commits。

---

### Task 1: multi_query.py(expand_queries + rrf_fuse)

**Files:** Create `src/rag/multi_query.py`, `tests/test_multi_query.py`

**Interfaces:** `expand_queries(llm, question, n=3)->list[str]`(含原问题、去重、失败兜底 [question]);`rrf_fuse(result_lists, top_k, k_rrf=60)->list[RetrievedChunk]`。

- [ ] **Step 1: 写失败测试** `tests/test_multi_query.py`:
```python
from rag.multi_query import expand_queries, rrf_fuse
from rag.models import Chunk, RetrievedChunk


def _rc(cid, score=1.0):
    return RetrievedChunk(chunk=Chunk(id=cid, text="t", source=cid, library="l", chunk_index=0), score=score)


class _FakeLLM:
    def __init__(self, reply): self.reply = reply
    def generate(self, prompt): return self.reply


def test_expand_includes_original_and_dedupes():
    out = expand_queries(_FakeLLM("变体一\n变体二\n变体一"), "原问题", n=3)
    assert out[0] == "原问题"
    assert "变体一" in out and "变体二" in out
    assert len(out) == len(set(out)) <= 3


def test_expand_n1_returns_only_original():
    assert expand_queries(_FakeLLM("x"), "q", n=1) == ["q"]


def test_expand_empty_reply_falls_back():
    assert expand_queries(_FakeLLM("   "), "q", n=3) == ["q"]


def test_rrf_fuse_ranks_consensus_first():
    # a 在两个列表都靠前 → 融合应排第一
    l1 = [_rc("a"), _rc("b"), _rc("c")]
    l2 = [_rc("a"), _rc("d")]
    out = rrf_fuse([l1, l2], top_k=3)
    assert out[0].chunk.id == "a"
    assert out[0].score >= out[1].score            # 分数降序


def test_rrf_fuse_dedupes_by_id():
    l1 = [_rc("a")]; l2 = [_rc("a")]
    out = rrf_fuse([l1, l2], top_k=5)
    assert len(out) == 1
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `src/rag/multi_query.py`:
```python
"""多查询检索:生成查询变体 + 客户端 RRF 融合(M11)。零依赖。"""
from rag.interfaces import LLM
from rag.models import RetrievedChunk

_EXPAND_PROMPT = """针对下面的问题,生成 {k} 个措辞或角度不同、但意图相同的检索查询,便于多路检索。\
每行一个,不要编号、不要解释。

问题:{question}
查询:"""


def expand_queries(llm: LLM, question: str, n: int = 3) -> list[str]:
    if n <= 1:
        return [question]
    try:
        reply = llm.generate(_EXPAND_PROMPT.format(k=n - 1, question=question))
    except Exception:
        return [question]
    variants = [ln.strip(" -*\t") for ln in reply.splitlines() if ln.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for q in [question, *variants]:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:n]


def rrf_fuse(result_lists: list[list[RetrievedChunk]], top_k: int,
             k_rrf: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, rc in enumerate(results):
            cid = rc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
            best.setdefault(cid, rc)
    ranked = sorted(best.values(), key=lambda rc: scores[rc.chunk.id], reverse=True)
    return [RetrievedChunk(chunk=rc.chunk, score=scores[rc.chunk.id])
            for rc in ranked[:top_k]]
```

- [ ] **Step 4: 确认通过** → PASS;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/multi_query.py tests/test_multi_query.py && git commit -m "feat(M11): 多查询扩展 + 客户端 RRF 融合"`

---

### Task 2: retrieve 集成 query_expander

**Files:** Modify `src/rag/retrieve.py`; Test `tests/test_retrieve.py`

**Interfaces:** `retrieve(..., query_expander: Callable[[str], list[str]] | None = None)`;None=单查询(现状);非 None→多查询各自检索+rrf_fuse,再(可选)重排。

- [ ] **Step 1: 写失败测试** 在 `tests/test_retrieve.py` 追加:
```python
def test_retrieve_multi_query_calls_underlying_per_variant_and_fuses():
    from rag.retrieve import retrieve
    calls = []
    class _Emb:
        def embed_one(self, t): calls.append(t); return [0.0]
    class _Store:
        def search(self, qv, top_k, library=None):
            return []
    # expander 返回 2 个变体
    retrieve("原问题", _Emb(), _Store(), top_k=4,
             query_expander=lambda q: ["查询A", "查询B"])
    assert len(calls) == 2                      # 每个变体各 embed 一次(各检索一次)

def test_retrieve_no_expander_single_query_regression():
    from rag.retrieve import retrieve
    calls = []
    class _Emb:
        def embed_one(self, t): calls.append(t); return [0.0]
    class _Store:
        def search(self, qv, top_k, library=None): return []
    retrieve("q", _Emb(), _Store(), top_k=4)     # 默认无 expander
    assert len(calls) == 1                       # 单查询,与现状一致
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** 重构 `src/rag/retrieve.py`:把单查询检索抽成内部 `_retrieve_one(q)`,retrieve 按 expander 决定单/多:
```python
from collections.abc import Callable
from rag.multi_query import rrf_fuse
from rag.sparse import encode_sparse

def retrieve(question, embedder, store, top_k, library=None, rewriter=None,
             reranker=None, rerank_factor=5, query_prefix="", hybrid=False,
             hybrid_prefetch_factor=5,
             query_expander: Callable[[str], list[str]] | None = None):
    recall_k = top_k * rerank_factor if reranker is not None else top_k

    def _retrieve_one(q):
        query = rewriter.rewrite(q) if rewriter is not None else q
        qv = embedder.embed_one(query_prefix + query)
        if hybrid:
            return store.hybrid_search(qv, encode_sparse(query), top_k=recall_k, library=library)
        return store.search(qv, top_k=recall_k, library=library)

    if query_expander is None:
        hits = _retrieve_one(question)
    else:
        lists = [_retrieve_one(q) for q in query_expander(question)]
        hits = rrf_fuse(lists, recall_k) if len(lists) > 1 else (lists[0] if lists else [])

    if reranker is not None:
        return reranker.rerank(question, hits, top_k=top_k)
    return hits
```
(注:保持 hybrid 分支的 hybrid_prefetch_factor 行为——若现有 hybrid_search 调用带该参数,内部一并传;以现有实现为准,别改 hybrid 语义。)

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_retrieve.py -v` → PASS(含现有回归);全量绿。
- [ ] **Step 5: 提交** `git add src/rag/retrieve.py tests/test_retrieve.py && git commit -m "feat(M11): retrieve 支持 query_expander 多查询融合(默认单查询)"`

---

### Task 3: ordering.py 上下文排序

**Files:** Create `src/rag/ordering.py`; Test `tests/test_ordering.py`

**Interfaces:** `reorder_for_long_context(chunks)->list[RetrievedChunk]`:输入按相关性降序,输出把最相关放两端。

- [ ] **Step 1: 写失败测试** `tests/test_ordering.py`:
```python
from rag.ordering import reorder_for_long_context
from rag.models import Chunk, RetrievedChunk


def _rc(cid):
    return RetrievedChunk(chunk=Chunk(id=cid, text="t", source=cid, library="l", chunk_index=0), score=1.0)


def test_most_relevant_at_ends():
    out = reorder_for_long_context([_rc("r0"), _rc("r1"), _rc("r2"), _rc("r3")])
    ids = [c.chunk.id for c in out]
    assert ids[0] == "r0"        # 最相关在首
    assert ids[-1] == "r1"       # 次相关在尾
    assert set(ids) == {"r0", "r1", "r2", "r3"}   # 不丢不重


def test_short_lists_unchanged():
    assert [c.chunk.id for c in reorder_for_long_context([_rc("a")])] == ["a"]
    assert reorder_for_long_context([]) == []
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `src/rag/ordering.py`:
```python
"""缓解 lost-in-the-middle:把最相关的块放到上下文两端(M11)。"""
from rag.models import RetrievedChunk


def reorder_for_long_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """输入按相关性降序。最相关放首、次相关放尾,由外向内交替填充。"""
    n = len(chunks)
    if n <= 2:
        return list(chunks)
    result: list[RetrievedChunk | None] = [None] * n
    left, right = 0, n - 1
    for i, rc in enumerate(chunks):        # i = 相关性排名(0 最高)
        if i % 2 == 0:
            result[left] = rc; left += 1
        else:
            result[right] = rc; right -= 1
    return [rc for rc in result if rc is not None]
```

- [ ] **Step 4: 确认通过** → PASS;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/ordering.py tests/test_ordering.py && git commit -m "feat(M11): 上下文排序(lost-in-the-middle)"`

---

### Task 4: generate.py 行内引用

**Files:** Modify `src/rag/generate.py`; Test `tests/test_generate.py`

**Interfaces:** `build_prompt(..., inline_citations=False)`、`answer(..., inline_citations=False)`;开时加引用指令;默认关=现状。

- [ ] **Step 1: 写失败测试** 在 `tests/test_generate.py` 追加:
```python
def test_inline_citations_adds_instruction():
    from rag.generate import build_prompt
    p = build_prompt("q", [_rc_simple()], inline_citations=True)
    assert "[1]" in p or "编号" in p           # 含引用标注指令

def test_inline_citations_off_by_default():
    from rag.generate import build_prompt
    p = build_prompt("q", [_rc_simple()])
    assert "标注" not in p                     # 默认不加引用指令
```
(`_rc_simple` 用文件里现有 helper。)

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `build_prompt`/`answer` 加 `inline_citations: bool = False`;开时在 system 规则里追加一条:
```python
    cite_rule = ("\n4. 引用了哪段【资料】,就在该句末尾用其编号标注,如 [1]、[2]。"
                 if inline_citations else "")
    system = ( ...现有三条规则... + cite_rule )
```
(把 cite_rule 拼进 system 字符串;history 参数若已存在则共存。`answer` 透传 inline_citations。)

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_generate.py -v` → PASS(含回归);全量绿。
- [ ] **Step 5: 提交** `git add src/rag/generate.py tests/test_generate.py && git commit -m "feat(M11): 生成支持行内引用(默认关)"`

---

### Task 5: config + pipeline + scripts 装配

**Files:** Modify `src/rag/config.py`, `src/rag/pipeline.py`, `scripts/{eval,serve,ablate}.py`; Test `tests/test_pipeline.py`

**Interfaces:** `Settings`: `multi_query=False`/`multi_query_n=3`/`reorder_context=False`/`inline_citations=False`;`RagPipeline` 透传三项;scripts 装配。

- [ ] **Step 1: 写失败测试** 在 `tests/test_pipeline.py` 追加一个:pipeline 开 inline_citations 时,喂给 fake LLM 的 prompt 含引用指令(捕获 prompt 的 fake LLM)。（multi_query/reorder 的端到端在各自单元已覆盖,pipeline 层验 inline 透传即可。）
- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现**
  - `config.py`:加 4 个字段。
  - `pipeline.py`:`RagPipeline.__init__` 加 `multi_query=False, multi_query_n=3, reorder_context=False, inline_citations=False` + llm 已有;`ask` 里:构造 `query_expander`(multi_query 开时 `lambda q: expand_queries(self.llm, q, self.multi_query_n)`)传 retrieve;retrieve 后若 reorder_context 则 `reorder_for_long_context`;`generate_answer(..., inline_citations=self.inline_citations)`。
  - `scripts/eval.py`/`serve.py`/`ablate.py`:同样从 settings 构造 query_expander、按开关 reorder、传 inline_citations 给 generate。(serve 的 pipeline 构造传四参;eval 单轮路径直接在 retrieve/generate 调用点接。)
- [ ] **Step 4: 确认通过 + 脚本语法** 全量 `.venv/bin/python -m pytest -q` 绿;`for f in eval serve ablate; do .venv/bin/python -c "import ast; ast.parse(open('scripts/$f.py').read())" && echo "$f ok"; done`。
- [ ] **Step 5: 提交** `git add -A && git commit -m "feat(M11): 装配多查询/上下文排序/行内引用开关"`

---

### Task 6: 度量(实机,控制器执行)
- [ ] eval:baseline / +multi_query / +reorder / +inline(逐个 `MULTI_QUERY=true` 等)。对照当前基线。
- [ ] 记 notes/07;有效项设推荐,无效默认关。重点看 multi_query 是否救回"图数据库插入点"漏召回题。

---

## Self-Review
- Spec §1 multi_query→T1+T2;§2 ordering→T3;§3 inline→T4;§4 装配→T5;§6 度量→T6;文件全覆盖;非目标(不做 HyDE/bge/多轮/新依赖)守住。
- 回归:query_expander 默认 None=单查询逐字节现状(T2 回归测试);inline_citations 默认 False=现状(T4);reorder 仅在开关开时应用。
- 类型:query_expander `Callable[[str],list[str]]`;rrf_fuse 输入 list[list[RetrievedChunk]];reorder 输入=输出类型。
