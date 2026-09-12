# M8 通用化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把技术文档专属的耦合点(人设/拒答话术/改写 prompt/噪声词/embedding 前缀)抽成命名 profile(`profiles/*.toml`),让换 profile + 换库即可换领域;默认 tech profile 复刻现状。

**Architecture:** 新增 `DomainProfile` + `load_profile`(stdlib tomllib)。各组件不认识 profile,只接收朴素参数(字符串/列表),且每个新参数都有**复刻现状的默认值**——所以逐个模块改造时全量测试始终绿。装配层(scripts)加载 profile 并注入。embedding 前缀在真实调用点(ingest/retrieve)拼,不进 embedder。

**Tech Stack:** Python 3.11+(tomllib 内置)、pydantic v2、pytest。**零新第三方依赖。**

**Spec:** `specs/2026-09-13-m8-generalization-design.md`

## Global Constraints

- Python `>=3.11`;测试用 `.venv/bin/python -m pytest`(系统 python3 是 3.9,会因 `str | None` 失败)。
- **零新依赖**:profile 用 stdlib `tomllib` 读。
- **默认值复刻现状**:每个新增参数的默认值 = 当前写死的值,保证每次提交全量测试绿、现有 Qdrant 库(无前缀向量)保持有效。
- **embedding 前缀在 `profiles/tech.toml` 里先留空**(Task 1–8 期间),Task 9 才设成 nomic 值并重灌——避免中间出现 query 带前缀/doc 无前缀的失配。
- 不改检索算法本身(多路召回是 M9)。
- conventional commits;可测逻辑在 `src/rag`,`scripts/` 仅装配。
- 测试输出里有 1 条预存 StarletteDeprecationWarning(fastapi 依赖),非本分支引入,忽略。

---

### Task 1: DomainProfile + load_profile + profiles

**Files:**
- Create: `src/rag/profile.py`, `profiles/tech.toml`, `profiles/generic.toml`
- Modify: `src/rag/config.py`(加 `profile` 字段)
- Test: `tests/test_profile.py`

**Interfaces:**
- Produces: `DomainProfile`(pydantic,字段 persona/refusal_text/refusal_marker/rewrite_prompt/noise_markers/embed_query_prefix/embed_doc_prefix,均有默认);`load_profile(name: str, profiles_dir=Path("profiles")) -> DomainProfile`;`Settings.profile: str = "tech"`。

- [ ] **Step 1: 写失败测试** — 创建 `tests/test_profile.py`:

```python
from pathlib import Path

import pytest

from rag.profile import DomainProfile, load_profile


def test_load_tech_profile():
    p = load_profile("tech")
    assert "技术文档" in p.persona
    assert p.refusal_text == "根据现有资料无法回答"
    assert p.refusal_marker == "无法回答"
    assert "{question}" in p.rewrite_prompt          # 改写 prompt 保留占位符
    assert "sponsor" in p.noise_markers
    # Task 1–8 期间 tech 前缀留空(Task 9 才启用)
    assert p.embed_query_prefix == ""
    assert p.embed_doc_prefix == ""


def test_load_generic_profile():
    p = load_profile("generic")
    assert p.rewrite_prompt == ""                    # generic 不改写
    assert p.noise_markers == []


def test_missing_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("no_such_profile")


def test_defaults_fill_missing_fields(tmp_path):
    (tmp_path / "min.toml").write_text('persona = "只有人设"\n', encoding="utf-8")
    p = load_profile("min", profiles_dir=tmp_path)
    assert p.persona == "只有人设"
    assert p.refusal_marker == "无法回答"             # 缺字段回落默认
```

- [ ] **Step 2: 运行确认失败** — `cd /Users/mi/Documents/test/rag-docs && .venv/bin/python -m pytest tests/test_profile.py -v` → FAIL(`rag.profile` 不存在)

- [ ] **Step 3: 实现** — 创建 `src/rag/profile.py`:

```python
"""领域 profile:把技术文档专属的耦合点(人设/改写/噪声/前缀)外置成可切换配置。

用 stdlib tomllib 读 profiles/<name>.toml(零依赖)。字段均有默认,缺字段回落。
"""
import tomllib
from pathlib import Path

from pydantic import BaseModel


class DomainProfile(BaseModel):
    persona: str = "你是一个严谨的问答助手"
    refusal_text: str = "根据现有资料无法回答"
    refusal_marker: str = "无法回答"
    rewrite_prompt: str = ""                 # 空 = 不改写
    noise_markers: list[str] = []
    embed_query_prefix: str = ""
    embed_doc_prefix: str = ""


def load_profile(name: str, profiles_dir: Path = Path("profiles")) -> DomainProfile:
    path = Path(profiles_dir) / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"profile 不存在: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DomainProfile(**data)
```

创建 `profiles/tech.toml`(**前缀留空**,rewrite_prompt 原样搬自现 query_rewrite._REWRITE_PROMPT):

```toml
persona = "你是一个严谨的技术文档问答助手"
refusal_text = "根据现有资料无法回答"
refusal_marker = "无法回答"
rewrite_prompt = """你在为技术文档检索改写查询。文档是英文的。
请针对下面的问题,只输出最相关的英文关键词/术语(空格分隔,不要解释、不要标点):

问题:{question}
英文关键词:"""
noise_markers = ["sponsor", "fastapi cloud", "conf", "deploying to", "deployment successful"]
embed_query_prefix = ""
embed_doc_prefix = ""
```

创建 `profiles/generic.toml`:

```toml
persona = "你是一个严谨的问答助手"
refusal_text = "根据现有资料无法回答"
refusal_marker = "无法回答"
rewrite_prompt = ""
noise_markers = []
embed_query_prefix = ""
embed_doc_prefix = ""
```

在 `src/rag/config.py` 的 `Settings` 里(切分策略附近)加:

```python
    # M8:领域 profile(profiles/<name>.toml),换领域=换名字
    profile: str = "tech"
```

- [ ] **Step 4: 运行确认通过** — `.venv/bin/python -m pytest tests/test_profile.py -v` → PASS(4 条)

- [ ] **Step 5: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/profile.py profiles/tech.toml profiles/generic.toml src/rag/config.py tests/test_profile.py
git commit -m "feat: DomainProfile + load_profile + tech/generic profiles"
```

---

### Task 2: generate.py 注入 persona/refusal_text

**Files:** Modify `src/rag/generate.py`; Test `tests/test_generate.py`

**Interfaces:**
- Consumes: 无(用默认常量)
- Produces: `build_prompt(question, chunks, persona=_DEFAULT_PERSONA, refusal_text=_DEFAULT_REFUSAL) -> str`;`answer(question, chunks, llm, persona=_DEFAULT_PERSONA, refusal_text=_DEFAULT_REFUSAL) -> Answer`。默认值复刻现状。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_generate.py` 追加:

```python
from rag.generate import build_prompt
from rag.models import Chunk, RetrievedChunk


def _rc(text="正文"):
    return RetrievedChunk(chunk=Chunk(id="s::0", text=text, source="s.md",
                                      library="l", chunk_index=0), score=1.0)


def test_build_prompt_injects_persona_and_refusal():
    prompt = build_prompt("问题", [_rc()], persona="你是保险顾问",
                          refusal_text="资料里没有这条")
    assert "你是保险顾问" in prompt
    assert "资料里没有这条" in prompt


def test_build_prompt_defaults_preserve_current_behavior():
    prompt = build_prompt("问题", [_rc()])
    assert "技术文档问答助手" in prompt
    assert "根据现有资料无法回答" in prompt
```

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_generate.py -k "persona or defaults_preserve" -v` → FAIL(build_prompt 不接受 persona)

- [ ] **Step 3: 实现** — 在 `src/rag/generate.py`,把写死的 `_SYSTEM` 换成默认常量 + 参数化:

```python
_DEFAULT_PERSONA = "你是一个严谨的技术文档问答助手"
_DEFAULT_REFUSAL = "根据现有资料无法回答"


def build_prompt(question: str, chunks: list[RetrievedChunk],
                 persona: str = _DEFAULT_PERSONA,
                 refusal_text: str = _DEFAULT_REFUSAL) -> str:
    system = (
        f"{persona}。请严格遵守以下规则:\n"
        f"1. 只能依据下面【资料】中的内容回答问题。\n"
        f"2. 如果【资料】中没有足够信息回答,必须回答\"{refusal_text}\","
        f"不要编造、不要凭常识补充。\n"
        f"3. 回答尽量简洁、准确,可引用资料中的术语。"
    )
    if chunks:
        blocks = [f"[资料{i} | 来源:{rc.chunk.source}]\n{rc.chunk.text}"
                  for i, rc in enumerate(chunks, start=1)]
        context = "\n\n".join(blocks)
    else:
        context = "(无相关资料)"
    return (f"{system}\n\n【资料】\n{context}\n\n【问题】\n{question}\n\n【回答】")


def answer(question: str, chunks: list[RetrievedChunk], llm: LLM,
           persona: str = _DEFAULT_PERSONA,
           refusal_text: str = _DEFAULT_REFUSAL) -> Answer:
    text = llm.generate(build_prompt(question, chunks, persona, refusal_text))
    sources: list[str] = []
    for rc in chunks:
        if rc.chunk.source not in sources:
            sources.append(rc.chunk.source)
    return Answer(text=text, sources=sources)
```

删除旧的模块级 `_SYSTEM` 常量。

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_generate.py -v` → PASS（含既有测试）
- [ ] **Step 5: 提交** — `git add src/rag/generate.py tests/test_generate.py && git commit -m "feat: generate 注入 persona/refusal_text(默认复刻现状)"`

---

### Task 3: agent.py 注入 persona/refusal_text

**Files:** Modify `src/rag/agent.py`; Test `tests/test_agent.py`

**Interfaces:**
- Produces: `RagAgent(llm, retriever, top_k=4, max_steps=5, persona=_DEFAULT_PERSONA, refusal_text=_DEFAULT_REFUSAL)`;system 段由 persona/refusal_text 构造。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_agent.py` 追加(沿用文件里已有的 FakeChatLLM 风格;若无,构造一个返回最终文本的假 ChatLLM):

```python
def test_agent_system_prompt_uses_injected_persona():
    from rag.agent import RagAgent
    agent = RagAgent(llm=_make_fake_chat_llm_returning_text("答"),
                     retriever=lambda q, library=None, top_k=4: [],
                     persona="你是保险顾问", refusal_text="资料没有")
    # 取 agent 构造的 system 段(实现里存为 self._system 或首条 message)
    assert "你是保险顾问" in agent._system
    assert "资料没有" in agent._system
```

> 说明:实现需把构造好的 system 文本存成 `self._system`(供测试读取,也是 ask() 里用的那条 system message)。`_make_fake_chat_llm_returning_text` 用文件里既有的假 ChatLLM 或按 test_agent 现有模式写一个最小版。

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_agent.py -k persona -v` → FAIL

- [ ] **Step 3: 实现** — 在 `src/rag/agent.py`:把模块级写死的 `_SYSTEM` 改为默认常量,`RagAgent.__init__` 接收 persona/refusal_text 并构造 `self._system`:

```python
_DEFAULT_PERSONA = "你是一个严谨的技术文档问答助手"
_DEFAULT_REFUSAL = "根据现有资料无法回答"


def _build_system(persona: str, refusal_text: str) -> str:
    return (
        f"{persona},可以使用 search_docs 工具检索文档。\n\n"
        f"工作方式:\n"
        f"- 需要资料时,**用结构化的工具调用**触发 search_docs,不要把工具调用当文本写进正文。\n"
        f"- 若结果不够好,可以换个查询词再检索一轮。\n"
        f"- 收集到足够资料后,只依据检索到的资料回答。\n"
        f"- 如果检索不到相关资料,必须回答\"{refusal_text}\",不要编造。\n"
        f"- 回答尽量简洁、准确。"
    )


class RagAgent:
    def __init__(self, llm, retriever, top_k: int = 4, max_steps: int = 5,
                 persona: str = _DEFAULT_PERSONA,
                 refusal_text: str = _DEFAULT_REFUSAL):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_steps = max_steps
        self._system = _build_system(persona, refusal_text)
```

在 `ask()` 里把 `Message(role="system", content=_SYSTEM)` 改为 `content=self._system`。删除旧 `_SYSTEM` 模块常量。

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_agent.py -v` → PASS
- [ ] **Step 5: 提交** — `git add src/rag/agent.py tests/test_agent.py && git commit -m "feat: agent 注入 persona/refusal_text"`

---

### Task 4: query_rewrite.py 注入 prompt

**Files:** Modify `src/rag/query_rewrite.py`; Test `tests/test_query_rewriter.py`

**Interfaces:**
- Produces: `LlmQueryRewriter(llm, prompt=_DEFAULT_REWRITE_PROMPT)`;`rewrite` 用 `self.prompt.format(question=...)`。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_query_rewriter.py` 追加:

```python
def test_rewriter_uses_injected_prompt():
    from rag.query_rewrite import LlmQueryRewriter

    class _FakeLLM:
        def __init__(self): self.seen = None
        def generate(self, prompt): self.seen = prompt; return "扩展词"

    llm = _FakeLLM()
    r = LlmQueryRewriter(llm, prompt="自定义改写:{question}")
    out = r.rewrite("原问题")
    assert llm.seen == "自定义改写:原问题"      # 用了注入 prompt
    assert "原问题" in out                       # 仍拼回原问题兜底
```

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_query_rewriter.py -k injected -v` → FAIL

- [ ] **Step 3: 实现** — 在 `src/rag/query_rewrite.py`:保留现 prompt 文本为 `_DEFAULT_REWRITE_PROMPT`,`__init__` 接收 prompt:

```python
class LlmQueryRewriter(QueryRewriter):
    def __init__(self, llm: LLM, prompt: str = _DEFAULT_REWRITE_PROMPT):
        self.llm = llm
        self.prompt = prompt

    def rewrite(self, question: str) -> str:
        expansion = self.llm.generate(self.prompt.format(question=question)).strip()
        if not expansion:
            return question.strip()
        return f"{expansion} {question}".strip()
```

(把原模块级 `_REWRITE_PROMPT` 改名为 `_DEFAULT_REWRITE_PROMPT`,内容不变。)

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_query_rewriter.py -v` → PASS
- [ ] **Step 5: 提交** — `git add src/rag/query_rewrite.py tests/test_query_rewriter.py && git commit -m "feat: query_rewrite 注入 prompt"`

---

### Task 5: chunking.py + loader.py 注入 noise_markers

**Files:** Modify `src/rag/chunking.py`, `src/rag/loader.py`; Test `tests/test_markdown_chunking.py`

**Interfaces:**
- Produces: `is_noise(heading, body, noise_markers=_DEFAULT_NOISE_MARKERS) -> bool`;`chunk_markdown(text, chunk_size, overlap, noise_markers=_DEFAULT_NOISE_MARKERS)`;`load_chunks_from_dir(docs_dir, chunk_size, overlap, strategy="char", noise_markers=_DEFAULT_NOISE_MARKERS)`。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_markdown_chunking.py` 追加:

```python
def test_chunk_markdown_uses_injected_noise_markers():
    from rag.chunking import chunk_markdown
    text = "# 广告位\n买保险找我们\n\n# 正文\n这是有用的内容。"
    # 用自定义噪声词过滤"广告位"段;默认的 FastAPI 词此处不该生效
    chunks = chunk_markdown(text, chunk_size=800, overlap=0,
                            noise_markers=["广告位", "买保险"])
    joined = "\n".join(chunks)
    assert "有用的内容" in joined
    assert "买保险找我们" not in joined
```

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_markdown_chunking.py -k injected -v` → FAIL

- [ ] **Step 3: 实现**
  - `chunking.py`:把 `_NOISE_MARKERS` 改名 `_DEFAULT_NOISE_MARKERS`(内容不变);`is_noise(heading, body, noise_markers=_DEFAULT_NOISE_MARKERS)`;`chunk_markdown(..., noise_markers=_DEFAULT_NOISE_MARKERS)`,内部 `flush()` 调 `is_noise(cur_heading, body, noise_markers)`。
  - `loader.py`:`load_chunks_from_dir(..., noise_markers=_DEFAULT_NOISE_MARKERS)`(从 chunking import 默认),strategy=="markdown" 时 `chunk_markdown(text, chunk_size, overlap, noise_markers)`。

```python
# chunking.py
def is_noise(heading: str, body: str,
             noise_markers: tuple[str, ...] | list[str] = _DEFAULT_NOISE_MARKERS) -> bool:
    blob = f"{heading}\n{body}".lower()
    return any(marker in blob for marker in noise_markers)


def chunk_markdown(text: str, chunk_size: int, overlap: int,
                   noise_markers=_DEFAULT_NOISE_MARKERS) -> list[str]:
    ...
    def flush():
        body = "\n".join(cur_body).strip()
        if not body:
            return
        if is_noise(cur_heading, body, noise_markers) or _is_html_boilerplate(body):
            return
        ...
```

```python
# loader.py
from rag.chunking import chunk_text, chunk_markdown, _DEFAULT_NOISE_MARKERS

def load_chunks_from_dir(docs_dir, chunk_size, overlap, strategy="char",
                         noise_markers=_DEFAULT_NOISE_MARKERS):
    ...
        if strategy == "markdown":
            pieces = chunk_markdown(text, chunk_size=chunk_size, overlap=overlap,
                                    noise_markers=noise_markers)
        else:
            pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    ...
```

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_markdown_chunking.py tests/test_loader.py -v` → PASS
- [ ] **Step 5: 提交** — `git add src/rag/chunking.py src/rag/loader.py tests/test_markdown_chunking.py && git commit -m "feat: chunking/loader 注入 noise_markers"`

---

### Task 6: evaluate.py is_refusal 注入 marker

**Files:** Modify `src/rag/evaluate.py`; Test `tests/test_evaluate.py`

**Interfaces:**
- Produces: `is_refusal(answer_text, marker=_DEFAULT_REFUSAL_MARKER) -> bool`;`evaluate_sample(sample, retrieved, answer, scorer, refusal_marker=_DEFAULT_REFUSAL_MARKER)`。默认 "无法回答"。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_evaluate.py` 追加:

```python
def test_is_refusal_uses_injected_marker():
    from rag.evaluate import is_refusal
    assert is_refusal("这条资料没有", marker="资料没有") is True
    assert is_refusal("这条资料没有", marker="无法回答") is False
```

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_evaluate.py -k injected_marker -v` → FAIL

- [ ] **Step 3: 实现** — 在 `src/rag/evaluate.py`:把 `_REFUSAL_MARKER` 改名 `_DEFAULT_REFUSAL_MARKER`;

```python
def is_refusal(answer_text: str, marker: str = _DEFAULT_REFUSAL_MARKER) -> bool:
    return marker in answer_text


def evaluate_sample(sample, retrieved, answer, scorer,
                    refusal_marker: str = _DEFAULT_REFUSAL_MARKER) -> dict:
    negative = not sample.expected_sources
    refusal = is_refusal(answer.text, refusal_marker)
    ...  # 其余不变
```

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_evaluate.py -v` → PASS
- [ ] **Step 5: 提交** — `git add src/rag/evaluate.py tests/test_evaluate.py && git commit -m "feat: evaluate is_refusal 注入 marker"`

---

### Task 7: retrieve.py query_prefix + ingest.py doc_prefix

**Files:** Modify `src/rag/retrieve.py`, `src/rag/ingest.py`; Test `tests/test_retrieve.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `retrieve(..., query_prefix: str = "")`（在 embed_one 前拼 query_prefix + query）;`ingest_directory(..., doc_prefix: str = "")`（embed 前给每个 chunk 文本拼 doc_prefix）。默认空串 = 现状。

- [ ] **Step 1: 写失败测试**

`tests/test_retrieve.py` 追加(用捕获 embed_one 入参的假 embedder):

```python
def test_retrieve_prepends_query_prefix():
    from rag.retrieve import retrieve

    class _CapEmbedder:
        def __init__(self): self.seen = None
        def embed_one(self, text): self.seen = text; return [0.0]
        def embed(self, texts): return [[0.0] for _ in texts]

    class _EmptyStore:
        def search(self, query_vector, top_k, library=None): return []

    emb = _CapEmbedder()
    retrieve("路径参数", emb, _EmptyStore(), top_k=4, query_prefix="search_query: ")
    assert emb.seen == "search_query: 路径参数"
```

`tests/test_ingest.py` 追加(捕获 embed 入参):

```python
def test_ingest_prepends_doc_prefix():
    from rag.ingest import ingest_directory
    # 用文件里既有的 fake embedder/store 风格;断言喂给 embed 的文本以 doc_prefix 开头
    # (若已有 fake 不便捕获,构造最小捕获版 embedder 记录 embed() 收到的 texts)
    ...
    assert all(t.startswith("search_document: ") for t in captured_texts)
```

> 说明:`test_ingest.py` 里已有 fake provider;实现测试时复用它们或加一个记录 `embed()` 入参的最小假 embedder。断言核心:传入 `doc_prefix="search_document: "` 后,喂给 embedder 的每段文本都带该前缀。

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_retrieve.py tests/test_ingest.py -k prefix -v` → FAIL

- [ ] **Step 3: 实现**

`retrieve.py`:

```python
def retrieve(question, embedder, store, top_k, library=None,
             rewriter=None, reranker=None, rerank_factor=5,
             query_prefix: str = "") -> list[RetrievedChunk]:
    query = rewriter.rewrite(question) if rewriter is not None else question
    query_vector = embedder.embed_one(query_prefix + query)
    recall_k = top_k * rerank_factor if reranker is not None else top_k
    hits = store.search(query_vector, top_k=recall_k, library=library)
    if reranker is not None:
        return reranker.rerank(question, hits, top_k=top_k)
    return hits
```

`ingest.py`:

```python
def ingest_directory(docs_dir, embedder, store, chunk_size, overlap, vector_size,
                     strategy="char", doc_prefix: str = "") -> int:
    chunks = load_chunks_from_dir(docs_dir, chunk_size=chunk_size, overlap=overlap,
                                  strategy=strategy)
    if not chunks:
        return 0
    store.ensure_collection(vector_size=vector_size)
    vectors = embedder.embed([doc_prefix + c.text for c in chunks])
    store.upsert(chunks, vectors)
    return len(chunks)
```

- [ ] **Step 4: 确认通过** — `.venv/bin/python -m pytest tests/test_retrieve.py tests/test_ingest.py -v` → PASS
- [ ] **Step 5: 提交** — `git add src/rag/retrieve.py src/rag/ingest.py tests/test_retrieve.py tests/test_ingest.py && git commit -m "feat: ingest/retrieve 支持 embedding 前缀(默认空)"`

---

### Task 8: pipeline 透传 + 装配层注入 profile

**Files:** Modify `src/rag/pipeline.py`, `scripts/serve.py`, `scripts/eval.py`, `scripts/ingest.py`, `scripts/ask.py`, `scripts/ablate.py`, `scripts/gen_eval.py`; Test `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1–7 的所有注入参数 + `load_profile`。
- Produces: `RagPipeline(..., persona=..., refusal_text=..., query_prefix="")`,`ask` 把它们透传给 retrieve(query_prefix)与 generate_answer(persona, refusal_text);各 script 开头 `profile = load_profile(get_settings().profile)` 并注入到 rewriter(prompt)、reranker(无)、pipeline/agent(persona/refusal/query_prefix)、loader/ingest(noise_markers/doc_prefix)、eval(refusal_marker)。

- [ ] **Step 1: 写失败测试** — 在 `tests/test_pipeline.py` 追加(用捕获 prompt 的假 LLM,验证 profile 的 persona 流到生成):

```python
def test_pipeline_threads_persona_into_generation():
    from rag.pipeline import RagPipeline

    class _CapLLM:
        def __init__(self): self.seen = None
        def generate(self, prompt): self.seen = prompt; return "答"

    class _Emb:
        def embed_one(self, t): return [0.0]
        def embed(self, ts): return [[0.0] for _ in ts]

    class _Store:
        def search(self, v, top_k, library=None): return []

    llm = _CapLLM()
    p = RagPipeline(_Emb(), _Store(), llm, top_k=4,
                    persona="你是保险顾问", refusal_text="资料没有")
    p.ask("问题")
    assert "你是保险顾问" in llm.seen
```

- [ ] **Step 2: 确认失败** — `.venv/bin/python -m pytest tests/test_pipeline.py -k persona -v` → FAIL

- [ ] **Step 3: 实现**

`pipeline.py`——`RagPipeline.__init__` 增加 `persona`/`refusal_text`/`query_prefix`(默认复刻),`ask` 透传:

```python
def __init__(self, embedder, store, llm, top_k, rewriter=None, reranker=None,
             rerank_factor=5, persona=None, refusal_text=None, query_prefix=""):
    ...
    from rag.generate import _DEFAULT_PERSONA, _DEFAULT_REFUSAL
    self.persona = persona if persona is not None else _DEFAULT_PERSONA
    self.refusal_text = refusal_text if refusal_text is not None else _DEFAULT_REFUSAL
    self.query_prefix = query_prefix

def ask(self, question, library=None):
    chunks = retrieve(question, self.embedder, self.store, top_k=self.top_k,
                      library=library, rewriter=self.rewriter, reranker=self.reranker,
                      rerank_factor=self.rerank_factor, query_prefix=self.query_prefix)
    return generate_answer(question, chunks, self.llm,
                           persona=self.persona, refusal_text=self.refusal_text)
```

各 script 装配层——在每个 build/main 开头加载 profile 并注入。示例(`scripts/serve.py` 的 `build_app`):

```python
from rag.profile import load_profile
...
profile = load_profile(settings.profile)
rewriter = LlmQueryRewriter(llm, profile.rewrite_prompt) if (settings.query_rewrite and profile.rewrite_prompt) else None
reranker = LlmReranker(llm) if settings.rerank else None
pipeline = RagPipeline(embedder, store, llm, top_k=settings.top_k,
                       rewriter=rewriter, reranker=reranker, rerank_factor=settings.rerank_factor,
                       persona=profile.persona, refusal_text=profile.refusal_text,
                       query_prefix=profile.embed_query_prefix)
def retriever(query, library=None, top_k=settings.top_k):
    return retrieve(query, embedder, store, top_k=top_k, library=library,
                    rewriter=rewriter, reranker=reranker,
                    rerank_factor=settings.rerank_factor,
                    query_prefix=profile.embed_query_prefix)
agent = RagAgent(llm=build_chat_llm(settings), retriever=retriever, top_k=settings.top_k,
                 max_steps=settings.agent_max_steps,
                 persona=profile.persona, refusal_text=profile.refusal_text)
```

对应地:
- `scripts/eval.py`:同上装配 rewriter(profile.rewrite_prompt)/pipeline 路径不用(它直接调 retrieve+generate),要给 retrieve 传 `query_prefix=profile.embed_query_prefix`、给 generate_answer 传 persona/refusal_text、给 evaluate_sample 传 `refusal_marker=profile.refusal_marker`;agent 分支的 `"无法回答"` 判定改用 `profile.refusal_marker`。
- `scripts/ingest.py`:`load_chunks_from_dir(..., noise_markers=profile.noise_markers)` 由 ingest_directory 内部完成?否——ingest_directory 目前不收 noise_markers。**在 ingest_directory 增加 `noise_markers` 参数并透传给 loader**,scripts/ingest.py 传 `noise_markers=profile.noise_markers, doc_prefix=profile.embed_doc_prefix`。
- `scripts/ablate.py`:rewriter 用 profile.rewrite_prompt;retrieve 传 query_prefix;generate_answer 传 persona/refusal;evaluate_sample 传 refusal_marker。
- `scripts/gen_eval.py`:`load_chunks_from_dir(..., noise_markers=profile.noise_markers)`。
- `scripts/ask.py`:按其现有装配,注入 persona/refusal/query_prefix(参照 serve.py)。

> 补充:Task 7 的 `ingest_directory` 需再加一个 `noise_markers=_DEFAULT_NOISE_MARKERS` 参数透传给 `load_chunks_from_dir`(Task 5 已让 loader 支持)。实现 Task 8 时一并补上并在 `tests/test_ingest.py` 加一条"注入 noise_markers 透传到 loader"的断言。

- [ ] **Step 4: 确认通过 + 全量绿 + 脚本语法**

```bash
cd /Users/mi/Documents/test/rag-docs
.venv/bin/python -m pytest -q
for f in serve eval ingest ask ablate gen_eval; do .venv/bin/python -c "import ast; ast.parse(open('scripts/$f.py').read())" && echo "$f ok"; done
```
Expected: 全量 PASS;6 个脚本各打印 `ok`。

- [ ] **Step 5: 提交** — `git add -A && git commit -m "feat: pipeline 透传 + 装配层加载并注入 profile"`

---

### Task 9: 启用 nomic 前缀 + 重灌库 + eval 前后对比(实机度量)

> **本任务需 Ollama + Qdrant 在线,由控制器/操作者交互执行,非 TDD 子代理任务。** 它验证 M8 的唯一行为变化(embedding 前缀),并据结果决定保留或回滚。

**Files:** Modify `profiles/tech.toml`(设前缀); `notes/07-检索优化.md`(记录结果)

- [ ] **Step 1: 记录前缀关的 baseline**(已实测,直接抄):`hit@4 94.4% / recall@4 0.944 / precision@4 0.597 / nDCG@4 0.795 / MRR 0.759 / 拒答正确率 86.4%`(default profile 前缀为空时的数)。

- [ ] **Step 2: 开启前缀** — 编辑 `profiles/tech.toml`:
```toml
embed_query_prefix = "search_query: "
embed_doc_prefix = "search_document: "
```

- [ ] **Step 3: 重灌库**（doc 现在会带前缀）:
```bash
cd /Users/mi/Documents/test/rag-docs && .venv/bin/python scripts/ingest.py
```

- [ ] **Step 4: 跑 eval(前缀开)**:
```bash
.venv/bin/python scripts/eval.py
```
记下新指标。

- [ ] **Step 5: 判定并记录** — 在 `notes/07-检索优化.md` 追加一张"前缀关 vs 前缀开"对比表 + 结论。
  - 若指标整体上升 → 保留(tech.toml 前缀保持设置)。
  - 若下降或无变化 → **回滚**:tech.toml 前缀清空,再重灌一次库恢复原状。
  - 提交:`git add profiles/tech.toml notes/07-检索优化.md && git commit -m "chore(M8): 度量 embedding 前缀效果并据 eval 定去留"`

---

## Self-Review

**Spec coverage:** DomainProfile+load_profile(§1→T1)、各模块注入(§2:generate T2/agent T3/rewrite T4/chunking+loader T5/evaluate T6/前缀 T7)、装配层(§3→T8)、度量(§4→T9)、测试(§5 分散各任务)、影响文件(§6 全覆盖)、非目标(§7 未越界)、诚实提醒(§8:前缀重灌绑定见 T9,默认复刻见全局约束)。✅

**Placeholder scan:** test_ingest 前缀/noise 断言给了"复用既有 fake 或加最小捕获版"的明确指引,非占位符;其余步骤均含实际代码/命令。✅

**Type consistency:**
- `_DEFAULT_PERSONA`/`_DEFAULT_REFUSAL` 在 generate.py(T2)定义,pipeline(T8)从 generate import 复用;agent(T3)自带同名默认常量(各自模块内,值相同)。✅
- 新参数默认值处处 = 现状(persona/refusal/prompt/markers/marker/prefix),保证每步全量绿。✅
- `ingest_directory` 的 `noise_markers` 参数在 T8 补充说明中明确加入(T7 只加 doc_prefix;T8 补 noise_markers 透传),已在 T8 Step 3 与"补充"注明,避免 T5(loader 支持)与调用方脱节。✅
- profile 前缀:模型默认 ""(T1)、tech.toml T1–T8 留空、T9 才设值——与全局约束"避免失配"一致。✅
