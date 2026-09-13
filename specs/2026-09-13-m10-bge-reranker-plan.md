# M10 bge cross-encoder Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** 实现 CrossEncoderReranker(bge-reranker),挂进 build_reranker 工厂,作为可选重排 provider。惰性加载,默认不启用。

**Spec:** `specs/2026-09-13-m10-bge-reranker-design.md`

## Global Constraints
- `.venv/bin/python -m pytest`(venv=3.12;sentence-transformers/torch 已装)。
- torch/sentence_transformers **只在 CrossEncoderReranker 首次 rerank 时 import**(惰性);单测靠注入假 scorer,不加载真模型。
- rerank 默认关;rerank_provider 默认 llm_listwise;既有 LlmReranker/ListwiseLlmReranker 不动。conventional commits。

---

### Task 1: CrossEncoderReranker + build_reranker + config + dep

**Files:** Modify `src/rag/rerank.py`, `src/rag/config.py`, `pyproject.toml`; Test `tests/test_reranker.py`

**Interfaces:** `CrossEncoderReranker(model_name, scorer=None)` 实现 Reranker;`build_reranker(provider, llm, cross_encoder_model="BAAI/bge-reranker-v2-m3")` 加 cross_encoder 分支;`Settings.rerank_cross_encoder_model`。

- [ ] **Step 1: 写失败测试** 在 `tests/test_reranker.py` 追加:
```python
def test_cross_encoder_reranks_by_injected_scorer():
    from rag.rerank import CrossEncoderReranker
    cands = [_rc("a"), _rc("b"), _rc("c")]      # 复用文件里已有的 _rc helper
    # 注入假 scorer:给 pairs 打分,让 c>a>b
    def fake_scorer(pairs):
        # pairs = [(q, text), ...] 顺序同 candidates
        return [0.2, 0.1, 0.9]                   # a,b,c
    r = CrossEncoderReranker("dummy-model", scorer=fake_scorer)
    out = r.rerank("q", cands, top_k=2)
    assert [c.chunk.source for c in out] == ["c.md", "a.md"]   # 按分降序取 top2
    assert out[0].score == 0.9


def test_cross_encoder_empty():
    from rag.rerank import CrossEncoderReranker
    assert CrossEncoderReranker("m", scorer=lambda p: []).rerank("q", [], top_k=4) == []


def test_build_reranker_cross_encoder_no_model_load():
    from rag.rerank import build_reranker, CrossEncoderReranker
    r = build_reranker("cross_encoder", llm=None, cross_encoder_model="BAAI/bge-reranker-v2-m3")
    assert isinstance(r, CrossEncoderReranker)
    assert r._model is None                      # 惰性:构造不加载模型
```
(注:`_rc` helper 若文件里签名不同,按现有的用;确保返回带 .chunk.source/.chunk.text 的 RetrievedChunk。)

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_reranker.py -k cross_encoder -v` → FAIL
- [ ] **Step 3: 实现** 在 `src/rag/rerank.py` 追加:
```python
class CrossEncoderReranker(Reranker):
    """bge 类 cross-encoder 重排。torch/sentence_transformers 惰性加载(首次 rerank 时)。"""

    def __init__(self, model_name: str, scorer=None):
        self.model_name = model_name
        self._scorer = scorer          # callable(list[tuple[str,str]])->list[float];测试注入
        self._model = None

    def _score(self, pairs):
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model.predict(pairs)

    def rerank(self, question, candidates, top_k):
        if not candidates:
            return []
        pairs = [(question, rc.chunk.text) for rc in candidates]
        scores = self._score(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda t: t[0], reverse=True)
        return [RetrievedChunk(chunk=rc.chunk, score=float(s)) for s, rc in ranked[:top_k]]
```
`build_reranker` 加参数 + 分支:
```python
def build_reranker(provider, llm, cross_encoder_model="BAAI/bge-reranker-v2-m3"):
    if provider == "llm_listwise": return ListwiseLlmReranker(llm)
    if provider == "llm_pointwise": return LlmReranker(llm)
    if provider == "cross_encoder": return CrossEncoderReranker(cross_encoder_model)
    raise ValueError(f"未知 rerank_provider: {provider}(可选 llm_listwise | llm_pointwise | cross_encoder)")
```
`config.py` 加:`rerank_cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"`。
`pyproject.toml` dependencies 加:`"sentence-transformers>=3",   # M10 bge cross-encoder 重排(惰性加载)`。

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_reranker.py -v` → PASS;全量 `.venv/bin/python -m pytest -q` 绿(不应触发 torch 加载——惰性)。
- [ ] **Step 5: 提交** `git add src/rag/rerank.py src/rag/config.py pyproject.toml tests/test_reranker.py && git commit -m "feat(M10): CrossEncoderReranker(bge)+ 工厂分支 + sentence-transformers 依赖(惰性)"`

---

### Task 2: 装配传 cross_encoder_model

**Files:** Modify `scripts/{serve,eval,ablate}.py`

- [ ] **Step 1: 改装配** 三脚本里 `build_reranker(settings.rerank_provider, llm)` → `build_reranker(settings.rerank_provider, llm, settings.rerank_cross_encoder_model)`。
- [ ] **Step 2: 校验** `for f in serve eval ablate; do .venv/bin/python -c "import ast; ast.parse(open('scripts/$f.py').read())" && echo "$f ok"; done`;全量绿。
- [ ] **Step 3: 提交** `git add scripts/serve.py scripts/eval.py scripts/ablate.py && git commit -m "feat(M10): 装配传 rerank_cross_encoder_model"`

---

### Task 3: 度量 rerank provider 对比(实机,控制器执行,首次下模型)

- [ ] eval 三组(同库):`RERANK=false`(基线)/ `RERANK=true RERANK_PROVIDER=llm_listwise` / `RERANK=true RERANK_PROVIDER=cross_encoder`(首次会下 bge-v2-m3 ~2.3GB)。记 nDCG/MRR/生成分。
- [ ] notes/07 追加三 provider 对比 + 结论。bge 胜 listwise → 推荐 provider;否则 listwise 默认、bge 可选。

---

## Self-Review
- Spec §1 CrossEncoderReranker→T1;§2 工厂→T1;§3 config→T1;§4 dep→T1;§5 装配→T2;§7 度量→T3。
- 惰性:__init__ 不加载(scorer/_model);测试注入 scorer 不碰 torch(T1 test_build_reranker_cross_encoder_no_model_load 验 _model is None)。
- 回归:build_reranker 既有两分支不变;rerank 默认关;不用 cross_encoder 则 torch 不 import。
- 类型:scorer callable(list[(str,str)])->list[float];rerank 返回 list[RetrievedChunk] 带新分。
