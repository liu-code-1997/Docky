# M10 Listwise 重排 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** 加 Listwise LLM 重排(一次调用排序,零依赖)+ build_reranker 工厂,为将来 bge cross-encoder 预留扩展点。

**Spec:** `specs/2026-09-13-m10-listwise-rerank-design.md`

## Global Constraints
- `.venv/bin/python -m pytest`;零新依赖;rerank 总开关默认关,行为不变;可测逻辑在 src/rag。conventional commits。预存 warning 忽略。

---

### Task 1: ListwiseLlmReranker + build_reranker + config

**Files:** Modify `src/rag/rerank.py`, `src/rag/config.py`; Test `tests/test_reranker.py`

**Interfaces:** `ListwiseLlmReranker(llm)` 实现 `Reranker.rerank`;`build_reranker(provider, llm)->Reranker`;`Settings.rerank_provider="llm_listwise"`。

- [ ] **Step 1: 写失败测试** 在 `tests/test_reranker.py` 追加:

```python
from rag.rerank import ListwiseLlmReranker, build_reranker, LlmReranker
from rag.models import Chunk, RetrievedChunk


def _rc(cid, text="t"):
    return RetrievedChunk(chunk=Chunk(id=cid, text=text, source=f"{cid}.md",
                                      library="l", chunk_index=0), score=1.0)


class _FakeLLM:
    def __init__(self, reply): self.reply = reply
    def generate(self, prompt): return self.reply


def test_listwise_reorders_by_llm_output():
    cands = [_rc("a"), _rc("b"), _rc("c")]
    r = ListwiseLlmReranker(_FakeLLM("2,3,1"))
    out = r.rerank("q", cands, top_k=3)
    assert [c.chunk.source for c in out] == ["b.md", "c.md", "a.md"]


def test_listwise_appends_unmentioned_and_truncates():
    cands = [_rc("a"), _rc("b"), _rc("c")]
    r = ListwiseLlmReranker(_FakeLLM("3"))          # 只提到 c
    out = r.rerank("q", cands, top_k=2)
    assert out[0].chunk.source == "c.md"            # 提到的在前
    assert len(out) == 2                             # 其余按原序补,截断到 top_k


def test_listwise_unparseable_falls_back_to_original_order():
    cands = [_rc("a"), _rc("b")]
    r = ListwiseLlmReranker(_FakeLLM("乱码没有数字"))
    out = r.rerank("q", cands, top_k=2)
    assert [c.chunk.source for c in out] == ["a.md", "b.md"]


def test_listwise_empty():
    assert ListwiseLlmReranker(_FakeLLM("")).rerank("q", [], top_k=4) == []


def test_build_reranker_dispatch():
    assert isinstance(build_reranker("llm_listwise", _FakeLLM("")), ListwiseLlmReranker)
    assert isinstance(build_reranker("llm_pointwise", _FakeLLM("")), LlmReranker)
    import pytest
    with pytest.raises(ValueError):
        build_reranker("nope", _FakeLLM(""))
```

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_reranker.py -v` → FAIL
- [ ] **Step 3: 实现** 在 `src/rag/rerank.py` 追加(保留现有 LlmReranker):

```python
_LISTWISE_PROMPT = """下面是若干候选资料,请按与【问题】的相关性从高到低排序。
只输出候选编号,用逗号分隔(如 3,1,4,2),不要解释、不要输出其它内容。

【问题】{question}

{candidates}
排序:"""


def _parse_order(reply: str, n: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for tok in re.findall(r"\d+", reply):
        i = int(tok)
        if 1 <= i <= n and i not in seen:
            seen.add(i)
            order.append(i)
    return order


class ListwiseLlmReranker(Reranker):
    """一次 LLM 调用对全部候选排序(比逐条打分快,零新依赖)。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def rerank(self, question: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        blocks = "\n".join(f"[{i}] {rc.chunk.text[:300]}"
                           for i, rc in enumerate(candidates, start=1))
        reply = self.llm.generate(
            _LISTWISE_PROMPT.format(question=question, candidates=blocks))
        order = _parse_order(reply, len(candidates))
        seen = set(order)
        order += [i for i in range(1, len(candidates) + 1) if i not in seen]  # 未提及者补末尾
        ranked = [candidates[i - 1] for i in order]
        return ranked[:top_k]


def build_reranker(provider: str, llm: LLM) -> Reranker:
    """按名字构造重排器。将来接 bge cross-encoder 只需加一分支。"""
    if provider == "llm_listwise":
        return ListwiseLlmReranker(llm)
    if provider == "llm_pointwise":
        return LlmReranker(llm)
    raise ValueError(f"未知 rerank_provider: {provider}(可选 llm_listwise | llm_pointwise)")
```

在 `src/rag/config.py` 的 rerank 字段附近加:`rerank_provider: str = "llm_listwise"`。

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_reranker.py -v` → PASS;全量 `.venv/bin/python -m pytest -q` 绿。
- [ ] **Step 5: 提交** `git add src/rag/rerank.py src/rag/config.py tests/test_reranker.py && git commit -m "feat(M10): ListwiseLlmReranker + build_reranker 工厂 + rerank_provider 配置"`

---

### Task 2: 装配层用 build_reranker

**Files:** Modify `scripts/serve.py`, `scripts/eval.py`, `scripts/ablate.py`

**Interfaces:** 消费 Task 1 的 build_reranker + settings.rerank_provider。

- [ ] **Step 1: 改装配** 三个脚本里,把 `LlmReranker(llm) if <rerank开关> else None` 全部替换为
  `build_reranker(settings.rerank_provider, llm) if <rerank开关> else None`,并把 import 从
  `from rag.rerank import LlmReranker` 改为 `from rag.rerank import build_reranker`(若某脚本仍需 LlmReranker 名字则一并 import;实际只用 build_reranker)。
  - `serve.py`:`settings.rerank` 分支。
  - `eval.py`:`use_rerank` 分支。
  - `ablate.py`:循环内 `use_rerank` 分支。
- [ ] **Step 2: 静态校验 + 全量**
```bash
cd /Users/mi/Documents/test/rag-docs
for f in serve eval ablate; do .venv/bin/python -c "import ast; ast.parse(open('scripts/$f.py').read())" && echo "$f ok"; done
.venv/bin/python -m pytest -q
```
- [ ] **Step 3: 提交** `git add scripts/serve.py scripts/eval.py scripts/ablate.py && git commit -m "feat(M10): 装配层改用 build_reranker"`

---

### Task 3: 度量 rerank 关/开(listwise)(实机,控制器执行)

- [ ] **Step 1: eval rerank 关**(基线,已知):hit 94.2 / recall 0.936 / nDCG 0.823 / MRR 0.796。
- [ ] **Step 2: eval rerank 开(listwise)** `RERANK=true .venv/bin/python scripts/eval.py`,记指标。
- [ ] **Step 3: 记录判定** notes/07 追加 rerank 关/开对比 + 结论。升则可设推荐;平/降则默认关(机制+扩展点保留)。提交 notes(+ 若改默认则 config)。

---

## Self-Review
- Spec §1 ListwiseLlmReranker→T1;§2 build_reranker+config→T1;§3 装配→T2;§4 度量→T3;§5 测试→T1;§6 文件全覆盖;§7 非目标未越界(不装 bge)。
- 兜底:解析失败→原序 top_k(不比不重排差);未提及候选补末尾(不丢)。
- 类型:`build_reranker(provider:str, llm)->Reranker` 在 T1 定,T2 装配调用一致;rerank 总开关默认关,行为不变。
