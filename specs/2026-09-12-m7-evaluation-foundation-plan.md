# M7 评估地基 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把评估侧从「单来源 hit@k/MRR」升级为「多来源 recall@k/nDCG/precision@k + 负例拒答正确率」,并新增评估集生成工具与一键消融对比表,让后续检索/生成优化有可信标尺。

**Architecture:** 只改评估侧,不动 retrieve/generate/pipeline/agent 链路。数据模型 `EvalSample` 由单来源升多来源(带旧字段兼容);`evaluate.py` 扩展指标并把正例/负例分组聚合;新增 `src/rag/gen_eval.py`(从 docs 出候选题)与 `src/rag/ablate.py`(对比表格式化)两块纯逻辑,各配一个 `scripts/` 薄 CLI。

**Tech Stack:** Python 3.11+、pydantic v2、pytest(pythonpath=src)。不引入任何新第三方依赖。

**Spec:** `specs/2026-09-11-m7-evaluation-foundation-design.md`

## Global Constraints

- Python `>=3.11`;测试用 `pytest`,`pythonpath=["src"]`、`testpaths=["tests"]`(见 pyproject.toml)。
- **不改动检索/生成链路**:`retrieve.py` / `generate.py` / `pipeline.py` / `agent.py` / `interfaces.py` 的 `VectorStore` 一行不动。
- **不新增第三方依赖**:gen_eval / ablate 复用现有 `LLM` provider 与 `loader`。
- 提交信息用 conventional commits(与仓库历史一致:`feat:` / `refactor:` / `test:` / `docs:` / `chore:`)。
- 可测逻辑放 `src/rag/`,`scripts/` 只做装配(沿用 `evaluate.py` ↔ `scripts/eval.py` 的既有分工)。

---

### Task 1: EvalSample 升级为多来源(带旧字段兼容)

**Files:**
- Modify: `src/rag/models.py`(`EvalSample` 定义)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: 无
- Produces: `EvalSample.expected_sources: list[str]`(正例=相关来源列表,负例=`[]`);加载旧字段 `expected_source`(str 或 null)时自动映射为 `expected_sources`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_models.py` 末尾追加:

```python
from rag.models import EvalSample


def test_eval_sample_multi_source():
    s = EvalSample(question="q", expected_sources=["a.md", "b.md"])
    assert s.expected_sources == ["a.md", "b.md"]


def test_eval_sample_negative_defaults_empty():
    s = EvalSample(question="q")
    assert s.expected_sources == []


def test_eval_sample_compat_old_single_source():
    s = EvalSample(question="q", expected_source="a.md")
    assert s.expected_sources == ["a.md"]


def test_eval_sample_compat_old_null_source():
    s = EvalSample(question="q", expected_source=None)
    assert s.expected_sources == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_models.py -k eval_sample -v`
Expected: FAIL(`expected_sources` 不存在 / 传入 `expected_source` 报未知字段或未映射)

- [ ] **Step 3: 改 EvalSample**

在 `src/rag/models.py`,把 `EvalSample` 替换为:

```python
class EvalSample(BaseModel):
    """评估集的一条样本(M4;M7 升多来源)。"""
    question: str
    expected_sources: list[str] = []          # 相关来源文件;负例(拒答样本)为 []
    expected_keywords: list[str] = []         # 方法A:答案里应出现的关键词
    expected_answer: str = ""                 # 方法B/C:标准答案(A 用不到)

    @model_validator(mode="before")
    @classmethod
    def _compat_expected_source(cls, data):
        """兼容旧字段 expected_source(str|None)→ expected_sources。
        迁移完 dataset.json 后仍保留,以兜住任何遗留旧格式输入。"""
        if isinstance(data, dict) and "expected_source" in data:
            data = dict(data)
            src = data.pop("expected_source")
            data.setdefault("expected_sources", [src] if src else [])
        return data
```

并把文件顶部的导入改为:

```python
from pydantic import BaseModel, model_validator
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_models.py -k eval_sample -v`
Expected: PASS(4 条)

- [ ] **Step 5: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/models.py tests/test_models.py
git commit -m "feat: EvalSample 支持多来源 expected_sources(兼容旧字段)"
```

---

### Task 2: 迁移 eval/dataset.json 到多来源

**Files:**
- Modify: `eval/dataset.json`(22 条)

**Interfaces:**
- Consumes: Task 1 的 `EvalSample`(兼容读入旧字段)
- Produces: 每条含 `expected_sources`(单来源→单元素列表,`null`→`[]`),不再含 `expected_source`。

- [ ] **Step 1: 跑迁移脚本(用兼容加载 → 重新序列化)**

Run:

```bash
cd /Users/mi/Documents/test/rag-docs && python -c "
import json
from pathlib import Path
from rag.models import EvalSample

p = Path('eval/dataset.json')
data = json.loads(p.read_text(encoding='utf-8'))
samples = [EvalSample(**d) for d in data]
p.write_text(json.dumps([s.model_dump() for s in samples],
                        ensure_ascii=False, indent=2) + '\n',
             encoding='utf-8')
print('migrated', len(samples), 'samples')
"
```

Expected: 打印 `migrated 22 samples`

- [ ] **Step 2: 校验迁移结果**

Run:

```bash
cd /Users/mi/Documents/test/rag-docs && python -c "
import json
d = json.load(open('eval/dataset.json', encoding='utf-8'))
assert all('expected_sources' in x for x in d), '缺 expected_sources'
assert all('expected_source' not in x for x in d), '仍有旧字段'
pos = [x for x in d if x['expected_sources']]
neg = [x for x in d if not x['expected_sources']]
print('positives', len(pos), 'negatives', len(neg))
assert len(pos) == 18 and len(neg) == 4, (len(pos), len(neg))
print('OK')
"
```

Expected: `positives 18 negatives 4` 然后 `OK`

- [ ] **Step 3: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add eval/dataset.json
git commit -m "refactor: 迁移评估集为多来源 expected_sources"
```

---

### Task 3: 多来源检索排序指标(hit/mrr/recall/precision/ndcg)

**Files:**
- Modify: `src/rag/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `EvalSample.expected_sources`、`RetrievedChunk`(`rc.chunk.source`)
- Produces:
  - `hit_at_k(expected_sources: list[str], retrieved: list[RetrievedChunk]) -> bool`
  - `reciprocal_rank(expected_sources: list[str], retrieved) -> float`
  - `recall_at_k(expected_sources, retrieved) -> float`
  - `precision_at_k(expected_sources, retrieved) -> float`(分母为 `len(retrieved)`,即返回的 top-k 数)
  - `ndcg_at_k(expected_sources, retrieved) -> float`(二元相关性;IDCG 用 top-k 内相关数为理想上界)

- [ ] **Step 1: 写失败测试**

把 `tests/test_evaluate.py` 中**检索指标相关的旧测试整体替换**为下面这段(保留文件里与打分器无关的其它测试;若旧测试引用 `expected_source`/旧 `hit_at_k` 签名,一并删除):

```python
import math

from rag.evaluate import (
    hit_at_k, reciprocal_rank, recall_at_k, precision_at_k, ndcg_at_k,
)
from rag.models import Chunk, RetrievedChunk


def _rc(source: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::0", text="t", source=source,
                    library="lib", chunk_index=0),
        score=score,
    )


def test_hit_at_k_any_relevant():
    retrieved = [_rc("x.md"), _rc("a.md")]
    assert hit_at_k(["a.md", "b.md"], retrieved) is True
    assert hit_at_k(["z.md"], retrieved) is False


def test_reciprocal_rank_first_relevant_position():
    retrieved = [_rc("x.md"), _rc("a.md"), _rc("b.md")]
    assert reciprocal_rank(["b.md"], retrieved) == 1.0 / 3
    assert reciprocal_rank(["a.md"], retrieved) == 1.0 / 2
    assert reciprocal_rank(["z.md"], retrieved) == 0.0


def test_recall_at_k_distinct_sources():
    retrieved = [_rc("a.md"), _rc("a.md"), _rc("x.md")]
    # 相关源 {a,b},召回命中 {a} → 1/2
    assert recall_at_k(["a.md", "b.md"], retrieved) == 0.5
    assert recall_at_k(["a.md"], retrieved) == 1.0


def test_precision_at_k_relevant_chunks_over_k():
    retrieved = [_rc("a.md"), _rc("x.md"), _rc("b.md"), _rc("y.md")]
    # 4 个里 2 个相关 → 0.5
    assert precision_at_k(["a.md", "b.md"], retrieved) == 0.5


def test_ndcg_at_k_rewards_higher_rank():
    top = [_rc("a.md"), _rc("x.md")]     # 相关在第1位
    low = [_rc("x.md"), _rc("a.md")]     # 相关在第2位
    assert ndcg_at_k(["a.md"], top) == 1.0
    assert ndcg_at_k(["a.md"], low) == math.log2(2) / math.log2(3)


def test_metrics_empty_and_no_hit_edges():
    assert recall_at_k(["a.md"], []) == 0.0
    assert precision_at_k(["a.md"], []) == 0.0
    assert ndcg_at_k(["a.md"], [_rc("x.md")]) == 0.0   # n_rel=0
    assert hit_at_k([], [_rc("a.md")]) is False        # 负例无排序指标
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_evaluate.py -k "hit_at_k or reciprocal or recall or precision or ndcg or edges" -v`
Expected: FAIL(`recall_at_k` / `precision_at_k` / `ndcg_at_k` 未定义)

- [ ] **Step 3: 实现指标**

把 `src/rag/evaluate.py` 顶部导入与四个指标函数替换/新增为:

```python
"""评估逻辑(M4;M7 升多来源)。检索层指标 + 生成层评分 + 聚合,全为纯函数。"""
import math

from rag.interfaces import AnswerScorer
from rag.models import Answer, EvalSample, RetrievedChunk

_REFUSAL_MARKER = "无法回答"


def _relevance_flags(expected_sources: list[str],
                     retrieved: list[RetrievedChunk]) -> list[bool]:
    """逐个检索块是否相关(source ∈ expected_sources)。"""
    relevant = set(expected_sources)
    return [rc.chunk.source in relevant for rc in retrieved]


def hit_at_k(expected_sources: list[str],
             retrieved: list[RetrievedChunk]) -> bool:
    """top-k 中出现任一相关来源即命中。"""
    if not expected_sources:
        return False
    return any(_relevance_flags(expected_sources, retrieved))


def reciprocal_rank(expected_sources: list[str],
                    retrieved: list[RetrievedChunk]) -> float:
    """第一个相关块的倒数排名;无则 0。"""
    if not expected_sources:
        return 0.0
    for i, rel in enumerate(_relevance_flags(expected_sources, retrieved),
                            start=1):
        if rel:
            return 1.0 / i
    return 0.0


def recall_at_k(expected_sources: list[str],
                retrieved: list[RetrievedChunk]) -> float:
    """top-k 命中的不同相关来源数 / 相关来源总数。"""
    if not expected_sources:
        return 0.0
    retrieved_sources = {rc.chunk.source for rc in retrieved}
    hit_sources = retrieved_sources & set(expected_sources)
    return len(hit_sources) / len(set(expected_sources))


def precision_at_k(expected_sources: list[str],
                   retrieved: list[RetrievedChunk]) -> float:
    """top-k 中相关块数 / 返回块数(k)。"""
    if not expected_sources or not retrieved:
        return 0.0
    flags = _relevance_flags(expected_sources, retrieved)
    return sum(flags) / len(retrieved)


def ndcg_at_k(expected_sources: list[str],
              retrieved: list[RetrievedChunk]) -> float:
    """二元相关性 nDCG@k;IDCG 用 top-k 内相关数为理想上界(见 spec 诚实说明)。"""
    if not expected_sources or not retrieved:
        return 0.0
    flags = _relevance_flags(expected_sources, retrieved)
    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2)
              for i, rel in enumerate(flags))
    n_rel = sum(flags)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg
```

> 注意:保留文件里原有的 `is_refusal` / `evaluate_sample` / `aggregate`(Task 4 再改)。若原有旧的 `hit_at_k` / `reciprocal_rank`(单来源签名)存在,用上面的新版覆盖。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_evaluate.py -k "hit_at_k or reciprocal or recall or precision or ndcg or edges" -v`
Expected: PASS(6 条)

- [ ] **Step 5: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/evaluate.py tests/test_evaluate.py
git commit -m "feat: 多来源检索指标 recall@k/precision@k/nDCG@k"
```

---

### Task 4: 负例拒答正确性 + evaluate_sample/aggregate 分组

**Files:**
- Modify: `src/rag/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 3 的指标函数、`AnswerScorer.score`
- Produces:
  - `is_refusal(answer_text: str) -> bool`(不变)
  - `evaluate_sample(sample, retrieved, answer, scorer) -> dict`:含 `negative`/`refusal`/`refusal_correct`;正例另含 `hit`/`mrr`/`recall`/`precision`/`ndcg`/`gen_score`
  - `aggregate(rows) -> dict`:键 `n`/`n_positive`/`n_negative`/`hit_rate`/`avg_mrr`/`avg_recall`/`avg_precision`/`avg_ndcg`/`avg_gen_score`/`refusal_accuracy`

- [ ] **Step 1: 写失败测试**

在 `tests/test_evaluate.py` 追加:

```python
from rag.evaluate import evaluate_sample, aggregate, is_refusal
from rag.models import Answer, EvalSample


class _FixedScorer:
    def score(self, answer_text, sample):
        return 1.0


def test_evaluate_sample_positive():
    sample = EvalSample(question="q", expected_sources=["a.md"])
    retrieved = [_rc("a.md"), _rc("x.md")]
    row = evaluate_sample(sample, retrieved, Answer(text="答案"), _FixedScorer())
    assert row["negative"] is False
    assert row["hit"] is True
    assert row["gen_score"] == 1.0
    assert row["refusal_correct"] is True    # 正例未拒答 = 正确


def test_evaluate_sample_positive_wrong_refusal():
    sample = EvalSample(question="q", expected_sources=["a.md"])
    row = evaluate_sample(sample, [_rc("a.md")],
                          Answer(text="根据现有资料无法回答"), _FixedScorer())
    assert row["refusal_correct"] is False   # 正例误拒 = 错


def test_evaluate_sample_negative_refusal_correct():
    sample = EvalSample(question="q")         # 负例
    row = evaluate_sample(sample, [],
                          Answer(text="根据现有资料无法回答"), _FixedScorer())
    assert row["negative"] is True
    assert row["refusal_correct"] is True     # 负例拒答 = 正确
    assert "hit" not in row                    # 负例不含排序指标


def test_aggregate_splits_positive_negative():
    rows = [
        {"negative": False, "hit": True, "mrr": 1.0, "recall": 1.0,
         "precision": 0.5, "ndcg": 1.0, "gen_score": 0.8,
         "refusal": False, "refusal_correct": True},
        {"negative": True, "refusal": True, "refusal_correct": True},
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["n_positive"] == 1
    assert agg["n_negative"] == 1
    assert agg["hit_rate"] == 1.0
    assert agg["avg_gen_score"] == 0.8
    assert agg["refusal_accuracy"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_evaluate.py -k "evaluate_sample or aggregate_splits" -v`
Expected: FAIL(`evaluate_sample` 仍是旧的单来源版本,断言不符)

- [ ] **Step 3: 实现分组聚合**

把 `src/rag/evaluate.py` 里的 `is_refusal` / `evaluate_sample` / `aggregate` 替换为:

```python
def is_refusal(answer_text: str) -> bool:
    """答案是否为拒答。"""
    return _REFUSAL_MARKER in answer_text


def evaluate_sample(sample: EvalSample, retrieved: list[RetrievedChunk],
                    answer: Answer, scorer: AnswerScorer) -> dict:
    """评估一条:正例算排序指标+生成分,负例只算拒答正确性。"""
    negative = not sample.expected_sources
    refusal = is_refusal(answer.text)
    row: dict = {
        "question": sample.question,
        "negative": negative,
        "refusal": refusal,
    }
    if negative:
        row["refusal_correct"] = refusal          # 负例应拒答
    else:
        es = sample.expected_sources
        row["hit"] = hit_at_k(es, retrieved)
        row["mrr"] = reciprocal_rank(es, retrieved)
        row["recall"] = recall_at_k(es, retrieved)
        row["precision"] = precision_at_k(es, retrieved)
        row["ndcg"] = ndcg_at_k(es, retrieved)
        row["gen_score"] = scorer.score(answer.text, sample)
        row["refusal_correct"] = not refusal      # 正例不应误拒
    return row


def aggregate(rows: list[dict]) -> dict:
    """正例算排序指标+生成分,拒答正确率跨全体;两组分开、互不污染。"""
    positives = [r for r in rows if not r["negative"]]
    negatives = [r for r in rows if r["negative"]]

    def avg(items: list[dict], key: str) -> float:
        return sum(i[key] for i in items) / len(items) if items else 0.0

    return {
        "n": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "hit_rate": avg(positives, "hit"),
        "avg_mrr": avg(positives, "mrr"),
        "avg_recall": avg(positives, "recall"),
        "avg_precision": avg(positives, "precision"),
        "avg_ndcg": avg(positives, "ndcg"),
        "avg_gen_score": avg(positives, "gen_score"),
        "refusal_accuracy": avg(rows, "refusal_correct"),
    }
```

- [ ] **Step 4: 运行全部 evaluate 测试确认通过**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_evaluate.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate_sample/aggregate 正负例分组 + 拒答正确率"
```

---

### Task 5: 更新 scripts/eval.py 输出到新指标

**Files:**
- Modify: `scripts/eval.py`(打印逐条与汇总部分;装配逻辑不变)

**Interfaces:**
- Consumes: Task 4 的 `evaluate_sample` / `aggregate` 新返回结构
- Produces: 单次 eval CLI 打印新指标;agent 分支构造与 `aggregate` 兼容的 row。

- [ ] **Step 1: 更新单轮分支的逐条打印**

在 `scripts/eval.py` 的 `else`(单轮)分支里,把逐条 `print(...)` 替换为:

```python
            retrieved = retrieve(s.question, embedder, store,
                                 top_k=settings.top_k, library=None,
                                 rewriter=rewriter, reranker=reranker,
                                 rerank_factor=settings.rerank_factor)
            ans = generate_answer(s.question, retrieved, llm)
            r = evaluate_sample(s, retrieved, ans, scorer)
            rows.append(r)
            if r["negative"]:
                print(f"[neg] {'拒答✓' if r['refusal_correct'] else '误答✗'}  {s.question}")
            else:
                print(f"hit={'✓' if r['hit'] else '✗'} "
                      f"recall={r['recall']:.2f} ndcg={r['ndcg']:.2f} "
                      f"gen={r['gen_score']:.2f}  {s.question}")
```

- [ ] **Step 2: 更新 agent 分支的 row(与 aggregate 兼容)**

把 agent 分支里 `rows.append({...})` 替换为(agent 不单独计检索命中,按负例结构只上报拒答与生成分不适用,这里按正例但排序指标置 0):

```python
            ans = agent.ask(s.question, library=None)
            gen = scorer.score(ans.text, s)
            refusal = "无法回答" in ans.text
            negative = not s.expected_sources
            row = {"question": s.question, "negative": negative,
                   "refusal": refusal,
                   "refusal_correct": refusal if negative else not refusal}
            if not negative:
                row.update({"hit": False, "mrr": 0.0, "recall": 0.0,
                            "precision": 0.0, "ndcg": 0.0, "gen_score": gen})
            rows.append(row)
            print(f"gen={gen:>4.2f} {'拒答' if refusal else '答  '}  {s.question}")
```

- [ ] **Step 3: 更新汇总打印**

把 `=== 汇总 ===` 之后的打印替换为:

```python
    agg = aggregate(rows)
    print("-" * 72)
    print(f"\n=== 汇总(正例 {agg['n_positive']} / 负例 {agg['n_negative']})===")
    if not args.agent:
        print(f"hit@{settings.top_k}:        {agg['hit_rate']:.1%}")
        print(f"recall@{settings.top_k}:     {agg['avg_recall']:.3f}")
        print(f"precision@{settings.top_k}:  {agg['avg_precision']:.3f}")
        print(f"nDCG@{settings.top_k}:       {agg['avg_ndcg']:.3f}")
        print(f"MRR:            {agg['avg_mrr']:.3f}")
    print(f"生成分({scorer_name}): {agg['avg_gen_score']:.3f}")
    print(f"拒答正确率:      {agg['refusal_accuracy']:.1%}")
```

- [ ] **Step 4: 冒烟验证(不连模型,仅确认导入/语法)**

Run: `cd /Users/mi/Documents/test/rag-docs && python -c "import ast; ast.parse(open('scripts/eval.py').read()); print('syntax ok')"`
Expected: `syntax ok`

> 完整联调需 Ollama+Qdrant 起来、且已迁移 dataset.json;此处只做静态校验,真实跑分在消融任务里做。

- [ ] **Step 5: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add scripts/eval.py
git commit -m "refactor: eval CLI 输出适配多来源指标与正负例分组"
```

---

### Task 6: 评估集生成工具 gen_eval(核心 + CLI)

**Files:**
- Create: `src/rag/gen_eval.py`
- Create: `scripts/gen_eval.py`
- Test: `tests/test_gen_eval.py`

**Interfaces:**
- Consumes: `LLM.generate`、`Chunk`、`loader.load_chunks_from_dir`
- Produces:
  - `build_gen_prompt(passage: str) -> str`
  - `parse_candidate(reply: str) -> dict | None`(键 `question`/`expected_keywords`/`expected_answer`)
  - `generate_candidate(chunk: Chunk, llm: LLM) -> dict | None`(附 `expected_sources=[chunk.source]`)
  - `sample_chunks(chunks, per_source, seed=0) -> list[Chunk]`
  - `generate_candidates(chunks, llm, per_source=2, seed=0) -> list[dict]`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_gen_eval.py`:

```python
from rag.gen_eval import (
    parse_candidate, generate_candidate, sample_chunks, generate_candidates,
)
from rag.models import Chunk


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt):
        return self.reply


def _chunk(source, i=0):
    return Chunk(id=f"{source}::{i}", text="正文", source=source,
                 library="lib", chunk_index=i)


def test_parse_candidate_extracts_json():
    reply = '前言 {"question": "Q?", "expected_keywords": ["k1"], "expected_answer": "A"} 结尾'
    got = parse_candidate(reply)
    assert got == {"question": "Q?", "expected_keywords": ["k1"],
                   "expected_answer": "A"}


def test_parse_candidate_bad_json_returns_none():
    assert parse_candidate("没有 json") is None


def test_generate_candidate_attaches_source():
    llm = _FakeLLM('{"question": "Q?", "expected_keywords": [], "expected_answer": "A"}')
    cand = generate_candidate(_chunk("a.md"), llm)
    assert cand["expected_sources"] == ["a.md"]
    assert cand["question"] == "Q?"


def test_sample_chunks_stratified_per_source():
    chunks = [_chunk("a.md", 0), _chunk("a.md", 1),
              _chunk("b.md", 0), _chunk("b.md", 1)]
    picked = sample_chunks(chunks, per_source=1, seed=0)
    assert {c.source for c in picked} == {"a.md", "b.md"}
    assert len(picked) == 2


def test_generate_candidates_skips_unparseable():
    chunks = [_chunk("a.md")]
    ok = generate_candidates(chunks, _FakeLLM('{"question":"Q?","expected_keywords":[],"expected_answer":"A"}'), per_source=1)
    bad = generate_candidates(chunks, _FakeLLM("垃圾"), per_source=1)
    assert len(ok) == 1 and bad == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_gen_eval.py -v`
Expected: FAIL(`rag.gen_eval` 模块不存在)

- [ ] **Step 3: 实现 src/rag/gen_eval.py**

创建 `src/rag/gen_eval.py`:

```python
"""从已有 chunk 反向生成评估候选题(M7)。领域无关:复用 loader 从 docs 出题。

只产候选写入 eval/candidates.json,绝不覆盖正式集;人工复核后再合并。
负例(拒答样本)不由本工具生成——LLM 不擅长造"库里没有"的题,手工加。
"""
import json
import random
import re

from rag.interfaces import LLM
from rag.models import Chunk

_GEN_PROMPT = """根据下面这段资料,出一道能被它回答的问题,并给出答案要点。
严格只输出 JSON,不要解释,格式:
{{"question": "问题", "expected_keywords": ["关键词1", "关键词2"], "expected_answer": "标准答案"}}

资料:
{passage}
JSON:"""


def build_gen_prompt(passage: str) -> str:
    return _GEN_PROMPT.format(passage=passage[:1000])


def parse_candidate(reply: str) -> dict | None:
    """从 LLM 回复里抽出 JSON 候选;抽不到或缺 question 则返回 None。"""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    q = obj.get("question")
    if not q or not isinstance(q, str):
        return None
    return {
        "question": q.strip(),
        "expected_keywords": obj.get("expected_keywords") or [],
        "expected_answer": obj.get("expected_answer") or "",
    }


def generate_candidate(chunk: Chunk, llm: LLM) -> dict | None:
    parsed = parse_candidate(llm.generate(build_gen_prompt(chunk.text)))
    if parsed is None:
        return None
    return {
        "question": parsed["question"],
        "expected_sources": [chunk.source],
        "expected_keywords": parsed["expected_keywords"],
        "expected_answer": parsed["expected_answer"],
    }


def sample_chunks(chunks: list[Chunk], per_source: int,
                  seed: int = 0) -> list[Chunk]:
    """按 source 分层抽样,保证覆盖不同文件;可复现(seed)。"""
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    rng = random.Random(seed)
    selected: list[Chunk] = []
    for source in sorted(by_source):
        group = list(by_source[source])
        rng.shuffle(group)
        selected.extend(group[:per_source])
    return selected


def generate_candidates(chunks: list[Chunk], llm: LLM,
                        per_source: int = 2, seed: int = 0) -> list[dict]:
    out: list[dict] = []
    for c in sample_chunks(chunks, per_source, seed):
        cand = generate_candidate(c, llm)
        if cand:
            out.append(cand)
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_gen_eval.py -v`
Expected: PASS(5 条)

- [ ] **Step 5: 写 CLI(装配,不单测)**

创建 `scripts/gen_eval.py`:

```python
"""从 docs 生成评估候选题,写入 eval/candidates.json(供人工复核)。

用法:
    python scripts/gen_eval.py --docs docs --per-source 2 --out eval/candidates.json

依赖:Ollama 已启动(仅需 LLM,不需 Qdrant)。
"""
import argparse
import json
from pathlib import Path

from rag.config import get_settings
from rag.providers.ollama_llm import OllamaLLM
from rag.loader import load_chunks_from_dir
from rag.gen_eval import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="生成评估候选题")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval/candidates.json")
    args = parser.parse_args()

    settings = get_settings()
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.eval_temperature)
    chunks = load_chunks_from_dir(Path(args.docs),
                                  chunk_size=settings.chunk_size,
                                  overlap=settings.chunk_overlap,
                                  strategy=settings.chunk_strategy)
    candidates = generate_candidates(chunks, llm,
                                     per_source=args.per_source, seed=args.seed)
    Path(args.out).write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"生成候选 {len(candidates)} 条 → {args.out}(请人工复核后并入 dataset.json)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 静态校验 CLI**

Run: `cd /Users/mi/Documents/test/rag-docs && python -c "import ast; ast.parse(open('scripts/gen_eval.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 7: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/gen_eval.py scripts/gen_eval.py tests/test_gen_eval.py
git commit -m "feat: 评估集候选生成工具 gen_eval(领域无关,产候选供人工复核)"
```

---

### Task 7: 一键消融对比表 ablate(核心 + CLI)

**Files:**
- Create: `src/rag/ablate.py`
- Create: `scripts/ablate.py`
- Test: `tests/test_ablate.py`

**Interfaces:**
- Consumes: Task 4 的 `aggregate` 返回结构
- Produces: `format_comparison_table(results: list[dict]) -> str`,`results` 每项形如 `{"config": str, "agg": dict}`,返回 markdown 表。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_ablate.py`:

```python
from rag.ablate import format_comparison_table


def test_format_comparison_table_markdown():
    results = [
        {"config": "rewrite=off rerank=off",
         "agg": {"hit_rate": 0.4, "avg_recall": 0.3, "avg_precision": 0.1,
                 "avg_ndcg": 0.35, "avg_mrr": 0.3, "avg_gen_score": 0.6,
                 "refusal_accuracy": 1.0}},
        {"config": "rewrite=on rerank=on",
         "agg": {"hit_rate": 0.9, "avg_recall": 0.8, "avg_precision": 0.4,
                 "avg_ndcg": 0.85, "avg_mrr": 0.75, "avg_gen_score": 0.82,
                 "refusal_accuracy": 1.0}},
    ]
    table = format_comparison_table(results)
    lines = table.splitlines()
    assert lines[0].startswith("| 配置 |")
    assert set(lines[1]) <= set("|-")          # 分隔行只有 | 和 -
    assert "rewrite=off rerank=off" in table
    assert "0.900" in table                     # hit_rate=0.9 → 0.900
    assert len(lines) == 4                       # 表头 + 分隔 + 2 行数据
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_ablate.py -v`
Expected: FAIL(`rag.ablate` 不存在)

- [ ] **Step 3: 实现 src/rag/ablate.py**

创建 `src/rag/ablate.py`:

```python
"""消融对比表格式化(M7)。把多组配置的 aggregate 结果排成一张 markdown 表。"""

# (aggregate 键, 表头显示名),顺序即列顺序
_COLUMNS = [
    ("hit_rate", "hit@k"),
    ("avg_recall", "recall@k"),
    ("avg_precision", "precision@k"),
    ("avg_ndcg", "nDCG@k"),
    ("avg_mrr", "MRR"),
    ("avg_gen_score", "gen"),
    ("refusal_accuracy", "拒答正确率"),
]


def format_comparison_table(results: list[dict]) -> str:
    """results: [{"config": str, "agg": {metric: value}}, ...] → markdown 表。"""
    header = "| 配置 | " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 1)
    lines = [header, sep]
    for r in results:
        agg = r["agg"]
        cells = " | ".join(f"{agg.get(key, 0.0):.3f}" for key, _ in _COLUMNS)
        lines.append(f"| {r['config']} | {cells} |")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/mi/Documents/test/rag-docs && python -m pytest tests/test_ablate.py -v`
Expected: PASS

- [ ] **Step 5: 写 CLI(扫 rewrite×rerank,不单测)**

创建 `scripts/ablate.py`:

```python
"""一键消融:扫 query_rewrite × rerank 四组,跑评估集,输出 markdown 对比表。

用法:
    python scripts/ablate.py [--dataset eval/dataset.json] [--scorer keyword]

依赖:Ollama 与 Qdrant 已启动,且已灌库、已迁移 dataset.json。
诚实边界:chunk_strategy 是灌库期参数(切换需重灌),不在此进程内扫描。
"""
import argparse
import json
from pathlib import Path

from rag.config import get_settings
from rag.providers.ollama_embedder import OllamaEmbedder
from rag.providers.ollama_llm import OllamaLLM
from rag.providers.qdrant_store import QdrantStore
from rag.retrieve import retrieve
from rag.generate import answer as generate_answer
from rag.scoring import get_scorer
from rag.evaluate import evaluate_sample, aggregate
from rag.query_rewrite import LlmQueryRewriter
from rag.rerank import LlmReranker
from rag.ablate import format_comparison_table
from rag.models import EvalSample


def main() -> None:
    parser = argparse.ArgumentParser(description="消融扫描 rewrite×rerank")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--scorer", default=None)
    args = parser.parse_args()

    settings = get_settings()
    scorer_name = args.scorer or settings.eval_scorer
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model,
                    temperature=settings.eval_temperature)
    store = QdrantStore(collection_name=settings.collection_name,
                        url=settings.qdrant_url)
    scorer = get_scorer(scorer_name, llm=llm, embedder=embedder)
    samples = [EvalSample(**d) for d in
               json.loads(Path(args.dataset).read_text(encoding="utf-8"))]

    results = []
    for use_rewrite in (False, True):
        for use_rerank in (False, True):
            rewriter = LlmQueryRewriter(llm) if use_rewrite else None
            reranker = LlmReranker(llm) if use_rerank else None
            rows = []
            for s in samples:
                retrieved = retrieve(s.question, embedder, store,
                                     top_k=settings.top_k, library=None,
                                     rewriter=rewriter, reranker=reranker,
                                     rerank_factor=settings.rerank_factor)
                ans = generate_answer(s.question, retrieved, llm)
                rows.append(evaluate_sample(s, retrieved, ans, scorer))
            results.append({
                "config": f"rewrite={'on' if use_rewrite else 'off'} "
                          f"rerank={'on' if use_rerank else 'off'}",
                "agg": aggregate(rows),
            })

    print(f"\n评分器={scorer_name} top_k={settings.top_k} "
          f"样本={len(samples)}(chunk 策略需手动重灌后单独跑)\n")
    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 静态校验 CLI**

Run: `cd /Users/mi/Documents/test/rag-docs && python -c "import ast; ast.parse(open('scripts/ablate.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 7: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add src/rag/ablate.py scripts/ablate.py tests/test_ablate.py
git commit -m "feat: 一键消融 harness(扫 rewrite×rerank 出对比表)"
```

---

### Task 8: 更新 notes/06 记录新指标与 nDCG 诚实说明

**Files:**
- Modify: `notes/06-评估.md`

**Interfaces:**
- Consumes: 无
- Produces: 文档,无代码接口。

- [ ] **Step 1: 追加指标说明段**

在 `notes/06-评估.md` 末尾追加:

```markdown
## M7:多来源指标与消融 harness

评估集升为多来源(`expected_sources`),正例算排序指标、负例只算拒答正确性,两组分开聚合。

- **hit@k**:top-k 出现任一相关来源即命中。
- **recall@k**:命中的不同相关来源数 / 相关来源总数。
- **precision@k**:top-k 中相关块数 / 返回块数(k)。
- **MRR**:第一个相关块的倒数排名。
- **nDCG@k**(二元相关性):`DCG=Σ rel_i/log2(i+1)`,除以 IDCG。
  - **诚实说明**:相关性标注是文件级、排序是 chunk 级,存在粒度错配。IDCG 用「top-k 内实际相关数」作理想上界(语料中相关块总数未知),所以 nDCG 度量的是「已召回的相关块是否尽量靠前」,非绝对理想。
- **拒答正确率**:负例应拒答 + 正例不应误拒,跨全体聚合。

**评估集生成**:`scripts/gen_eval.py` 从 docs 出候选题(领域无关),只产 `eval/candidates.json`,人工复核后并入 `dataset.json`;负例手工加。
**一键消融**:`scripts/ablate.py` 扫 `query_rewrite × rerank` 四组出对比表;`chunk_strategy` 为灌库期参数需手动重灌后单独跑。
```

- [ ] **Step 2: 提交**

```bash
cd /Users/mi/Documents/test/rag-docs
git add notes/06-评估.md
git commit -m "docs: notes/06 记录 M7 多来源指标与 nDCG 粒度说明"
```

---

## Self-Review

**1. Spec coverage(逐节对照 spec):**
- §1 数据模型多来源 + 旧字段兼容 → Task 1 ✅;dataset.json 迁移 → Task 2 ✅
- §2 检索指标(hit/mrr/recall/precision/ndcg + 负例分组 + aggregate 拆分)→ Task 3 + Task 4 ✅
- §3 gen_eval(loader 出题、只产候选、负例手工)→ Task 6 ✅
- §4 ablate(扫 rewrite×rerank、markdown 表、chunk 策略手动边界)→ Task 7 ✅
- §5 测试(test_evaluate 多来源+负例+兼容加载、test_gen_eval、test_ablate)→ Task 1/3/4/6/7 ✅
- §影响文件表:models/evaluate/dataset.json/eval.py/gen_eval.py/ablate.py/test_*/notes06 → 全覆盖(gen_eval/ablate 核心落在 `src/rag/` 以便单测,较 spec 文件表更细,CLI 仍在 `scripts/`)✅
- §6 非目标(不做 groundedness / 多领域集 / 回归门禁 / 灌库期扫描)→ 计划未越界 ✅

**2. Placeholder scan:** 无 TBD/TODO;所有 step 含实际代码或可执行命令。✅

**3. Type consistency:**
- `expected_sources: list[str]` 在 Task 1 定义,Task 3/4/6 一致使用。✅
- 指标签名 `(expected_sources, retrieved)` 在 Task 3 定义,Task 4 `evaluate_sample` 按此调用。✅
- `aggregate` 键(`hit_rate`/`avg_recall`/`avg_precision`/`avg_ndcg`/`avg_mrr`/`avg_gen_score`/`refusal_accuracy`)在 Task 4 定义,Task 5(eval.py 打印)与 Task 7(ablate 列)一致引用。✅
- `results` 结构 `{"config", "agg"}` 在 Task 7 core 与 CLI 一致。✅
