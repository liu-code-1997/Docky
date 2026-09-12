# M7 设计:评估地基(让测量可信)

## 目标

后续所有检索/生成优化(多路召回、专用重排、通用化……)都要靠评估集来证明「到底有没有用」。
但现在评估集只有 22 条、每条**单个** `expected_source`,M5 spec 自己都承认「提升几个百分点
可能是噪声」。M7 先把**测量地基**打牢,之后每个优化才有客观标尺:

1. **扩充评估集**到 30-50 条(LLM 生成候选 + 人工复核),并把生成工具做成**领域无关**——
   M8 换新语料能重头用。
2. **升级检索指标**:支持一题多个相关来源,新增 recall@k / precision@k / nDCG@k,
   正例(排序指标)与负例(拒答正确性)分开聚合、互不污染。
3. **一键消融 harness**:一条命令扫描多组运行时开关组合,自动输出 markdown 对比表,
   延续 M4/M5 的消融文化。

**核心不变量**:检索/生成链路(retrieve / generate / pipeline / agent)**不动**,
M7 只碰评估侧(models 的 EvalSample、evaluate、scoring 调用方、新增两个脚本)。

## 现状(读码确认)

```
[Ingest] docs/*.md → 切分 → 向量化 → Qdrant
[Retrieve] 问题 →(改写)→ 向量 → dense 检索 →(LLM 重排)→ Top-K
[Eval] scripts/eval.py 装配 → evaluate.evaluate_sample → aggregate
```

- `EvalSample`:`expected_source: str | None`(单来源;负例为 `null`)、`expected_keywords`、`expected_answer`。
- `evaluate.py`:`hit_at_k` / `reciprocal_rank`(均基于单来源)、`is_refusal`、`aggregate`。全是纯函数。
- `scripts/eval.py`:已有 `--query-rewrite` / `--rerank` / `--agent` 开关,单次跑一组配置。
- `eval/dataset.json`:22 条 = 18 正例(FastAPI 相关)+ 4 负例(`expected_source: null`)。

## 1. 数据模型:单来源 → 多来源

`EvalSample.expected_source: str | None` **改为** `expected_sources: list[str] = []`。

- 正例:一个或多个相关来源文件;负例(库里查不到):空列表 `[]`。
- **迁移**现有 22 条:单来源包成单元素列表,`null` → `[]`。
- **兼容旧字段**:加载时若读到旧的 `expected_source`(str 或 null),自动转成 `expected_sources`
  列表。迁移完 `dataset.json` 后即可移除该兼容分支——避免一次性手改 JSON 出错。

`expected_keywords` / `expected_answer` 不变(仍供生成层评分用)。

## 2. 检索指标(evaluate.py)

相关性判定在**文件粒度**:一个检索到的 chunk「相关」当且仅当 `chunk.source ∈ expected_sources`。
设 top-k 检索结果为按序的 chunk 列表,R = `expected_sources`(|R| ≥ 1 为正例)。

| 指标 | 定义 |
|---|---|
| `hit@k`(保留,重定义) | top-k 中出现**任一**相关来源即 True |
| `mrr`(保留,重定义) | 第一个「source ∈ R」的 chunk 的倒数排名;无则 0 |
| `recall@k`(新) | top-k 命中的**不同相关来源数** / \|R\| |
| `precision@k`(新) | top-k 中「source ∈ R」的 **chunk 数** / k |
| `ndcg@k`(新) | 见下 |
| 拒答正确性(新) | 见「负例处理」 |

**nDCG@k(二元相关性)**:`rel_i = 1 if 第 i 个 chunk 的 source ∈ R else 0`;
`DCG@k = Σ_{i=1..k} rel_i / log2(i+1)`。理想排序取「相关的全排最前」:
`IDCG@k = Σ_{i=1..min(k, n_rel)} 1 / log2(i+1)`,其中 `n_rel` = top-k 内相关 chunk 数;
`nDCG@k = DCG/IDCG`(n_rel=0 时定义为 0)。

> **诚实提醒**:相关性标注是文件级、而排序是 chunk 级,存在粒度错配。IDCG 用「top-k 内实际
> 相关数」作理想上界(而非语料中相关 chunk 总数,后者未知),因此 nDCG 度量的是「已召回的相关块
> 是否尽量靠前」,不是绝对理想。这个简化写进 notes,读数字时心里有数。

**负例处理(expected_sources 为空)**:
- 对 recall/precision/nDCG/hit/MRR **无定义**,从这些指标的聚合中**排除**。
- 单独度量**拒答正确性**:
  - 负例应当拒答(答案含「无法回答」);
  - 正例应当**不**误拒。
- `aggregate()` 拆成两组返回:`positives`(排序指标 + 生成分)与 `negatives`/`refusal`
  (拒答正确率)。生成分 `gen_score` 仍对正例算(复用现有 scorer,不改)。

## 3. 评估集生成工具(新 `scripts/gen_eval.py`)

领域无关:**复用现有 `loader.load_chunks_from_dir` 从 docs 目录读 chunk** 出题(与 ingest 同一套加载),
所以 M8 换语料、M8-A 升级 loader 支持多格式后都可原样重用。**不碰 `VectorStore` 接口**——
M7 对检索链路零改动的不变量成立。

```
loader 加载 docs 目录 → chunk 列表(按 source 分层抽样,保证覆盖不同文件)
  → 对每个抽中的 chunk,LLM 生成「一个能被这段内容回答的问题 + expected_keywords + expected_answer」
  → 记 expected_sources = [chunk.source]
  → 汇总写入 eval/candidates.json
```

- **只产候选,绝不覆盖** `eval/dataset.json`。人工复核挑选/修订后,自己合并进正式集。
- **负例仍手工加**(LLM 不擅长造「库里根本没有」的问题):现有 4 条保留,按需再补。
- 抽样数量、每源出题数由命令行参数控制;目标把正式集扩到 30-50 条。
- LLM 复用现有 `ChatLLM`/`LLM` provider(装配同 `scripts/eval.py`),温度用 `eval_temperature`(0)保证可复现。

## 4. 一键消融 harness(新 `scripts/ablate.py`)

- 扫描**运行时开关**组合:`query_rewrite {on,off} × rerank {on,off}`(2×2=4 组),
  每组复用 `eval` 的装配与逐条评估逻辑,跑完整评估集。
- 输出**一张 markdown 对比表**:每行一组配置,列为各指标(hit@k / recall@k / precision@k /
  nDCG@k / MRR / gen_score / 拒答正确率),可直接贴进 `notes/07`。
- **诚实边界**:`chunk_strategy`(char/markdown)是**灌库期**参数,切换要重灌库,**不进程内扫描**。
  表中作为一列说明「需手动重灌后单独跑」,不假装能一键切。
- 复用现有 eval 装配,不重写检索/生成链路。

## 5. 测试(TDD,沿用 FakeLLM / 纯函数风格)

- `test_evaluate`(扩写):
  - 多来源 hit@k / recall@k / precision@k / nDCG@k / MRR 的手造用例(含「部分命中」)。
  - 负例:拒答正确、误拒的判定;负例从排序指标聚合中排除。
  - 边界:空检索、相关源一个都没召回、n_rel=0 的 nDCG。
  - 旧字段 `expected_source` 兼容加载 → `expected_sources`。
- `test_gen_eval`:FakeLLM 编排,验证候选的结构(question/expected_sources/keywords/answer)
  与来源标注正确;不真连模型。
- `test_ablate`:对比表格式化是纯函数,喂假 rows 验证表格行列与数值正确。

## 6. 非目标 / 诚实提醒(YAGNI)

- **不做**忠实度 / groundedness 的 LLM 裁判(单独一块,留后面)。
- **不做**多领域语料评估集——那是 M8 的事;M7 只强化现有技术集 + 把生成工具做成领域无关。
- **不做**基线回归告警 / CI 门禁。
- **不**自动扫描灌库期参数(chunk 策略)。
- 评估集仍偏小(30-50 条),跨库比较时注意样本量;负例只有个位数,拒答正确率波动大,如实标注。
- LLM 生成的候选题会有噪声(问题太泛、答案不准),**人工复核是必需环节**,不可直接入正式集。

## 影响文件一览

| 文件 | 改动 |
|---|---|
| `src/rag/models.py` | `EvalSample.expected_source` → `expected_sources: list[str]` |
| `src/rag/evaluate.py` | 多来源指标 + recall/precision/nDCG + 负例分组聚合 |
| `eval/dataset.json` | 迁移 22 条为多来源;人工复核后扩到 30-50 条 |
| `scripts/eval.py` | 单次评估输出改用新指标(装配逻辑不变) |
| `scripts/gen_eval.py` | 新增:从 chunk 生成评估候选 |
| `scripts/ablate.py` | 新增:扫描运行时开关组合出对比表 |
| `tests/test_evaluate.py` | 扩写多来源 + 负例 + 兼容加载 |
| `tests/test_gen_eval.py` | 新增 |
| `tests/test_ablate.py` | 新增 |
| `notes/06-评估.md` | 记录新指标定义与 nDCG 简化的诚实说明 |
