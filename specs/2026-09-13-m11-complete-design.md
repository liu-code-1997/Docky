# M11 补齐设计:多查询 + 上下文排序 + 行内引用

## 目标

补齐 M11(生成/查询理解)当初只做了"放宽拒答"的其余项。三项独立开关,默认全关(行为不变,回归安全),
各自/组合可用 56 题 eval 量化。都**零新依赖**。

- **多查询(multi-query)**:生成多个查询变体各自检索、RRF 融合——救诊断里"漏召回"的题(如"图数据库
  插入点"正确文档没进候选)。检索侧。
- **上下文排序(lost-in-the-middle)**:最相关块放首尾、次要埋中间,缓解 LLM"只看两头"。生成侧。
- **行内引用**:答案里用 [编号] 标注引用的资料,提可信度/可核查。生成侧。

**核心不变量**:三开关默认关 = 逐字节现状;检索/重排/embedding 算法不动;单轮链路不破。

## 1. 多查询(`src/rag/multi_query.py` + retrieve 集成)

```python
def expand_queries(llm, question, n=3) -> list[str]:
    # LLM 生成 n-1 个不同措辞/角度的检索查询,+ 原问题,去重;失败→[question]
def rrf_fuse(result_lists: list[list[RetrievedChunk]], top_k, k_rrf=60) -> list[RetrievedChunk]:
    # 客户端 RRF:按 chunk.id 累加 1/(k_rrf+rank),降序取 top_k;分数写回 RetrievedChunk.score
```

retrieve 集成(加**一个**可选参数 `query_expander: callable(question)->list[str] | None`):
- None(默认)→ 单查询,**与现状完全一致**;
- 非 None → `queries = query_expander(question)`;对每个 query 走现有(改写→前缀→embed→search/hybrid)得一个结果列表;
  多于 1 个 → `rrf_fuse`;然后(若有 reranker)在融合结果上重排。
- `query_expander` 在装配层用 llm 构造:`lambda q: expand_queries(llm, q, settings.multi_query_n)`。

配置:`multi_query: bool = False`、`multi_query_n: int = 3`。

## 2. 上下文排序(`src/rag/ordering.py`)

```python
def reorder_for_long_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    # 输入已按相关性降序;输出:最相关放首,其余交替填到尾部/中间,使高相关聚在两端。
    # 经典做法:排名 0,2,4… 放左半正序,1,3,5… 放右半——即最相关在首、次相关在尾。
```
应用点:生成前(pipeline.ask / eval / conversation 里,`reorder_context` 开时对最终 chunks 排序再喂 generate)。
配置:`reorder_context: bool = False`。

## 3. 行内引用(generate.py)

`build_prompt(..., inline_citations: bool = False)`:开时在规则里加一条"引用了哪段【资料】就在句末用其编号
[1]/[2] 标注",资料块已带 `[资料i]` 编号(现状)。默认关=现状。`answer(..., inline_citations=False)` 透传。
配置:`inline_citations: bool = False`。

## 4. 装配

config 三开关 + multi_query_n。`pipeline.ask` / `scripts/eval.py` / `scripts/serve.py` / `scripts/ablate.py`:
- 构造 `query_expander`(multi_query 开时)传给 retrieve;
- retrieve 返回后,`reorder_context` 开则 `reorder_for_long_context`;
- generate_answer 传 `inline_citations=settings.inline_citations`。
（ConversationalRag 若已合入亦同;本里程碑在 main 上做,M12 后续再叠。）

## 5. 测试(TDD)
- `test_multi_query`:expand_queries(FakeLLM 多行回复→去重含原问题;空/异常→[question]);rrf_fuse(两个列表,
  都靠前的 chunk 融合后排第一;分数单调)。
- `test_retrieve`:query_expander=None → 单查询路径不变(回归);给一个返回 2 变体的 expander → 调了 2 次底层检索并融合(捕获 fake)。
- `test_ordering`:reorder_for_long_context —— 最相关在首、次相关在尾;空/单元素安全。
- `test_generate`:inline_citations=True → prompt 含引用指令;默认 False → 不含(回归)。

## 6. 度量(退出条件)
eval 分别测:baseline、+multi_query、+reorder、+inline、(可选)全开。对照当前基线
(hit 94.2 / recall 0.936 / nDCG 0.823 / MRR 0.796 / gen 0.609 / refusal 85.7)。
重点:multi_query 是否救回漏召回题(recall/hit↑);reorder/inline 看 gen 分与 nDCG。
结论记 notes/07;有效项设推荐,无效默认关(机制保留)。

## 7. 影响文件
| 文件 | 改动 |
|---|---|
| `src/rag/multi_query.py` | 新增 expand_queries + rrf_fuse |
| `src/rag/ordering.py` | 新增 reorder_for_long_context |
| `src/rag/retrieve.py` | 加 query_expander 参数(默认 None=现状) |
| `src/rag/generate.py` | inline_citations 参数 |
| `src/rag/config.py` | multi_query/multi_query_n/reorder_context/inline_citations |
| `src/rag/pipeline.py` | 透传三开关 |
| `scripts/{eval,serve,ablate}.py` | 装配 |
| `tests/*` | 上述 |
| `notes/07` | 度量记录 |

## 8. 非目标(YAGNI)
- 不做 HyDE(本轮不选)。
- 不引任何新依赖。
- 不动重排/embedding/hybrid 算法本身。
- 不做多轮(M12)/bge(M10 补齐单独做)。

## 9. 诚实提醒
- 多查询每轮多 1 次 LLM 调用(生成变体)+ N 次检索,变慢;弱模型变体可能同质化 → 收益以 eval 为准。
- 上下文排序对短 top_k(=4)效果可能不明显(4 个块"中间"很小);如实看数字。
- 行内引用依赖 LLM 听话标注,7B 可能漏标/乱标;是体验增强,不强求准确率指标。
- 三项若 eval 无正向,就默认关(同 M8 前缀/M9 hybrid 的处理),机制留作扩展。
