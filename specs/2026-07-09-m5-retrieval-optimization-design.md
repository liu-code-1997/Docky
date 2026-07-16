# M5 设计:检索优化(消融实验)

## 目标

M4 诊断出瓶颈在检索层(hit@4=40%、MRR=0.3)。M5 用三种独立可开关的改进提升检索质量,
并用消融实验(ablation)逐个评估,看清每个改动的孤立贡献——而不是一把梭后不知道谁起了作用。

已验证根因:问"路径参数",召回的 Top-4 全是"FastAPI Cloud 赞助""FastAPI Conf 会议"
"部署成功"这类营销噪声块,正确的 tutorial-path-params.md 一块都没进来。

## 三种改进及其作用环节

```
[Ingest]  文档 → 切分 → 向量化 → 入库
                  ↑① markdown 切分 + 噪声过滤(灌库时;改了要重新 ingest)

[Retrieve] 问题 → ②改写 → 向量化 → 检索候选 → ③重排 → Top-K
                  ↑② 查询改写(运行时)      ↑③ 重排(运行时)
```

| # | 改进 | 环节 | 开关(env) | 默认 | 依赖 |
|---|---|---|---|---|---|
| ① | markdown 切分 + 噪声过滤 | 灌库 | `CHUNK_STRATEGY=char\|markdown` | char | 无 |
| ② | 查询改写 | 检索前 | `QUERY_REWRITE=true\|false` | false | 复用 LLM |
| ③ | 重排 rerank | 检索后 | `RERANK=true\|false` | false | 复用 LLM |

默认全 baseline(char / false / false),行为与 M4 一致,保证每个开关效果可干净归因。
三个开关可任意组合(2×2×2)。切换①后必须重新灌库(库里存的是旧切法的向量)。

## 各改进实现要点

### ① markdown 切分 + 噪声过滤
- chunking.py 新增 `chunk_markdown(text, ...)`:按 markdown 标题层级切,每块保留标题路径作上下文。
- 噪声过滤:丢弃匹配噪声特征的块(标题/正文含 sponsor、FastAPI Cloud、Conf、Deploy 成功样板等)。
- loader 按 `chunk_strategy` 选择 `chunk_text`(旧)或 `chunk_markdown`(新)。

### ② 查询改写 QueryRewriter
- 新增接口 `QueryRewriter` + `LlmQueryRewriter`(复用 LLM):把中文问题改写/扩展,补上英文术语。
- retrieve 在 `query_rewrite=True` 时先改写再向量化。解决中文问 / 英文档跨语言。

### ③ 重排 Reranker
- 新增接口 `Reranker` + `LlmReranker`(复用 LLM):对候选逐个打"相关性分",重排取前 top_k。
- retrieve 在 `rerank=True` 时先召回 `top_k × rerank_factor`(默认 5,即 20)候选,再重排。

## 架构一致性

三者都可插拔:①是 chunking 的策略分支,②③是新接口 + LLM 实现。
retrieve() 按开关决定是否启用。默认全关 = baseline 不变。

## 消融评估流程

```
baseline            现在:hit@4=40%  MRR=0.30
+① markdown 切分     重新灌库 → eval → 记提升
+② 查询改写          eval → 记提升
+③ 重排              eval → 记提升
```
每步一张对比表(notes/07),清楚看到谁贡献大。逐个做,任一步可停。

## 测试(TDD)

- `test_markdown_chunking`:按标题切、噪声过滤。
- `test_query_rewriter`:FakeLLM 验证改写。
- `test_reranker`:FakeLLM 验证重排序。
- retrieve 加开关后的分支测试。

## 非目标 / 诚实提醒(YAGNI)

- ②③用 LLM 会变慢(③尤其:N 次调用/问题),本地无新依赖的代价,notes 标注。
- 只有 10 条评估样本,提升几个百分点可能是噪声;必要时先扩充评估集。
- 不引入外部 rerank/embedding 模型(如 bge-reranker),用现有 LLM 兜底即可;换专用模型是以后的事。
- 不做检索缓存、不做多集合管理。
