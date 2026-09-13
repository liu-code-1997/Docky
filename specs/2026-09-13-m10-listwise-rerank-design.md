# M10 设计:Listwise LLM 重排(可插拔,为 cross-encoder 预留)

## 目标

现有 `LlmReranker` 是**逐条**打分(N 次 LLM 调用/查询,慢)。M10 加一个 **Listwise 重排**:
一次 LLM 调用对全部候选排序。零新依赖。并把重排做成**可插拔工厂**,将来接 bge cross-encoder
只需加一个类 + 工厂一分支 + 配置一个值,retrieve/pipeline 不动。

**度量靶子**:当前(rerank 关,含 M11 放宽拒答):hit@4 94.2% / recall 0.936 / nDCG 0.823 / MRR 0.796。
诊断指出重排 ROI 有限,但可能救 2–3 道**排序问题**(如"path operation"召回到了 path-params 而非 first-steps)。
**用户已知 ROI 有限,仍要做,并要求为 bge 预留扩展**——故重点是**架构可插拔**,提升与否以 eval 为准。

## 1. ListwiseLlmReranker(rerank.py 新增)

实现现有 `Reranker` 接口(`rerank(question, candidates, top_k)`)。一次 LLM 调用:

```
prompt:列出候选 [1]..[N](各截断 ~300 字),要 LLM 按相关性从高到低只输出编号排序(如 "3,1,4,2")。
解析:正则取整数 → 只保留 1..N → 去重保序 → 映射回候选。
兜底:未被提及的候选按原序补到末尾(保证不丢);解析失败 → 全部原序 → 取 top_k(等价不重排,安全)。
```

## 2. build_reranker 工厂(rerank.py)

```python
def build_reranker(provider: str, llm) -> Reranker:
    if provider == "llm_listwise": return ListwiseLlmReranker(llm)
    if provider == "llm_pointwise": return LlmReranker(llm)   # 旧实现保留
    raise ValueError(f"未知 rerank_provider: {provider}")
    # 将来:if provider == "cross_encoder": return CrossEncoderReranker(...)  # bge,需新依赖
```

`Settings.rerank_provider: str = "llm_listwise"`(默认新的;`rerank` 总开关不变,默认关)。

## 3. 装配

`scripts/{serve,eval,ablate}.py`:把 `reranker = LlmReranker(llm) if settings.rerank else None`
改为 `reranker = build_reranker(settings.rerank_provider, llm) if settings.rerank else None`。

## 4. 度量(退出条件)

eval 对比 `rerank 关 vs 开(listwise)`(同库,只切开关)。看 nDCG/MRR/hit 是否升、有没有救回排序题;
结论记入 notes/07。有效则可设 `RERANK=true` 推荐;无效则默认关(机制+扩展点保留)。

## 5. 测试(TDD)
- `test_reranker`:FakeLLM 返回 "2,1,3" → 重排为该序;解析失败(乱码)→ 原序 top_k;
  未提及候选补末尾;空候选返回 []。
- `build_reranker`:各 provider 返回对应类型;未知值报错。

## 6. 影响文件
| 文件 | 改动 |
|---|---|
| `src/rag/rerank.py` | 新增 ListwiseLlmReranker + build_reranker;LlmReranker 保留 |
| `src/rag/config.py` | `rerank_provider: str = "llm_listwise"` |
| `scripts/{serve,eval,ablate}.py` | 用 build_reranker |
| `tests/test_reranker.py` | listwise + factory |
| `notes/07` | rerank 关/开对比 |

## 7. 非目标(YAGNI)
- 不引 torch/sentence-transformers/bge(留扩展点,不实装)。
- 不引 rerank API。
- 不动检索/生成/hybrid。

## 8. 诚实提醒
- 诊断已示重排 ROI 有限(检索近顶),很可能又是"数字持平/微动"——**以 eval 为准**,无效就默认关,但扩展点是长期价值。
- listwise 把 ~20 候选(各截 300 字≈6k)塞进一次调用,qwen2.5:7b 上下文够;更大 top_k×factor 时注意长度。
- 弱模型可能不严格输出编号 → 解析兜底(失败=原序)保证不崩、不比不重排更差。
