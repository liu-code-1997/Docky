# M10 补齐设计:bge 专用 cross-encoder 重排

## 目标

填上 M10 当初留的扩展点:实现 `CrossEncoderReranker`(bge-reranker),挂进已有 `build_reranker` 工厂,
成为可选的第三种重排 provider。用户已确认接受重依赖(torch + sentence-transformers)。

**核心不变量**:`Reranker` 接口不变;`LlmReranker`/`ListwiseLlmReranker` 保留;rerank 总开关默认关;
`rerank_provider` 默认仍 `llm_listwise`(bge 需显式 `RERANK_PROVIDER=cross_encoder` 才用)——**不装/不用 bge
的用户,torch 不会被 import**(惰性加载),系统照常跑。

**诚实前提**:多查询已把检索推到 hit 98%/recall 0.974(近顶),bge 作为排序侧优化空间有限;做它主要为
"补齐能力 + 留一个高质量重排选项",提升与否以 eval 为准。

## 1. CrossEncoderReranker(rerank.py)

实现 `Reranker.rerank(question, candidates, top_k)`:
- 惰性加载:`__init__(model_name, scorer=None)` 只存 model_name,**不加载模型**;首次 rerank 时才
  `from sentence_transformers import CrossEncoder` 并加载(`scorer` 注入用于测试,免 torch)。
- rerank:`scores = model.predict([(question, c.chunk.text) for c in candidates])`;按分降序;取 top_k;
  分数写回 RetrievedChunk.score。空候选→[]。
- **torch/sentence_transformers 只在 rerank 首次调用时 import**——不用 bge 的路径完全不碰重依赖。

## 2. build_reranker 扩展

```python
def build_reranker(provider, llm, cross_encoder_model="BAAI/bge-reranker-v2-m3"):
    if provider == "llm_listwise": return ListwiseLlmReranker(llm)
    if provider == "llm_pointwise": return LlmReranker(llm)
    if provider == "cross_encoder": return CrossEncoderReranker(cross_encoder_model)
    raise ValueError(...)
```
(加 `cross_encoder_model` 参数,默认 v2-m3;llm 对 cross_encoder 用不到但签名统一。)

## 3. config
- `rerank_cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"`(多语种,适配中文问/英文档;~2.3GB,首次用时下载;可改小如 bge-reranker-base)。
- `rerank_provider` 已存在(加一个合法值 `cross_encoder`)。

## 4. 依赖
`pyproject.toml` dependencies 加 `sentence-transformers>=3`(拉入 torch)。**能力所需的重依赖,已确认接受。**
注:这是项目首个重依赖;惰性 import 保证只在选 cross_encoder 时才加载。

## 5. 装配
`scripts/{serve,eval,ablate}.py`:`build_reranker(settings.rerank_provider, llm, settings.rerank_cross_encoder_model)`。

## 6. 测试(TDD)
- `test_reranker`:`CrossEncoderReranker(model, scorer=fake)`——注入假 scorer(返回给定分数),验证按分重排、
  取 top_k、空候选→[];**不加载 torch/模型**(靠 scorer 注入)。`build_reranker("cross_encoder", ...)`
  返回 CrossEncoderReranker 且**不触发模型加载**(惰性)。
- 不做真模型的单测(重、需下载);真效果在度量步验。

## 7. 度量(退出条件)
装好 + 首次下载模型后,eval 对比(同库):rerank 关 / listwise / **cross_encoder**。看 nDCG/MRR/生成分。
结论记 notes/07。bge 胜过 listwise 则作为推荐重排 provider;否则 listwise 仍默认、bge 作为可选。

## 8. 影响文件
| 文件 | 改动 |
|---|---|
| `src/rag/rerank.py` | CrossEncoderReranker + build_reranker 加分支/参数 |
| `src/rag/config.py` | rerank_cross_encoder_model |
| `pyproject.toml` | sentence-transformers |
| `scripts/{serve,eval,ablate}.py` | build_reranker 传 model |
| `tests/test_reranker.py` | CrossEncoderReranker(注入 scorer)+ 工厂惰性 |
| `notes/07` | rerank 三 provider 对比 |

## 9. 非目标(YAGNI)
- 不做 GPU/MPS 调优(CPU 跑够用,eval 慢可接受)。
- 不做模型缓存管理/量化。
- 不改检索/生成/多查询。
- 不把 cross_encoder 设默认(需显式启用)。

## 10. 诚实提醒
- **首个重依赖**:torch ~GB,bge-v2-m3 模型 ~2.3GB 首次下载。惰性 import 保护未启用路径。
- CPU 上 cross-encoder 对 20 候选逐对打分,比 listwise 慢;eval 会更慢。
- 多查询后检索近顶,bge 提升可能有限——**以 eval 为准**,无益就作为可选项留着,不设默认。
- 单测靠注入假 scorer,不覆盖真实模型行为;真效果只有度量步的 eval 数字。
