# M4 设计:评估(三种评分器可插拔)

## 目标

回答 notes/00 的终极问题"我怎么知道 RAG 到底好不好",把 M2 暴露的
"检索不准"从主观感受变成**可量化、可复现**的分数。检索层 + 生成层都评,
生成层评分方法通过环境变量在三种之间切换。

## 组件

### 1. 评估集 `eval/dataset.json`
手写 JSON 数组,约 8–12 条,覆盖 5 篇 fastapi 文档 + 至少 1 条"库里没有答案"
的问题(测拒答)。每条:

```json
{
  "question": "FastAPI 里路径参数怎么声明?",
  "expected_source": "fastapi/tutorial-path-params.md",
  "expected_keywords": ["路径参数", "path"],
  "expected_answer": "路径参数在路径中用花括号 {} 声明,并在函数参数里接收。"
}
```

`expected_source` 无答案的样本设为 null(或空),`expected_answer` 只有 B/C 用。

### 2. 可插拔评分器 `src/rag/scoring.py` + `AnswerScorer` 接口(interfaces.py)

三个实现,都实现 `score(answer_text, sample) -> float`,返回 0–1:

| 方法 | 类 | 依赖 | 分数含义 | 局限 |
|---|---|---|---|---|
| A `keyword` | KeywordScorer | 无 | 期望关键词命中比例 | 换个说法会误判 |
| B `llm_judge` | LlmJudgeScorer | 复用 LLM | LLM 打 0/0.5/1 | 有噪声、慢 |
| C `semantic` | SemanticScorer | 复用 Embedder | 答案与标准答案余弦相似度 | 语义像≠事实对、要阈值 |

工厂 `get_scorer(name, llm=None, embedder=None)` 按名字返回实现。
依赖倒置一致性:B 复用现有 LLM,C 复用现有 Embedder,不引新技术。

### 3. 配置 `config.py`
加字段 `eval_scorer: str = "keyword"`(keyword | llm_judge | semantic)。
环境变量 `EVAL_SCORER=semantic` 切换,`.env.example` 同步。

### 4. 评估逻辑 `src/rag/evaluate.py`(纯函数,可单测)

检索层(与评分器无关,始终算):
- `hit@k`:期望来源是否在 Top-K
- `MRR`:期望来源的倒数排名(越靠前越高)

生成层:
- 调选定 scorer 得每条答案分
- `refusal`:答案是否为拒答(检测"无法回答")

聚合报告:平均 hit@k、平均 MRR、平均生成分、拒答数。

### 5. CLI `scripts/eval.py`
装配真实 provider + 选定 scorer,跑完整评估集,打印表格报告
(每条问题 hit/rank/gen-score + 底部汇总)。

## 测试(TDD)

- `test_scoring.py`:A 纯文本断言命中率;B 用 FakeLLM(返回 "1"/"0")验证解析;
  C 用 FakeEmbedder(已知向量)验证余弦。
- `test_evaluate.py`:fake 检索结果验证 hit@k / MRR / 聚合。
- CLI 靠真实运行验证。

## 局限(诚实标注)

三种分数不同等可信:A 确定但粗糙,B 有随机噪声,C 依赖阈值且"语义像≠事实对"。
报告与 notes/06 会如实标注,不假装三者等价。

## 非目标(YAGNI)

- 不做自动阈值调优(留给人看数字判断)
- 不做大规模评估集(8–12 条够暴露问题)
- 不改 M1–M3 的检索/生成逻辑(M4 只度量,不优化;优化是看到数字之后的事)
