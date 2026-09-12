# M8 设计:通用化(领域无关引擎 + 命名 profile)

## 目标

把散落在代码里的「技术文档专属」耦合点抽成**命名 profile**,让同一套引擎换个 profile + 换知识库
就能服务不同领域。耦合点:人设/拒答话术(写死在 generate.py、agent.py)、查询改写 prompt
(写死"文档是英文的")、chunking 噪声词(写死 FastAPI 专属)、embedding 前缀(压根没加)。

顺带修一个高性价比点:nomic-embed-text 官方要求的 `search_query:` / `search_document:` 前缀
现在两个都没加。M8 把前缀纳入 profile,tech profile 开启并**重灌库**,用 M7 评估地基量效果——
这是评估地基的第一个真实用例。

**核心不变量**:默认 `tech` profile 完全复刻现有行为(唯一新变量是 embedding 前缀);
检索算法本身不动(多路召回是 M9);不做多格式加载(M8-A)。

## 现状(耦合点)

- `generate.py` `_SYSTEM`:写死"你是一个严谨的技术文档问答助手…无法回答"。
- `agent.py` `_SYSTEM`:另一份写死的技术文档助手人设。
- `query_rewrite.py` `_REWRITE_PROMPT`:写死"文档是英文的,只输出英文关键词"。
- `chunking.py` `_NOISE_MARKERS`:写死 `sponsor / fastapi cloud / conf / deploying to / deployment successful`。
- `ollama_embedder.py`:无任何前缀;ingest 与 retrieve 直接 embed 原文。
- `evaluate.py` `_REFUSAL_MARKER = "无法回答"`:写死。

## 1. DomainProfile + profiles/<name>.toml

新增 `DomainProfile`(pydantic 模型)与 `load_profile(name) -> DomainProfile`,用 stdlib `tomllib`
读 `profiles/<name>.toml`(**零新依赖**,Python 3.11 自带)。`Settings` 增加 `profile: str = "tech"`。

| 字段 | 类型 | 作用 | tech 默认 |
|---|---|---|---|
| `persona` | str | 注入 generate/agent 的人设身份句 | "你是一个严谨的技术文档问答助手" |
| `refusal_text` | str | 指示 LLM 拒答时输出的话 | "根据现有资料无法回答" |
| `refusal_marker` | str | 检测拒答的子串(给 evaluate.is_refusal) | "无法回答" |
| `rewrite_prompt` | str | 查询改写 prompt;**空串=不改写** | 现有英文扩展 prompt(原样搬入) |
| `noise_markers` | list[str] | chunking 噪声过滤词 | 现有 5 个 marker |
| `embed_query_prefix` | str | 查询向量化前缀 | `"search_query: "` |
| `embed_doc_prefix` | str | 文档向量化前缀 | `"search_document: "` |

缺字段用默认值(模型里给默认);缺文件抛清晰错误。

## 2. 各模块消费方式(沿用依赖注入)

组件只依赖朴素参数(字符串/列表),不认识 profile;**只有装配层加载 profile 并注入**。

- **generate.py**:`build_prompt(question, chunks, persona, refusal_text)` 用注入值拼系统段;
  `answer(...)` 增加 `persona`/`refusal_text` 参数。
- **agent.py**:`RagAgent(..., persona, refusal_text)`,据此构造循环 system 段(保留其工具循环结构)。
- **query_rewrite.py**:`LlmQueryRewriter(llm, prompt)` 接收 prompt 字符串。`QUERY_REWRITE` 开关
  仍作总闸:开关关或 prompt 为空 → 不装 rewriter。
- **chunking.py**:`is_noise(heading, body, noise_markers)`、`chunk_markdown(text, ..., noise_markers)`
  接收 markers;`loader.load_chunks_from_dir(..., noise_markers)` 透传;HTML 样板检测保留(格式通用)。
- **embedding 前缀**:不进 embedder(会误伤同用 `embed_one` 的 SemanticScorer),在两个真实调用点拼:
  - `ingest.py`:`embedder.embed([doc_prefix + c.text for c in chunks])`
  - `retrieve.py`:query 改写后拼 `query_prefix + query` 再 `embed_one`(retrieve 增加 `query_prefix: str = ""` 参数)
- **evaluate.py**:`is_refusal(answer_text, marker)` 接收 marker(默认 "无法回答",行为不变)。
  `evaluate_sample` 透传 marker;装配层从 profile 注入。

## 3. 装配点

`serve.py` / `eval.py` / `ingest.py` / `ablate.py` / `gen_eval.py` / `scripts/ask.py` 开头
`profile = load_profile(settings.profile)`,把各字段注入对应组件。装配是唯一知道 profile 的层。

## 4. 交付 + 度量(退出条件)

1. `profiles/tech.toml` 落地,全部测试绿——行为与现状一致(除 embedding 前缀外)。
2. **带 doc 前缀重灌库**,eval 对比:
   - 前缀关(现有 baseline,已实测:hit@4 94.4% / recall@4 0.944 / precision@4 0.597 / nDCG@4 0.795 / MRR 0.759 / 拒答正确率 86.4%)
   - 前缀开(重灌后)
   结论记入 `notes/07`。**有效保留;无效回滚 = profile 前缀清空 + 重灌,无需改码。**
3. 附极简 `profiles/generic.toml`(中性人设、空 rewrite_prompt、空 noise_markers、空/通用前缀),
   证明"换 profile 即换领域";不附第二套语料。

## 5. 测试(TDD)

- `test_profile`:tomllib 加载 tech.toml;缺字段回落默认;缺文件抛错;generic.toml 可加载。
- `test_generate` / `test_agent`:注入自定义 persona/refusal,断言系统段包含它们、拒答路径用 refusal_text。
- `test_query_rewriter`:注入自定义 prompt,FakeLLM 验证用的是注入 prompt;空 prompt 时装配层不装 rewriter(在装配处体现)。
- `test_markdown_chunking`:注入自定义 noise_markers,验证按注入词过滤(不再依赖写死的 FastAPI 词)。
- `test_ingest` / `test_retrieve`:断言喂给 embedder 的文本带 doc/query 前缀。
- `test_evaluate`:`is_refusal` 用注入 marker。

## 6. 影响文件一览

| 文件 | 改动 |
|---|---|
| `src/rag/profile.py` | 新增:DomainProfile + load_profile |
| `profiles/tech.toml` / `profiles/generic.toml` | 新增 |
| `src/rag/config.py` | 增加 `profile: str = "tech"` |
| `src/rag/generate.py` | persona/refusal_text 注入 |
| `src/rag/agent.py` | persona/refusal_text 注入 |
| `src/rag/query_rewrite.py` | prompt 注入 |
| `src/rag/chunking.py` | noise_markers 注入 |
| `src/rag/loader.py` | 透传 noise_markers |
| `src/rag/ingest.py` | doc 前缀 |
| `src/rag/retrieve.py` | query_prefix 参数 |
| `src/rag/evaluate.py` | is_refusal marker 注入 |
| `src/rag/pipeline.py` | 透传 persona/refusal_text/query_prefix |
| `scripts/*.py` | 装配层 load_profile 并注入 |
| `tests/*` | 对应更新 |
| `notes/07` | 记录前缀重灌前后对比 |

## 7. 非目标(YAGNI)

- 不做多格式加载器(PDF/HTML/docx)——M8-A。
- 不做第二套语料/多领域评估集。
- 不做 profile 热重载、继承/覆盖、动态注册。
- 不动检索算法(多路召回/专用重排是 M9/M10)。
- 前缀只做"整段前缀拼接";不做按模型自动探测前缀(换模型手改 profile)。

## 8. 诚实提醒

- **开前缀必须重灌库**:库里旧向量是无前缀算的;只给 query 加前缀会更糟。二者绑定。
- 前缀对 nomic 是官方推荐,但**是否真提升要以 eval 数字为准**,不是想当然;无效就回滚。
- 迁移面较广(动了 generate/agent/retrieve/pipeline 等多处签名),靠"默认 tech profile 复刻旧行为 + 全量测试绿"兜底防回退。
- profile 内容是纯文本 prompt/词表,人工维护;不做校验以外的智能。
