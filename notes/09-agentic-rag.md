# 09 · Agentic RAG(M6)

把"检索一次 → 生成一次"的直线,升级成 **LLM 自主决策的循环**:模型自己决定
要不要检索、检索够不够、要不要换 query 再查一轮,最后据资料作答。这是从 RAG 到
agent 的第一步。

## 单轮 vs Agentic

```
单轮(RagPipeline.ask):  问题 → 检索一次 → 生成一次 → 答案
Agentic(RagAgent.ask):  问题 → [LLM 决策] ⇄ search_docs 工具(可多轮) → 够了才作答
```

关键区别:"检索几次、用什么 query"从**程序写死**变成 **LLM 每步自己决定**。

## 架构:新增而不改造

- 新增 `ChatLLM` 接口(多轮消息 + 工具调用),**不动** M2-M5 的 `LLM`/`RagPipeline`,
  新老并存、可对比。
- `RagAgent` 只依赖 `ChatLLM` 接口 + 一个 `retriever` 回调;检索完整复用 M5 的
  retrieve(含查询改写/重排),被包成 `search_docs` 工具。
- 对外仍是一个 FastAPI:旧 `/ask`(单轮)保留,新增 `/agent/ask`,共用底层检索。

## Provider:一套循环驱动多家模型

- `OpenAICompatChatLLM` 一个类覆盖 OpenAI/DeepSeek/Groq/vLLM/**Ollama**(都提供
  `/v1/chat/completions`)。换厂商只改 `.env`(base_url/model/key_env),零代码。
- `ClaudeChatLLM` 单独一个类(tool-use 协议不同)。
- `build_chat_llm(settings)` 是"选哪家"的唯一决策点。

## 坑:本地 7B 把工具调用写进正文

- **现象**:qwen2.5:7b 有时不走结构化 `tool_calls` 字段,而是把
  `{"name":"search_docs","arguments":{...}}` 当普通文本吐在回答里 → agent 以为是
  最终答案 → 没触发检索、出处为空。
- **间歇性**:同样的问题,有时结构化正确、有时泄漏到正文。
- **两手修复**:
  1. **强化 system prompt**:明确"用结构化工具调用,不要把调用写进正文"。实测
     加了之后 3/3 都正确走结构化。
  2. **兜底解析** `_recover_tool_calls`:即便还是泄漏到正文,用正则把
     `{"name":..,"arguments":..}` 捞出来当工具调用执行。防御性,不依赖模型总是听话。
- **教训**:本地小模型的 function calling 不如大模型稳,agent 要对"模型不守格式"
  有容错。prompt 引导 + 解析兜底,双保险。

## 三方对比(22 条,keyword 评分,temperature=0)

| 方案 | 生成分 | 拒答 | LLM 调用/问 |
|---|---|---|---|
| 单轮 RAG(基线) | 0.288 | 7/22 | 1× |
| Agentic(本地 qwen2.5) | **0.356** | 6/22 | 2-5× |
| Agentic(Claude) | 未测(环境无 ANTHROPIC_API_KEY) | — | — |

- agent 生成分 **+24%**(0.288→0.356),拒答少 1 条。**正向但温和**。
- 原因:agent 能"换 query 再查一轮",比单轮固定检索更易找到对的资料;但当单轮
  检索已不错(基线 hit@4 77%)时,重试的边际收益有限——与 M5 重排的规律一致。
- 代价:LLM 调用翻 2-5 倍(本地慢但不花钱;Claude 花钱)。

## 非目标(A 阶段)

只有 `search_docs` 一个工具。web 搜索、代码执行等留到 B 阶段往同一循环加,
循环骨架不用重写。不做工具结果缓存、对话记忆持久化、流式输出。

---
相关笔记:[[04-检索与生成]] · [[05-服务化]] · [[07-检索优化]] · [[08-踩坑记录]]
