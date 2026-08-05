# M6 设计:Agentic RAG(让 LLM 自主决策检索)

## 目标

把固定的"检索一次 → 生成一次"直线,升级成 **LLM 自主决策的循环**:LLM 自己判断
要不要检索、检索够不够、要不要换个 query 再查一轮,最后据资料生成带出处的答案。

这是从 RAG 到 agent 的第一步(A 阶段)。循环骨架建好后,B 阶段只需往里加新工具
(web 搜索、代码执行等),不必重写。

**核心不变量**:现有 `RagPipeline`(M2-M5)完全保留、不动一行,与 agent 并存;
`retrieve` 检索链路(query_rewrite / rerank / markdown 切分)原样复用,被包成 agent 的一个工具。

## 单项目、进程内调用(不拆服务)

RAG 与 agent 在**同一个项目、同一个进程**里:`search_docs` 工具**直接函数调用**
现有 `retrieve()`,不经过 HTTP。RAG 是 agent 的一项能力(工具),不是独立系统。

- 对外仍是**一个 FastAPI 应用**:现有单轮 `/ask`(RagPipeline)保留,agent 另加
  `/agent/ask`,两端点共用底层 `retrieve`。
- **不拆两个服务**——独立伸缩、多消费方复用、跨团队/技术栈边界,这些拆分动因当前一个都不占;
  拆分只会多两套部署/配置/序列化/跨服务调试,学不到 agent 核心。真需要拆是以后的事。

## 单轮 RAG vs Agentic RAG

```
单轮(现状 RagPipeline.ask):
  问题 → retrieve 一次 → generate 一次 → 答案

Agentic(新 RagAgent.ask):
  问题 → [LLM 决策] ──需要查?──→ search_docs 工具 → 看结果
                ↑                                        ↓
                └──── 不够好? 换 query 再查一轮 ─────────┘
                ↓
           够了 → 生成带出处的答案(不知道就拒答,原则不变)
```

## 关键技术前提:两种模型都原生支持 function calling

本地 `qwen2.5:7b`(Ollama `/api/chat` 支持 `tools`/`tool_calls`)与 Claude
(anthropic SDK 原生 tool use)**都支持原生工具调用**。因此**一套 agent 循环**
即可同时驱动本地和 Claude,靠配置切换——不需要维护脆弱的 ReAct 文本解析。

## 新增接口与实现

现有 `LLM.generate(prompt) -> str` 是纯文本补全,不支持多轮消息与工具调用。
**新增 `ChatLLM` 接口(不改造 `LLM`)**,让 M2-M5 的 `RagPipeline` 零影响、可对比。

```python
class ChatLLM(ABC):
    """支持多轮消息 + 工具调用的对话式 LLM(agent 用)。"""
    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> ChatResponse:
        """返回助手回合:要么最终文本,要么一组 tool_calls。"""
```

| 组件 | 位置 | 说明 |
|---|---|---|
| `ChatLLM` 接口 + `Message`/`ToolSpec`/`ChatResponse` 模型 | `interfaces.py` / `models.py` | 多轮 + 工具调用抽象 |
| `OpenAICompatChatLLM` | `providers/openai_chat.py` | 走 `/v1/chat/completions`,传 `tools`,解析 `tool_calls`。**一个类覆盖一大片**:OpenAI、DeepSeek、Groq、本地 vLLM,**以及 Ollama**(Ollama 也暴露 `/v1` 兼容端点)。靠 base_url + model + key 区分 |
| `ClaudeChatLLM` | `providers/claude_chat.py` | anthropic SDK 原生 tool use(协议不同,单独实现);model 从配置、key 从环境变量 |
| `RagAgent` | `agent.py` | 循环本体 + `search_docs` 工具 |

**为什么这样分**:市面上绝大多数 LLM 服务(含本地 Ollama/vLLM)都提供 OpenAI 兼容的
`/v1/chat/completions` 接口,所以**一个 `OpenAICompatChatLLM` 就能驱动它们全部**——换厂商只改
`.env`(base_url/model/key),不写新代码。Claude 的 tool use 协议与 OpenAI 不同,单独一个类。
`ChatLLM` 接口保证:未来若真要接某个协议独特的厂商,再写一个类即可,agent 循环不受影响。

## Agent 循环(agent.py)

```python
class RagAgent:
    def __init__(self, llm: ChatLLM, embedder, store, top_k,
                 rewriter=None, reranker=None, max_steps=5): ...
    def ask(self, question, library=None) -> Answer:
        # 循环: llm.chat(messages, tools=[search_docs])
        #   有 tool_calls → 执行 search_docs(内部调现有 retrieve) → 回填结果 → 再 chat
        #   无 tool_calls → 该文本即最终答案
        #   到 max_steps 仍未收敛 → 兜底停止(防死循环),用已有资料强制生成
```

- **工具**:`search_docs(query: str, library: str | None) -> list[chunk]`,
  内部就是 `retrieve(query, embedder, store, top_k, library, rewriter, reranker)`。
- **出处**:从每次 `search_docs` 结果累积去重收集,复用 `Answer.sources`。
- **拒答**:system 段沿用 generate.py 的强约束——只依据检索资料,无则答"根据现有资料无法回答"。

## 配置(config.py + .env.example)

Provider 全可配:换厂商/模型/地址只改 `.env`,不改代码。密钥永远从环境变量读,不入配置文件。

```python
# provider:openai_compat(默认,覆盖 OpenAI/DeepSeek/Groq/vLLM/Ollama)| claude
chat_provider: str = "openai_compat"

# openai_compat 用的三个旋钮(换厂商只改这几行)
chat_base_url: str = "http://localhost:11434/v1"   # 默认本地 Ollama 的 /v1 端点
chat_model: str = "qwen2.5:7b"
chat_api_key_env: str = "OPENAI_API_KEY"           # 从哪个环境变量读 key(本地 Ollama 可留空)

# claude 用(chat_provider=claude 时生效)
claude_model: str = "claude-opus-5"
# ANTHROPIC_API_KEY 从环境变量读

agent_max_steps: int = 5                            # 循环兜底上限
```

换厂商示例(全部只改 `.env`,零代码改动):

```bash
# 本地 Ollama(默认)
CHAT_BASE_URL=http://localhost:11434/v1
CHAT_MODEL=qwen2.5:7b

# DeepSeek
CHAT_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
CHAT_API_KEY_ENV=DEEPSEEK_API_KEY     # 再 export DEEPSEEK_API_KEY=...

# Groq / OpenAI / 本地 vLLM 同理,改 base_url + model + key_env 即可
```

**不做**(YAGNI):provider 动态注册表 / 插件加载。`ChatLLM` 接口已提供"写个类即扩展"的
能力,再加一层动态机制是过度设计;真遇到协议独特的厂商再写一个类即可。

## 验证:三方对比实验(延续 M4/M5 消融风格)

复用 `eval/dataset.json`,量化 agent 到底有没有用,而非"看起来很酷":

| 方案 | 准确率 | LLM 调用次数 |
|---|---|---|
| 单轮 RAG(现状 RagPipeline) | 基线 | 1× |
| Agentic RAG(本地 qwen2.5) | ? | 2-5× |
| Agentic RAG(Claude) | ? | 2-5× |

顺带量化"本地模型 vs Claude 在 agent 场景的差距",指导后续选型。结论记入 notes/。

## 测试(TDD)

- `test_agent`:用 FakeChatLLM(可编排返回 tool_calls 或最终文本)验证循环——
  单轮直接答、多轮检索、到 max_steps 兜底、出处累积、拒答路径。
- `test_openai_chat` / `test_claude_chat`:mock HTTP/SDK,验证 tools 传参与 tool_calls 解析
  (openai_compat 用同一测试覆盖 OpenAI/DeepSeek/Ollama 等,因为它们共用 `/v1` 协议)。
- 不真连模型(与现有 test_ollama_* 一致的 mock 风格)。

## 非目标 / 诚实提醒(YAGNI)

- A 阶段**只有 `search_docs` 一个工具**。web 搜索、代码执行等留到 B 阶段往同一循环加。
- 本地 7b 的多轮工具调用不稳(选错工具、参数乱填、不收敛),用 `max_steps` 兜底;
  开发期建议用 Claude 保证"逻辑对",再回测本地能扛到什么程度。
- agent 每问 LLM 调用翻 2-5 倍,本地慢但不花钱;Claude 花钱,notes 标注成本。
- 不改造现有 `LLM`/`RagPipeline`,不动 M2-M5 链路;新老并存,便于对比。
- 不做工具结果缓存、不做对话记忆持久化、不做流式输出——A 阶段不需要。
- 评估集仍只有 10 条,提升几个百分点可能是噪声;必要时先扩充。
