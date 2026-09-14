# M12 设计:多轮会话 / 上下文管理

## 目标

把 Docky 从**单轮无状态**问答升级为**多轮会话**:能接住"它能限定类型吗?""那价格呢?"这类
依赖上文的追问。这是从"文档问答引擎"迈向"助手"的关键一步,与前面的检索/生成优化正交。

**路线(已定)**:**pipeline + condense**——不走 agent 塞历史(本地 7B 工具调用不稳,M6 已验)。
condense 是窄而稳的任务,且**完整复用现有检索栈(hybrid/rerank/改写/前缀)一行不改**,可控可测。

**核心不变量**:单轮 `/ask`、`/agent/ask` 原样保留;检索算法不动;`generate` 加可选 history 参数
默认空=现状(回归安全)。

## 1. SessionStore(可插拔,内存实现)

```python
class SessionStore(ABC):
    def get_history(self, session_id: str) -> list[Message]: ...
    def append(self, session_id: str, message: Message) -> None: ...

class InMemorySessionStore(SessionStore):   # dict[session_id, list[Message]]
```
复用现有 `models.Message`(role/content)。**Redis 扩展点**:将来加 `RedisSessionStore` 实现同接口即可
(同 M10 build_reranker 的可插拔套路),v1 不做。

## 2. condense(追问改写成独立问题)

`src/rag/condense.py`:`condense_question(llm, question, history) -> str`。
- 有历史时,LLM 据最近若干轮把 follow-up 改写成"脱离上文也能独立检索"的问题;
- **首轮/无历史 → 直接返回原问题**(不调用 LLM);
- 兜底:LLM 返回空/异常 → 退回原问题(不比单轮差)。
prompt 示例:给出对话历史 + 追问,要求"只输出改写后的完整独立问题,不要解释"。

## 3. ConversationalRag(新层,编排)

`src/rag/conversation.py`:
```python
class ConversationalRag:
    def __init__(self, retriever, llm, store, persona, refusal_text,
                 history_turns=6, condense=True): ...
    def chat(self, session_id, question, library=None) -> Answer:
        history = store.get_history(session_id)
        standalone = condense_question(llm, question, history) if (condense and history) else question
        chunks = retriever(standalone, library)          # 复用现有 retrieve(含 hybrid/rerank/改写/前缀)
        ans = generate_answer(question, chunks, llm, persona, refusal_text,
                              history=history[-history_turns:])
        store.append(session_id, Message(role="user", content=question))
        store.append(session_id, Message(role="assistant", content=ans.text))
        return ans
```
注:检索用 **standalone**(独立问题),生成用**原 question + 近 N 轮历史**(答得连贯)。
`retriever` 是与 eval/serve 装配一致的 retrieve 偏函数(带 hybrid/rerank/改写/前缀)。

## 4. generate.py:加可选 history

`build_prompt(..., history: list[Message] = [])` / `answer(..., history=[])`:
history 非空时,在【问题】前插入"【对话历史】\n<近 N 轮 user/assistant>"。默认空=逐字节现状。

## 5. API:`/chat` 端点

`POST /chat {session_id, question, library?}` → `Answer`。服务端用注入的 SessionStore 维护历史。
`create_app` 增加可选 `conversation` 依赖(同 agent 的注入方式);未配置则 503。`/ask`、`/agent/ask` 不变。

## 6. config
`history_turns: int = 6`(生成带几轮)、`condense: bool = True`。

## 7. 装配
`scripts/serve.py`:构造 InMemorySessionStore + retriever 偏函数 + ConversationalRag,注入 create_app。
`scripts/ask.py` 可加一个多轮交互模式(可选)。

## 8. 测试(TDD)
- `test_session_store`:append/get_history 顺序、隔离(不同 session 互不串)。
- `test_condense`:FakeLLM——有历史→用改写结果;无历史→原样返回(不调 LLM);空回复→兜底原问题。
- `test_conversation`:Fake retriever + FakeLLM——多轮:第 2 轮 chat 时 condense 收到了第 1 轮历史、
  检索用的是 standalone、历史被 append;首轮不 condense。
- `test_generate`:history 非空时 prompt 含历史;默认空时与现状一致(回归)。
- `test_api`:`/chat` 端点两轮调用,第二轮能引用第一轮(用 fake conversation 验证 session_id 透传)。

## 9. 验证(诚实)
多轮**无法用 56 题单轮评估集量化**。v1 = **脚本化多轮演示**:起服务/直接调 ConversationalRag,
第 1 轮"FastAPI 路径参数怎么声明?"→ 第 2 轮"它能限定类型吗?",确认 condense 把"它"补成
"路径参数",检索命中 tutorial-path-params、回答正确。这是能力烟囱验证,不是指标。多轮评估集留后续。

## 10. 影响文件
| 文件 | 改动 |
|---|---|
| `src/rag/interfaces.py` | SessionStore 接口 |
| `src/rag/session_store.py` | InMemorySessionStore |
| `src/rag/condense.py` | condense_question |
| `src/rag/conversation.py` | ConversationalRag |
| `src/rag/generate.py` | history 参数 |
| `src/rag/config.py` | history_turns / condense |
| `src/rag/api.py` | /chat 端点 |
| `scripts/serve.py`(/ ask.py 可选) | 装配 |
| `tests/*` | 上述 |

## 11. 非目标(YAGNI)
- 不做上下文压缩(compaction)/长期跨会话记忆/Redis 持久化。
- 不做 agent 多轮(依赖弱模型工具调用,风险大)。
- 不动检索算法/重排/embedding。
- 不做多轮评估集(v1 用演示验证)。

## 12. 诚实提醒
- condense 每轮多一次 LLM 调用(首轮免);弱模型偶尔改写跑偏 → 兜底退回原问题,不崩、不比单轮差。
- 生成带历史会让 prompt 变长;`history_turns` 控制上限,注意别稀释检索资料。
- 内存 store 重启即失、单进程;v1 够用,生产要 Redis(扩展点已留)。
- 多轮质量目前只有演示、无指标——如实标注,别当"已量化验证"。
