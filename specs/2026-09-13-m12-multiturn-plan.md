# M12 多轮会话 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** 多轮会话:SessionStore + condense(追问改写成独立问题)+ 生成带历史 + /chat 端点。pipeline+condense 路线,复用现有检索栈。

**Spec:** `specs/2026-09-13-m12-multiturn-design.md`

## Global Constraints
- `.venv/bin/python -m pytest`(venv=3.12)。零新依赖(复用 httpx/fastapi/pydantic)。
- 单轮 /ask、/agent/ask 不变;`generate` 的 history 默认空=逐字节现状(回归安全)。检索算法不动。
- conventional commits;可测逻辑在 src/rag,scripts 仅装配。预存 warning 忽略。

---

### Task 1: SessionStore + config

**Files:** Modify `src/rag/interfaces.py`, `src/rag/config.py`; Create `src/rag/session_store.py`, `tests/test_session_store.py`

**Interfaces:** `SessionStore.get_history(sid)->list[Message]` / `append(sid, msg)`;`InMemorySessionStore`;`Settings.history_turns=6`、`condense=True`。

- [ ] **Step 1: 写失败测试** `tests/test_session_store.py`:

```python
from rag.session_store import InMemorySessionStore
from rag.models import Message


def test_append_and_get_in_order():
    s = InMemorySessionStore()
    s.append("a", Message(role="user", content="q1"))
    s.append("a", Message(role="assistant", content="r1"))
    h = s.get_history("a")
    assert [m.content for m in h] == ["q1", "r1"]


def test_sessions_isolated():
    s = InMemorySessionStore()
    s.append("a", Message(role="user", content="qa"))
    assert s.get_history("b") == []


def test_unknown_session_empty():
    assert InMemorySessionStore().get_history("nope") == []
```

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_session_store.py -v` → FAIL
- [ ] **Step 3: 实现**
`src/rag/interfaces.py` 追加:
```python
class SessionStore(ABC):
    """多轮会话历史存储(M12)。"""
    @abstractmethod
    def get_history(self, session_id: str) -> list["Message"]: ...
    @abstractmethod
    def append(self, session_id: str, message: "Message") -> None: ...
```
(确保 `Message` 已在 interfaces 的 import 里;它已从 rag.models 导入。)
`src/rag/session_store.py`:
```python
"""会话历史存储:v1 内存实现。Redis 等持久化实现同接口即可扩展。"""
from rag.interfaces import SessionStore
from rag.models import Message


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._data: dict[str, list[Message]] = {}

    def get_history(self, session_id: str) -> list[Message]:
        return list(self._data.get(session_id, []))

    def append(self, session_id: str, message: Message) -> None:
        self._data.setdefault(session_id, []).append(message)
```
`src/rag/config.py` 加:`history_turns: int = 6` 和 `condense: bool = True`。

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_session_store.py -v` → PASS;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/interfaces.py src/rag/session_store.py src/rag/config.py tests/test_session_store.py && git commit -m "feat(M12): SessionStore 接口 + 内存实现 + 会话配置"`

---

### Task 2: condense 追问改写

**Files:** Create `src/rag/condense.py`, `tests/test_condense.py`

**Interfaces:** `condense_question(llm, question, history) -> str`(无历史→原问题;空回复→兜底原问题)。

- [ ] **Step 1: 写失败测试** `tests/test_condense.py`:

```python
from rag.condense import condense_question
from rag.models import Message


class _FakeLLM:
    def __init__(self, reply): self.reply = reply; self.seen = None
    def generate(self, prompt): self.seen = prompt; return self.reply


def test_no_history_returns_original_without_llm_call():
    llm = _FakeLLM("SHOULD NOT BE USED")
    out = condense_question(llm, "它能限定类型吗?", [])
    assert out == "它能限定类型吗?"
    assert llm.seen is None                      # 无历史不调 LLM


def test_with_history_uses_rewrite():
    hist = [Message(role="user", content="FastAPI 路径参数怎么声明?"),
            Message(role="assistant", content="用花括号 /items/{item_id}")]
    llm = _FakeLLM("路径参数可以限定类型吗?")
    out = condense_question(llm, "它能限定类型吗?", hist)
    assert out == "路径参数可以限定类型吗?"
    assert "路径参数怎么声明" in llm.seen           # 历史进了 prompt


def test_empty_reply_falls_back_to_original():
    hist = [Message(role="user", content="x"), Message(role="assistant", content="y")]
    out = condense_question(_FakeLLM("   "), "它呢?", hist)
    assert out == "它呢?"
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `src/rag/condense.py`:
```python
"""把依赖上文的追问改写成可独立检索的问题(M12)。"""
from rag.interfaces import LLM
from rag.models import Message

_CONDENSE_PROMPT = """给定下面的【对话历史】和用户的【追问】,把追问改写成一个脱离历史也能\
独立理解与检索的完整问题。只输出改写后的问题,不要解释、不要加引号。

【对话历史】
{history}

【追问】{question}
【独立问题】"""


def _render(history: list[Message]) -> str:
    role = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role.get(m.role, m.role)}:{m.content}" for m in history)


def condense_question(llm: LLM, question: str, history: list[Message]) -> str:
    if not history:
        return question
    reply = llm.generate(_CONDENSE_PROMPT.format(
        history=_render(history), question=question)).strip()
    return reply or question
```

- [ ] **Step 4: 确认通过** → PASS;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/condense.py tests/test_condense.py && git commit -m "feat(M12): condense 追问改写(无历史/空回复兜底)"`

---

### Task 3: generate.py 加可选 history

**Files:** Modify `src/rag/generate.py`; Test `tests/test_generate.py`

**Interfaces:** `build_prompt(..., history: list[Message] = [])`、`answer(..., history=[])`;history 非空时 prompt 含【对话历史】块;默认空=现状。

- [ ] **Step 1: 写失败测试** 在 `tests/test_generate.py` 追加:
```python
from rag.models import Message

def test_build_prompt_includes_history_when_given():
    from rag.generate import build_prompt
    hist = [Message(role="user", content="上一个问题"),
            Message(role="assistant", content="上一个回答")]
    p = build_prompt("现在的问题", [_rc_simple()], history=hist)
    assert "上一个问题" in p and "上一个回答" in p

def test_build_prompt_no_history_unchanged():
    from rag.generate import build_prompt
    p = build_prompt("q", [_rc_simple()])
    assert "对话历史" not in p           # 默认不插历史块
```
(`_rc_simple` 复用文件里已有的 helper;若名字不同则用现有的。)

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** 在 `src/rag/generate.py`:import `Message`;`build_prompt`/`answer` 加 `history: list["Message"] = []` 参数;在 system 段与【资料】之间(或【问题】之前)插入历史块:
```python
def build_prompt(question, chunks, persona=_DEFAULT_PERSONA, refusal_text=_DEFAULT_REFUSAL,
                 history=[]):
    system = ( ... 现有不变 ... )
    ... context 组装不变 ...
    hist_block = ""
    if history:
        role = {"user": "用户", "assistant": "助手"}
        lines = "\n".join(f"{role.get(m.role, m.role)}:{m.content}" for m in history)
        hist_block = f"【对话历史】\n{lines}\n\n"
    return (f"{system}\n\n{hist_block}【资料】\n{context}\n\n【问题】\n{question}\n\n【回答】")

def answer(question, chunks, llm, persona=_DEFAULT_PERSONA, refusal_text=_DEFAULT_REFUSAL,
           history=[]):
    text = llm.generate(build_prompt(question, chunks, persona, refusal_text, history))
    ... sources 不变 ...
```
(默认 `history=[]` 空;注意可变默认参数只读不改,安全。)

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_generate.py -v` → PASS(含现有回归);全量绿。
- [ ] **Step 5: 提交** `git add src/rag/generate.py tests/test_generate.py && git commit -m "feat(M12): generate 支持可选对话历史(默认空=现状)"`

---

### Task 4: ConversationalRag

**Files:** Create `src/rag/conversation.py`, `tests/test_conversation.py`

**Interfaces:** `ConversationalRag(retriever, llm, store, persona, refusal_text, history_turns=6, condense=True).chat(session_id, question, library=None) -> Answer`。retriever 是 `callable(query, library=None) -> list[RetrievedChunk]`。

- [ ] **Step 1: 写失败测试** `tests/test_conversation.py`:
```python
from rag.conversation import ConversationalRag
from rag.session_store import InMemorySessionStore
from rag.models import Chunk, RetrievedChunk


def _rc(src, text="资料"):
    return RetrievedChunk(chunk=Chunk(id=src, text=text, source=src, library="l", chunk_index=0), score=1.0)


class _LLM:
    def __init__(self): self.gen_prompts = []
    def generate(self, prompt):
        self.gen_prompts.append(prompt)
        return "路径参数可以限定类型" if "独立问题" in prompt else "答案"


def test_first_turn_no_condense_then_second_turn_condenses():
    llm = _LLM(); store = InMemorySessionStore()
    seen = {}
    def retriever(query, library=None):
        seen["last_query"] = query
        return [_rc("fastapi/tutorial-path-params.md")]
    conv = ConversationalRag(retriever, llm, store, persona="p", refusal_text="拒",
                             history_turns=6, condense=True)

    a1 = conv.chat("s1", "FastAPI 路径参数怎么声明?")
    assert seen["last_query"] == "FastAPI 路径参数怎么声明?"     # 首轮不 condense
    assert a1.text == "答案"

    conv.chat("s1", "它能限定类型吗?")
    assert seen["last_query"] == "路径参数可以限定类型"           # 第二轮:检索用 condense 后的独立问题
    assert len(store.get_history("s1")) == 4                    # 两轮 user+assistant
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现** `src/rag/conversation.py`:
```python
"""多轮会话编排(M12):condense 追问 → 复用现有检索 → 带历史生成 → 存回历史。"""
from rag.models import Answer, Message
from rag.condense import condense_question
from rag.generate import answer as generate_answer


class ConversationalRag:
    def __init__(self, retriever, llm, store, persona, refusal_text,
                 history_turns: int = 6, condense: bool = True):
        self.retriever = retriever          # callable(query, library=None) -> list[RetrievedChunk]
        self.llm = llm
        self.store = store
        self.persona = persona
        self.refusal_text = refusal_text
        self.history_turns = history_turns
        self.condense = condense

    def chat(self, session_id: str, question: str, library=None) -> Answer:
        history = self.store.get_history(session_id)
        standalone = (condense_question(self.llm, question, history)
                      if (self.condense and history) else question)
        chunks = self.retriever(standalone, library)
        ans = generate_answer(question, chunks, self.llm,
                              persona=self.persona, refusal_text=self.refusal_text,
                              history=history[-self.history_turns:])
        self.store.append(session_id, Message(role="user", content=question))
        self.store.append(session_id, Message(role="assistant", content=ans.text))
        return ans
```

- [ ] **Step 4: 确认通过** → PASS;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/conversation.py tests/test_conversation.py && git commit -m "feat(M12): ConversationalRag 多轮编排"`

---

### Task 5: /chat 端点 + serve 装配

**Files:** Modify `src/rag/api.py`, `scripts/serve.py`; Test `tests/test_api.py`

**Interfaces:** `create_app(pipeline, store, agent=None, conversation=None)`;`POST /chat {session_id, question, library?}`;serve 装配 ConversationalRag 注入。

- [ ] **Step 1: 写失败测试** 在 `tests/test_api.py` 追加(仿现有 fake 注入风格):
```python
def test_chat_endpoint_routes_to_conversation():
    from fastapi.testclient import TestClient
    from rag.api import create_app
    from rag.models import Answer

    class _FakeConv:
        def __init__(self): self.calls = []
        def chat(self, session_id, question, library=None):
            self.calls.append((session_id, question))
            return Answer(text=f"[{session_id}] {question}", sources=[])

    class _FakeStore:      # VectorStore stub for /libraries,/health
        def list_libraries(self): return ["fastapi"]

    conv = _FakeConv()
    app = create_app(pipeline=None, store=_FakeStore(), conversation=conv)
    c = TestClient(app)
    r = c.post("/chat", json={"session_id": "s1", "question": "hi"})
    assert r.status_code == 200
    assert r.json()["text"] == "[s1] hi"
    assert conv.calls == [("s1", "hi")]


def test_chat_503_when_not_configured():
    from fastapi.testclient import TestClient
    from rag.api import create_app
    class _S:
        def list_libraries(self): return []
    c = TestClient(create_app(pipeline=None, store=_S()))
    assert c.post("/chat", json={"session_id": "s", "question": "q"}).status_code == 503
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现**
`src/rag/api.py`:加 `ChatRequest(BaseModel)`(session_id: str, question: str(非空校验同 AskRequest), library: str|None=None);`create_app(..., conversation=None)`;端点:
```python
    @app.post("/chat", response_model=Answer)
    def chat(req: ChatRequest) -> Answer:
        if conversation is None:
            raise HTTPException(status_code=503, detail="conversation 未配置")
        return conversation.chat(req.session_id, req.question, library=req.library)
```
`scripts/serve.py` `build_app`:构造 `store_sess = InMemorySessionStore()`;retriever 偏函数(已有,复用带 hybrid/rerank/改写/前缀那个)接受 (query, library);`conversation = ConversationalRag(retriever2, llm, store_sess, profile.persona, profile.refusal_text, settings.history_turns, settings.condense)`,`create_app(..., conversation=conversation)`。
  - 注意:现有 retriever 闭包签名是 `(query, library=None, top_k=...)`;ConversationalRag 调 `retriever(standalone, library)` → 兼容(top_k 有默认)。若签名不符,包一个 `lambda q, library=None: retrieve(...)`。

- [ ] **Step 4: 确认通过 + 脚本语法** `.venv/bin/python -m pytest tests/test_api.py -v`;`.venv/bin/python -c "import ast; ast.parse(open('scripts/serve.py').read())"`;全量绿。
- [ ] **Step 5: 提交** `git add src/rag/api.py scripts/serve.py tests/test_api.py && git commit -m "feat(M12): /chat 端点 + serve 装配 ConversationalRag"`

---

### Task 6: 多轮演示验证(实机,控制器执行)

- [ ] **Step 1: 脚本化两轮** 直接构造 ConversationalRag(真实 retriever+OllamaLLM+InMemory store)或起服务打 /chat:
  第 1 轮 "FastAPI 路径参数怎么声明?" → 第 2 轮 "它能限定类型吗?"。
- [ ] **Step 2: 确认** condense 把第 2 轮"它"补成"路径参数…",检索命中 tutorial-path-params,回答顺上下文。
- [ ] **Step 3: 记 notes** 一段多轮能力说明 + 演示结论(诚实:仅演示、无指标)。

---

## Self-Review
- Spec §1 SessionStore→T1;§2 condense→T2;§3 generate history→T3;§3 ConversationalRag→T4;§5 /chat+装配→T5;§9 验证→T6;§10 文件全覆盖;§11 非目标未越界(不做 compaction/Redis/agent 多轮)。
- 回归:generate history 默认空=现状(T3 保留现有测试);单轮 /ask、/agent/ask 不改(T5 只加 /chat + create_app 新增可选参数,默认 None)。
- 类型一致:retriever `callable(query, library=None)->list[RetrievedChunk]`(T4 定义,T5 装配匹配);Message 复用;history 传参链 generate←conversation。
- 兜底:condense 无历史/空回复→原问题(不崩、不比单轮差)。
