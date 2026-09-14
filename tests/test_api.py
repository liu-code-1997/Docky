from fastapi.testclient import TestClient
from rag.api import create_app
from rag.models import Answer


class FakePipeline:
    """记录收到的参数,返回固定 Answer。"""
    def __init__(self):
        self.calls = []

    def ask(self, question: str, library: str | None = None) -> Answer:
        self.calls.append((question, library))
        return Answer(text="这是答案。", sources=["fastapi/index.md"])


class FakeStore:
    def __init__(self, libraries, ok=True):
        self._libraries = libraries
        self._ok = ok

    def list_libraries(self) -> list[str]:
        if not self._ok:
            raise RuntimeError("qdrant down")
        return self._libraries


class FakeAgent:
    def __init__(self):
        self.calls = []

    def ask(self, question: str, library: str | None = None) -> Answer:
        self.calls.append((question, library))
        return Answer(text="agent 答案", sources=["fastapi/a.md", "fastapi/b.md"])


def _client(pipeline=None, store=None, agent=None) -> TestClient:
    pipeline = pipeline or FakePipeline()
    store = store or FakeStore(["fastapi"])
    return TestClient(create_app(pipeline, store, agent=agent))


def test_ask_returns_answer_and_sources():
    pipe = FakePipeline()
    client = _client(pipeline=pipe)
    resp = client.post("/ask", json={"question": "路径参数怎么写?", "library": "fastapi"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "这是答案。"
    assert body["sources"] == ["fastapi/index.md"]
    # question 与 library 都被透传给 pipeline
    assert pipe.calls == [("路径参数怎么写?", "fastapi")]


def test_ask_library_is_optional():
    pipe = FakePipeline()
    client = _client(pipeline=pipe)
    resp = client.post("/ask", json={"question": "随便问问"})
    assert resp.status_code == 200
    assert pipe.calls == [("随便问问", None)]


def test_ask_rejects_empty_question():
    client = _client()
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422  # 校验失败


def test_ask_requires_question_field():
    client = _client()
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_libraries_endpoint_lists_libraries():
    client = _client(store=FakeStore(["fastapi", "qdrant"]))
    resp = client.get("/libraries")
    assert resp.status_code == 200
    assert resp.json() == {"libraries": ["fastapi", "qdrant"]}


def test_health_ok_when_store_reachable():
    client = _client(store=FakeStore(["fastapi"], ok=True))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_degraded_when_store_unreachable():
    client = _client(store=FakeStore([], ok=False))
    resp = client.get("/health")
    assert resp.status_code == 503


def test_agent_ask_returns_answer():
    agent = FakeAgent()
    client = _client(agent=agent)
    resp = client.post("/agent/ask", json={"question": "路径参数?", "library": "fastapi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "agent 答案"
    assert body["sources"] == ["fastapi/a.md", "fastapi/b.md"]
    assert agent.calls == [("路径参数?", "fastapi")]


def test_agent_ask_validates_blank_question():
    client = _client(agent=FakeAgent())
    resp = client.post("/agent/ask", json={"question": "  "})
    assert resp.status_code == 422


def test_agent_ask_503_when_agent_not_configured():
    # 没注入 agent 时,端点应返回 503 而非崩溃
    client = _client(agent=None)
    resp = client.post("/agent/ask", json={"question": "q"})
    assert resp.status_code == 503


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
