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


def _client(pipeline=None, store=None) -> TestClient:
    pipeline = pipeline or FakePipeline()
    store = store or FakeStore(["fastapi"])
    return TestClient(create_app(pipeline, store))


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
