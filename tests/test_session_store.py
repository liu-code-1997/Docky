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
