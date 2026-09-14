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
