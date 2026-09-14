"""FastAPI HTTP 外壳:把 RagPipeline 包成 Web 服务。

用工厂函数 create_app(pipeline, store) 注入依赖:
- 测试塞 fake,生产塞真实 provider(见 scripts/serve.py)。
- app 本身不做装配,只负责 HTTP <-> pipeline 的转换与校验。
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from rag.models import Answer


class AskRequest(BaseModel):
    question: str
    library: str | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question 不能为空")
        return v.strip()


class ChatRequest(BaseModel):
    session_id: str
    question: str
    library: str | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question 不能为空")
        return v.strip()


class LibrariesResponse(BaseModel):
    libraries: list[str]


def create_app(pipeline, store, agent=None, conversation=None) -> FastAPI:
    app = FastAPI(title="Docky", description="Docky · 面向学习的技术文档 RAG 问答小助手")

    @app.post("/ask", response_model=Answer)
    def ask(req: AskRequest) -> Answer:
        return pipeline.ask(req.question, library=req.library)

    @app.post("/agent/ask", response_model=Answer)
    def agent_ask(req: AskRequest) -> Answer:
        # M6:agent 自主检索循环。未配置 agent 时返回 503(而非崩溃)。
        if agent is None:
            raise HTTPException(status_code=503, detail="agent 未配置")
        return agent.ask(req.question, library=req.library)

    @app.post("/chat", response_model=Answer)
    def chat(req: ChatRequest) -> Answer:
        # M12:多轮会话。未配置 conversation 时返回 503(而非崩溃)。
        if conversation is None:
            raise HTTPException(status_code=503, detail="conversation 未配置")
        return conversation.chat(req.session_id, req.question, library=req.library)

    @app.get("/libraries", response_model=LibrariesResponse)
    def libraries() -> LibrariesResponse:
        return LibrariesResponse(libraries=store.list_libraries())

    @app.get("/health")
    def health():
        # 探活:能列出 library 说明 Qdrant 连得上。
        try:
            store.list_libraries()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"store unreachable: {exc}")
        return {"status": "ok"}

    return app
