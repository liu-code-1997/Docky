from rag.interfaces import Embedder, LLM
from rag.models import Chunk
from rag.providers.qdrant_store import QdrantStore
from rag.pipeline import RagPipeline


class FakeEmbedder(Embedder):
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "fastapi" in text.lower() else [0.0, 1.0, 0.0]

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


class FakeLLM(LLM):
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "答案"


def _store() -> QdrantStore:
    s = QdrantStore(location=":memory:", collection_name="test")
    s.ensure_collection(vector_size=3)
    s.upsert(
        [Chunk(id="fastapi/a.md::0", text="FastAPI 资料", source="fastapi/a.md",
               library="fastapi", chunk_index=0),
         Chunk(id="qdrant/b.md::0", text="Qdrant 资料", source="qdrant/b.md",
               library="qdrant", chunk_index=0)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    return s


def test_ask_runs_retrieve_then_generate():
    llm = FakeLLM()
    pipe = RagPipeline(FakeEmbedder(), _store(), llm, top_k=1)
    ans = pipe.ask("what is fastapi")

    assert ans.text == "答案"
    assert ans.sources == ["fastapi/a.md"]   # 检索到的来源透传到答案
    assert "FastAPI 资料" in llm.last_prompt  # 检索到的资料确实进了 prompt


def test_ask_forwards_library_filter():
    pipe = RagPipeline(FakeEmbedder(), _store(), FakeLLM(), top_k=5)
    ans = pipe.ask("what is fastapi", library="qdrant")
    assert ans.sources == ["qdrant/b.md"]  # 被 library 过滤限定
