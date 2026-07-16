"""RAG 问答流水线:把 retrieve 与 generate 两条链路串起来。

这是三条链路唯一的汇合点;M3 的 FastAPI 直接复用 RagPipeline,
不必重复装配逻辑。只依赖接口,具体 provider 在外部装配后注入。
"""
from rag.interfaces import Embedder, VectorStore, LLM, QueryRewriter, Reranker
from rag.models import Answer
from rag.retrieve import retrieve
from rag.generate import answer as generate_answer


class RagPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLM,
                 top_k: int, rewriter: QueryRewriter | None = None,
                 reranker: Reranker | None = None, rerank_factor: int = 5):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.rewriter = rewriter    # M5②:非 None 时检索前改写查询
        self.reranker = reranker    # M5③:非 None 时检索后重排
        self.rerank_factor = rerank_factor

    def ask(self, question: str, library: str | None = None) -> Answer:
        """问题 → 检索 Top-K → 据资料生成带出处的答案。"""
        chunks = retrieve(question, self.embedder, self.store,
                          top_k=self.top_k, library=library,
                          rewriter=self.rewriter, reranker=self.reranker,
                          rerank_factor=self.rerank_factor)
        return generate_answer(question, chunks, self.llm)
