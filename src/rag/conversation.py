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
                              history=history[-2 * self.history_turns:])
        self.store.append(session_id, Message(role="user", content=question))
        self.store.append(session_id, Message(role="assistant", content=ans.text))
        return ans
