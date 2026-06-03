from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, RETRIEVER_K


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "不明")
    filename = Path(source).name if source != "不明" else "不明"

    headers = []
    for key in ("Header 1", "Header 2", "Header 3"):
        value = doc.metadata.get(key)
        if value:
            headers.append(value)

    if headers:
        return f"{filename} > {' > '.join(headers)}"
    return filename


class RAGService:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.history = ChatMessageHistory()
        self._prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なアシスタントです。以下の会話履歴とコンテキスト（関連ノート）を使用して質問に答えてください。
答えがわからない場合は、適当に答えず「わからない」と答えてください。

会話履歴:
{chat_history}

コンテキスト:
{context}

質問: {question}
"""
        )
        self._chain = self._prompt | self.llm | StrOutputParser()

    def _format_history(self) -> str:
        if not self.history.messages:
            return "（なし）"
        lines = []
        for msg in self.history.messages:
            role = "ユーザー" if msg.type == "human" else "アシスタント"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        self.history.clear()

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def _empty_index_message(self) -> str:
        return (
            "ベクトルDBにドキュメントが登録されていないため、ノートを参照できません。"
            "サイドバーの「インデックスを更新」を実行してください。"
        )

    def invoke(self, question: str) -> tuple[str, list[Document]]:
        context_docs = self.retrieve(question)
        if not context_docs:
            answer = self._empty_index_message()
            self.history.add_user_message(question)
            self.history.add_ai_message(answer)
            return answer, context_docs
        answer = self._chain.invoke(
            {
                "chat_history": self._format_history(),
                "context": format_docs(context_docs),
                "question": question,
            }
        )
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        return answer, context_docs

    def stream(self, question: str):
        context_docs = self.retrieve(question)
        if not context_docs:
            answer = self._empty_index_message()
            self.history.add_user_message(question)
            self.history.add_ai_message(answer)
            yield ("token", answer)
            yield ("done", context_docs)
            return
        inputs = {
            "chat_history": self._format_history(),
            "context": format_docs(context_docs),
            "question": question,
        }

        full_answer: list[str] = []
        for chunk in self._chain.stream(inputs):
            full_answer.append(chunk)
            yield ("token", chunk)

        answer = "".join(full_answer)
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        yield ("done", context_docs)
