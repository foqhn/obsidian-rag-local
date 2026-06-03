# MVP（CLI）— フェーズ2の Web UI は `uv run streamlit run app.py` を使用してください。
import os
from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
# --- 変更点: 古い chains ではなく、最新の core モジュールを使用 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. Obsidian Vaultのパス設定（作成したテスト用フォルダを指定）
VAULT_PATH = r"C:\Users\harah\OneDrive - 国立大学法人東海国立大学機構\名古屋大学"

# 2. ドキュメントの読み込み
print("Obsidianノートを読み込んでいます...")
loader = ObsidianLoader(VAULT_PATH)
docs = loader.load()
print(f"{len(docs)} 件のノートを読み込みました。")

# 3. ベクトル化モデルの準備
print("ベクトル化モデルを準備しています...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# 4. ベクトルデータベース（Chroma）の構築
print("ベクトルデータベースを構築しています...")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. ローカルLLM（Ollama）の設定
# ※モデル名はダウンロードしたOllamaのモデルに合わせてください（例: "llama3"）
llm = ChatOllama(model="gemma4", temperature=0)

# 6. プロンプトの作成
template = """あなたは優秀なアシスタントです。以下のコンテキスト（関連ノート）を使用して質問に答えてください。
答えがわからない場合は、適当に答えず「わからない」と答えてください。

コンテキスト:
{context}

質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. LCELを用いた最新のRAGチェーン構築
# 検索してきたノート（ドキュメント）を文字列に結合する関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 回答を生成するためのフロー
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

# 検索結果（ソース）と回答を両方取得するためのフロー
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# 8. 実行テスト
query = "Gritcatについて教えてください。" 
print(f"\n質問: {query}")
print("回答を生成中...\n")

# チェーンの実行
response = rag_chain_with_source.invoke(query)

print("==================== 回答 ====================")
print(response["answer"])
print("\n==================== 参照元 ====================")
for doc in response["context"]:
    # ObsidianLoaderで取得したメタデータからファイルパスを表示
    print(f"ファイル: {doc.metadata.get('source', '不明')}")