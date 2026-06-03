import streamlit as st

from config import CHROMA_PERSIST_DIR, OLLAMA_MODEL, VAULT_PATH
from indexing import (
    close_vectorstore,
    get_collection_count,
    get_embeddings,
    index_exists,
    load_vectorstore,
    rebuild_index,
)
from rag import RAGService, format_source_label


@st.cache_resource(show_spinner="埋め込みモデルを読み込んでいます…")
def cached_embeddings():
    return get_embeddings()


def ensure_rag() -> RAGService | None:
    if st.session_state.get("rag") is not None:
        return st.session_state.rag

    if not index_exists():
        return None

    vectorstore = load_vectorstore(cached_embeddings())
    st.session_state.rag = RAGService(vectorstore)
    return st.session_state.rag


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("参照したノート"):
        for doc in sources:
            st.markdown(f"- **{format_source_label(doc)}**")


def main() -> None:
    st.set_page_config(page_title="Obsidian Local RAG", page_icon="📓", layout="wide")
    st.title("Obsidian Local RAG")
    st.caption("完全ローカル環境で Obsidian ノートを検索・参照して回答します。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("設定")
        st.text(f"Vault: {VAULT_PATH}")
        st.text(f"DB: {CHROMA_PERSIST_DIR}")
        st.text(f"LLM: {OLLAMA_MODEL}")

        doc_count = get_collection_count()
        if doc_count > 0:
            st.success(f"ベクトルDB: 読み込み済み（{doc_count:,} チャンク）")
        elif (CHROMA_PERSIST_DIR / "chroma.sqlite3").exists():
            st.error(
                "ベクトルDBファイルはありますが中身が空です。"
                "「インデックスを更新」を実行してください（初回は10〜30分かかることがあります）。"
            )
        else:
            st.warning("ベクトルDB: 未構築")

        if st.button("インデックスを更新", use_container_width=True):
            old_rag = st.session_state.pop("rag", None)
            if old_rag is not None:
                close_vectorstore(old_rag.vectorstore)
            with st.spinner("ノートを読み込み・分割・インデックス化しています…"):
                vectorstore = rebuild_index(embeddings=cached_embeddings())
                st.session_state.rag = RAGService(vectorstore)
            st.success("インデックスを更新しました。")
            st.rerun()

        if st.button("チャットをリセット", use_container_width=True):
            if st.session_state.get("rag"):
                st.session_state.rag.clear_history()
            st.session_state.messages = []
            st.rerun()

    rag = ensure_rag()
    if rag is None:
        st.info(
            "サイドバーの「インデックスを更新」を押して Obsidian ノートの "
            "ベクトルDBを構築してください。Vaultが大きい場合、完了まで時間がかかります。"
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))

    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            sources = None
            full_response = ""

            with st.spinner("回答を生成中…"):
                for event, payload in rag.stream(prompt):
                    if event == "token":
                        full_response += payload
                        placeholder.markdown(full_response + "▌")
                    elif event == "done":
                        sources = payload

            placeholder.markdown(full_response)
            render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
