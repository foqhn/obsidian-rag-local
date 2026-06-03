import gc
import shutil
import time
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MARKDOWN_HEADERS,
    VAULT_PATH,
)

CHROMA_COLLECTION_NAME = "langchain"


def _release_chroma_client(client) -> None:
    del client
    gc.collect()


def close_vectorstore(vectorstore: Chroma | None) -> None:
    """LangChain Chroma が保持する DB 接続を解放する（Windows のファイルロック対策）。"""
    if vectorstore is None:
        return
    for attr in ("_client", "_collection"):
        if hasattr(vectorstore, attr):
            setattr(vectorstore, attr, None)
    del vectorstore
    gc.collect()
    time.sleep(0.3)


def clear_vectorstore_data() -> None:
    """persist ディレクトリは残し、コレクションのみ削除する（rmtree より安全）。"""
    if not (CHROMA_PERSIST_DIR / "chroma.sqlite3").exists():
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        return

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        for col in client.list_collections():
            client.delete_collection(col.name)
    finally:
        _release_chroma_client(client)
    gc.collect()
    time.sleep(0.3)


def _rmtree_with_retry(path: Path, retries: int = 5) -> None:
    for attempt in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            gc.collect()
            time.sleep(0.5 * (attempt + 1))


def get_collection_count() -> int:
    """埋め込みモデルを読み込まずに Chroma の件数を取得する。"""
    if not (CHROMA_PERSIST_DIR / "chroma.sqlite3").exists():
        return 0
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        return sum(col.count() for col in client.list_collections())
    finally:
        _release_chroma_client(client)


def index_exists() -> bool:
    """DBファイルの有無ではなく、検索可能なドキュメントが1件以上あるかで判定する。"""
    return get_collection_count() > 0


def load_documents(vault_path: str | Path = VAULT_PATH):
    loader = ObsidianLoader(str(vault_path))
    return loader.load()


def split_documents(docs):
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for doc in docs:
        md_chunks = md_splitter.split_text(doc.page_content)
        for chunk in md_chunks:
            chunk.metadata = {**doc.metadata, **chunk.metadata}
        chunks.extend(char_splitter.split_documents(md_chunks))
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    embeddings = embeddings or get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )


def rebuild_index(
    vault_path: str | Path = VAULT_PATH,
    embeddings: HuggingFaceEmbeddings | None = None,
    vectorstore: Chroma | None = None,
) -> Chroma:
    embeddings = embeddings or get_embeddings()
    close_vectorstore(vectorstore)
    try:
        clear_vectorstore_data()
    except Exception:
        _rmtree_with_retry(CHROMA_PERSIST_DIR)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents(vault_path)
    chunks = split_documents(docs)

    if not chunks:
        raise ValueError(
            "分割後のチャンクが0件です。Vault内のノート内容を確認してください。"
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    count = vectorstore._collection.count()
    if count == 0:
        raise RuntimeError(
            "インデックス構築後もドキュメントが0件です。構築に失敗した可能性があります。"
        )
    return vectorstore
