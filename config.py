import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"

VAULT_PATH = os.environ.get(
    "OBSIDIAN_VAULT_PATH",
    r"C:\Users\harah\OneDrive - 国立大学法人東海国立大学機構\名古屋大学",
)

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")
RETRIEVER_K = 3

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
