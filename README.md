# Obsidian Local RAG System

Obsidian の Vault（Markdown ノート群）を読み込み、完全ローカル環境で AI とチャット（RAG）できるシステムです。外部 API を使わないため、機密情報やプライベートなメモも安全に扱えます。

## 特徴

- **完全ローカル稼働**: Ollama とローカル埋め込みモデルを使用（データは外部に送信されません）
- **Obsidian 対応**: `ObsidianLoader` で `.md` とメタデータを読み込み
- **見出しベースのチャンク化**: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` で文脈を保持
- **会話履歴（メモリ）**: `ChatMessageHistory` により同一セッション内の連続質問に対応
- **参照元の表示**: ファイル名と見出し（`note.md > 見出し1 > 見出し2`）をエキスパンダーで表示
- **永続化ベクトルDB**: Chroma を `data/chroma_db` に保存。起動時は既存 DB を読み込み、更新はボタン操作時のみ

## 技術スタック

| 項目 | 技術 |
|------|------|
| 言語 | Python 3.11+ |
| パッケージ管理 | [uv](https://github.com/astral-sh/uv) |
| UI | Streamlit |
| オーケストレーション | LangChain (LCEL) |
| LLM | Ollama |
| Embedding | HuggingFace (`intfloat/multilingual-e5-large`) |
| Vector DB | Chroma（ディスク永続化） |

## セットアップ

### 前提条件

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/)（利用するモデルを `ollama pull` で取得）

### インストール

```bash
uv sync
```

### 環境変数（任意）

| 変数 | 説明 |
|------|------|
| `OBSIDIAN_VAULT_PATH` | Obsidian Vault のパス（未設定時は `config.py` のデフォルト） |
| `OLLAMA_MODEL` | Ollama モデル名（デフォルト: `gemma4`） |

## 使い方

### Web UI（フェーズ2・推奨）

```bash
uv run streamlit run app.py
```

1. ブラウザでチャット画面が開きます
2. 初回または Vault 更新後は、サイドバーの **「インデックスを更新」** でベクトル DB を構築
3. 質問を入力すると、ストリーミング表示で回答が表示されます
4. 新しい話題では **「チャットをリセット」** で会話履歴をクリア

### CLI（MVP・フェーズ1）

```bash
uv run python main.py
```

`main.py` は動作確認済みの MVP 実装です。ターミナル上で固定質問を 1 回実行します。

## プロジェクト構成

```
obsidian-rag-local/
├── app.py          # Streamlit チャット UI
├── config.py       # パス・モデル名などの設定
├── indexing.py     # 読み込み・チャンク化・Chroma 永続化
├── rag.py          # RAG チェーン・会話履歴・ストリーミング
├── main.py         # MVP（CLI）
├── data/chroma_db/ # 永続化ベクトルDB（git 対象外）
└── pyproject.toml
```

## カスタマイズ

`config.py` または環境変数で以下を変更できます。

- Vault パス（`VAULT_PATH` / `OBSIDIAN_VAULT_PATH`）
- Ollama モデル（`OLLAMA_MODEL`）
- 検索件数（`RETRIEVER_K`）
- チャンクサイズ（`CHUNK_SIZE`, `CHUNK_OVERLAP`）

## ライセンス

個人利用を目的としたプロトタイプ実装です。
