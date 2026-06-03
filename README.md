Obsidian Local RAG System (MVP)
ObsidianのVault（Markdownノート群）を読み込み、完全ローカル環境でAIとチャット（RAG）ができるシステムです。外部のAPIを使用しないため、機密情報やプライベートなメモも安全に扱うことができます。
🌟 特徴 (Features)
完全ローカル稼働: Ollamaとローカルモデルを使用するため、データがインターネット外部に送信されません。
Obsidianネイティブ対応: LangChainの ObsidianLoader を使用し、.md ファイルやメタデータを直接読み込みます。
ソース提示機能: AIが回答を生成する際、根拠となったObsidianノートのファイル名を提示します。
最新のLangChain記法: 安定性の高い LCEL (LangChain Expression Language) アーキテクチャを採用しています。
🛠️ 技術スタック (Tech Stack)
言語: Python 3.10+
オーケストレーション: LangChain (LCEL)
LLM (テキスト生成): Ollama (例: llama3)
Embedding (ベクトル化): HuggingFace (intfloat/multilingual-e5-large)
Vector DB: Chroma
🚀 セットアップ (Setup)
1. 前提条件 (Prerequisites)
事前に以下のソフトウェアがインストールされている必要があります。
Python (3.10以上を推奨)
Ollama (https://ollama.com/)
Ollamaをインストール後、ターミナルでLLMモデル（例：Llama 3）をダウンロードしておいてください。
code
Bash
ollama pull llama3
2. 環境構築 (Installation)
プロジェクトのフォルダで仮想環境を作成し、必要なライブラリをインストールします。
code
Bash
# 仮想環境の作成と有効化 (Windowsの場合)
python -m venv venv
.\venv\Scripts\activate

# (Mac/Linuxの場合)
python3 -m venv venv
source venv/bin/activate

# 必須ライブラリのインストール
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama sentence-transformers chromadb
使い方 (Usage)
1. ノートの準備
プロジェクト直下に vault というフォルダを作成し、その中に読み込ませたい Obsidianのノート（.md ファイル）を配置してください。（※実際のObsidian Vaultのパスに変更することも可能です）
code
Text
📁 obsidian-rag-local/
 ├── 📁 venv/
 ├── 📁 vault/       ← ここに .md ファイルを入れる
 │    ├── note1.md
 │    └── note2.md
 ├── main.py
 └── README.md
2. スクリプトの実行
ターミナルで以下のコマンドを実行します。
code
Bash
python main.py
※初回実行時は、HuggingFaceのベクトル化モデル（約2GB）のダウンロードが自動で行われるため、数分時間がかかります。
実行すると、ノートの読み込みとベクトル化が行われ、ターミナル上で設定した質問に対するAIの回答と、参照元のファイル名が出力されます。
⚙️ カスタマイズ (Configuration)
main.py 内の以下の変数を変更することで、カスタマイズが可能です。
Vaultパスの変更:
自分の実際のObsidian Vaultフォルダを指定できます。
code
Python
VAULT_PATH = "C:/Users/YourName/Documents/Obsidian"
LLMモデルの変更:
Ollamaでダウンロードした別のモデル（gemma2 など）に変更できます。
code
Python
llm = ChatOllama(model="llama3", temperature=0)
質問内容の変更:
code
Python
query = "ノートに書かれている重要な情報を教えてください。" # ← ここを書き換える
🗺️ 今後のロードマップ (Roadmap)
現在のシステムはMVP（Minimum Viable Product）です。今後は以下の機能追加を予定しています。

チャンク分割の最適化: MarkdownHeaderTextSplitter を導入し、見出し（H1, H2）単位で文脈を壊さずにテキストを分割する機能。

対話型UIの実装: Streamlit等を用いた、ブラウザ上でチャットができるインターフェースの構築。

ハイブリッド検索: ベクトル検索（意味検索）と BM25（キーワード検索）の組み合わせによる、固有名詞の検索精度向上。

メタデータフィルタリング: Obsidianのタグやプロパティ（フロントマター）を使った検索の絞り込み。
ライセンス
個人利用を目的としたプロトタイプ実装です。