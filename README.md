# ⚖️ NyayaSetu AI

> **Bridge to Justice** — A production-ready Hybrid RAG system for Indian Criminal Law

A modular, enterprise-ready **Retrieval-Augmented Generation (RAG)** system for the three landmark Indian criminal statutes enacted in 2023–2024:

| Statute | Abbrev. | Replaces |
|---|---|---|
| Bharatiya Nyaya Sanhita, 2023 | **BNS** | Indian Penal Code (IPC) |
| Bharatiya Nagarik Suraksha Sanhita, 2023 | **BNSS** | Code of Criminal Procedure (CrPC) |
| Bharatiya Sakshya Adhiniyam, 2023 | **BSA** | Indian Evidence Act (IEA) |

Built with **LangChain**, **Weaviate Embedded**, **Streamlit**, and **Pydantic Settings**. Features native Weaviate **Hybrid Search** (inbuilt BM25 + 384-dimensional dense vectors fused via Reciprocal Rank Fusion) and citation-backed legal question answering.

---

## ✨ Key Features

- 🔍 **Hybrid Search** — Native Weaviate BM25 + dense vector search with RRF fusion (α = 0.25)
- 🧩 **Statute-Aware Chunking** — Regex-based section extraction preserving `section_no`, `title`, `act_code` metadata
- 📑 **Verifiable Citations** — Every answer cites exact statute section `[BNS • Section 103]`
- 🤖 **Multi-Provider LLM** — Groq (Llama / Qwen), Google Gemini, or OpenAI GPT-4
- 🏠 **100% Local Embeddings** — `all-MiniLM-L6-v2` (384 dims) runs on CPU, zero data leakage
- 📊 **RAGAS Evaluation** — Faithfulness & Answer Relevancy benchmarked on a golden legal Q&A dataset
- 🚫 **Fail-Fast Architecture** — Explicit `RuntimeError` on missing API keys, no silent fallback

---

## 📁 Project Structure

```text
rag-project/
│
├── data/
│   ├── raw/                    # Source PDFs (gitignored — place statute PDFs here)
│   └── processed/              # Extracted plain-text statute files
│
├── vectorstore/                # Weaviate Embedded DB (gitignored — auto-created on first run)
│
├── src/
│   ├── config.py               # Pydantic Settings — model names, chunk sizes, paths, API keys
│   │
│   ├── ingestion/
│   │   ├── loaders.py          # MultiFormatLoader: PDF / DOCX / TXT with metadata tagging
│   │   └── chunker.py          # DocumentChunker: statute-aware regex section splitter
│   │
│   ├── retrieval/
│   │   ├── embeddings.py       # Embedding model factory (HuggingFace / OpenAI)
│   │   └── vector_store.py     # VectorStoreManager: Weaviate Embedded + native hybrid search
│   │
│   ├── generation/
│   │   ├── prompts.py          # System & user prompt templates with citation enforcement
│   │   └── chain.py            # RAGPipeline: retrieve → format context → generate → cite
│   │
│   └── utils/
│       └── logger.py           # Structured logging
│
├── app/
│   └── main.py                 # Streamlit chat dashboard with provider sidebar
│
├── tests/
│   └── evaluate_ragas.py       # RAGAS benchmark runner (Faithfulness + Answer Relevancy)
│
├── eval_dataset/
│   ├── dataset.json            # 200-question golden Q&A evaluation dataset
│   └── ragas_benchmark_results.json  # Latest benchmark scores (gitignored)
│
├── .env.example                # Environment variable template
├── pyproject.toml              # Package metadata
├── reqs.txt                    # pip-installable dependencies
└── uv.lock                     # Reproducible dependency lock (uv)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Ravi92535/NyayaSetu-AI.git
cd NyayaSetu-AI

# Using pip
pip install -r reqs.txt

# Or using uv (recommended — uses uv.lock for reproducible installs)
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
# LLM Provider — choose one
LLM_PROVIDER=groq           # groq | gemini | openai
LLM_MODEL=qwen/qwen3.8-27b  # or llama-3.3-70b-versatile, gemini-1.5-flash, gpt-4o

# API Keys — only the key for your chosen provider is required
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# Optional overrides (defaults shown)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
WEAVIATE_HYBRID_ALPHA=0.25
```

> [!IMPORTANT]
> At least one valid API key (`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`) is required. The system uses a **Fail-Fast** architecture — it raises a clear `RuntimeError` immediately if no valid key is found, rather than silently degrading.

### 3. Add Statute PDFs

Place the three statute PDFs in `data/raw/`:
```
data/raw/
├── THE BHARATIYA NYAYA SANHITA, 2023.pdf
├── THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf
└── THE BHARATIYA SAKSHYA ADHINIYAM, 2023.pdf
```

### 4. Launch the App

```bash
# From project root
streamlit run app/main.py

# Or directly
python app/main.py
```

The Streamlit UI will open at `http://localhost:8501`. On first launch, it will ingest the PDFs and build the Weaviate vector index automatically.

---

## 🛠️ Module Breakdown

### 1. Ingestion (`src/ingestion/`)

**`loaders.py`** — `MultiFormatLoader`
- Handles PDF (`PyPDFLoader`), DOCX, and plain text/markdown files
- Automatically tags every document chunk with metadata: `filename`, `source`, `extension`, `file_size`

**`chunker.py`** — `DocumentChunker`
- `split_documents()`: Recursive character chunking (size=1000, overlap=200) with unique `chunk_id` generation (`{filename}_p{page}_c{index}`)
- `split_statute_sections()`: **Statute-aware regex extraction** — identifies section boundaries by number pattern (`\n103. ...`) and injects `section_no`, `title`, and `act_code` metadata into every chunk

```python
# Each legal chunk carries:
{
    "section_no": "103",
    "title": "Punishment for Murder",
    "act_code": "bns",
    "chunk_id": "bns_p12_c4"
}
```

---

### 2. Retrieval (`src/retrieval/`)

**`embeddings.py`** — Embedding factory
- Supports `huggingface` (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims, runs locally on CPU) and `openai` providers
- Deterministic hash-based fallback for offline unit testing

**`vector_store.py`** — `VectorStoreManager`
- Wraps Weaviate Embedded running locally (no external server required)
- Auto-persists index to `vectorstore/`
- Executes **native Weaviate hybrid search**: BM25 keyword matching + dense HNSW vector similarity, fused via **Reciprocal Rank Fusion (RRF)** at configurable `α` (default 0.25)
  - `α = 0.0` → pure BM25 keyword search
  - `α = 1.0` → pure dense vector search
  - `α = 0.25` → 75% semantic + 25% keyword (optimal for legal domain)

---

### 3. Generation (`src/generation/`)

**`prompts.py`**
- `RAG_SYSTEM_PROMPT`: Strict factual adherence prompt — instructs the LLM to answer **only from retrieved context**, cite sources, and explicitly state uncertainty rather than speculate
- `get_rag_prompt_template()`: Returns a `ChatPromptTemplate` with system + human message structure

**`chain.py`** — `RAGPipeline`
- End-to-end orchestrator: Hybrid Retrieval → Context Formatting → LLM Generation
- `format_context()`: Formats retrieved chunks with citation labels `[BNS Section 103: Punishment for Murder]`
- `query()`: Returns `{answer, sources, retrieved_count, latency_seconds}`
- `get_llm_model()`: Resolves the correct LLM client from env config; raises `RuntimeError` with actionable message if API key is missing

---

### 4. Interface (`app/main.py`)

Streamlit conversational dashboard:
- **Sidebar**: Provider selector (Groq / Gemini / OpenAI), model name input, live API key validation with ping test
- **Chat area**: Message history with expandable `📑 Relevant Statutory Citations` and latency display
- **Status badge**: Shows active provider/model or warns if no API key is configured

---

## 📊 RAGAS Evaluation

Evaluate the pipeline on the included 200-question golden dataset:

```bash
# Quick evaluation (5 samples)
python tests/evaluate_ragas.py

# Full benchmark (200 samples)
python tests/evaluate_ragas.py --samples 200
```

**Latest benchmark results** (2-sample fast run, Groq `qwen/qwen3.8-27b`):

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | **100.0%** | LLM never hallucinated — all claims grounded in retrieved sections |
| **Answer Relevancy** | **80.3%** | Answers directly address the question asked |

> [!NOTE]
> `Context Precision` and `Context Recall` metrics require OpenAI-tier API rate limits (3+ parallel calls). They are disabled by default for Groq free-tier compatibility. Enable them in `evaluate_ragas.py` by adding them to the `metrics` list.

---

## ⚙️ Configuration Reference

All settings live in `src/config.py` (Pydantic Settings — read from `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM backend: `groq`, `gemini`, or `openai` |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Model name for the chosen provider |
| `LLM_TEMPERATURE` | `0.0` | Strictly factual mode (no creativity) |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dimensional local sentence transformer |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `WEAVIATE_COLLECTION` | `LegalStatute` | Weaviate collection name |
| `WEAVIATE_HYBRID_ALPHA` | `0.25` | RRF fusion weight (0 = BM25, 1 = dense) |
| `WEAVIATE_URL` | `None` | `None` = embedded local; or `http://localhost:8080` for external |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│          Streamlit UI (app/main.py)     │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │     RAGPipeline.query()    │
    │    (src/generation/chain)  │
    └──────┬────────────-────────┘
           │
    ┌──────▼───────────────────────────────────┐
    │  VectorStoreManager.similarity_search()  │
    │  Weaviate Embedded — Hybrid Search        │
    │  BM25 (keyword) + HNSW (dense) + RRF     │
    │  all-MiniLM-L6-v2  •  384 dimensions     │
    └──────────────────┬───────────────────────┘
                       │  Top-K statute sections
    ┌──────────────────▼───────────────────────┐
    │  format_context() → Citation-labelled     │
    │  context string with [BNS Section 103]   │
    └──────────────────┬───────────────────────┘
                       │
    ┌──────────────────▼───────────────────────┐
    │  LLM.invoke(system_prompt + context +    │
    │             user_question)               │
    │  Groq / Gemini / OpenAI                  │
    └──────────────────┬───────────────────────┘
                       │
    ┌──────────────────▼───────────────────────┐
    │  Response: answer + citations + latency  │
    └──────────────────────────────────────────┘
```

---

## 🔐 Security & Compliance Notes

- **No data leaves the server for embeddings** — `all-MiniLM-L6-v2` runs entirely on-device
- **No silent degradation** — missing API key raises an explicit `RuntimeError` with guidance
- **Strict prompt constraints** — LLM is instructed to cite sources and refuse to speculate outside retrieved context
- **Sensitive files are gitignored** — `.env`, `vectorstore/`, `data/raw/`, eval results

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core`, `langchain-community` | RAG orchestration framework |
| `langchain-text-splitters` | Recursive character text chunking |
| `weaviate-client >= 4.10` | Vector database (embedded mode) |
| `sentence-transformers` | Local CPU embedding model |
| `langchain-huggingface` | HuggingFace embedding integration |
| `pypdf` | PDF loading |
| `python-docx` | DOCX loading |
| `streamlit >= 1.30` | Web UI |
| `pydantic-settings` | Typed environment configuration |
| `ragas` | RAG evaluation metrics (Faithfulness, Relevancy) |

Install all:
```bash
pip install -r reqs.txt
# or
uv sync
```

---

## 📄 License

This project is for research and educational purposes on publicly available Indian statutory texts.

---

*⚖️ NyayaSetu — Bridge to Justice*
