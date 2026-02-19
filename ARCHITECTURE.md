# Architecture Overview

## Goal

Provide a brief, readable overview of how your chatbot works: ingestion, indexing, retrieval with citations, and memory writing.

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)

- **Supported inputs:** `.txt`, `.md`, `.pdf`
- **Parsing approach:** Plain text for txt/md; PyPDF for PDF extraction
- **Chunking strategy:** Section-aware splitting at paragraph boundaries, 500 tokens per chunk with 50-token overlap
- **Metadata per chunk:** `source` (filename), `chunk_id`, `locator` (section/heading + chunk id)

### 2) Indexing / Storage

- **Vector store:** ChromaDB with persistent storage
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, no API key)
- **Persistence:** `chroma_db/` directory
- **Optional lexical index (BM25):** Not implemented; dense-only retrieval

### 3) Retrieval + Grounded Answering

- **Retrieval:** Top-k (k=5) cosine similarity search over embeddings
- **Citations:** Each citation includes `source`, `locator`, `snippet` (truncated chunk text)
- **LLM:** OpenAI `gpt-4o-mini` or Ollama (e.g. `llama3.2`) when `USE_OLLAMA=1`; context-grounded prompt; instructed to cite sources in the answer
- **Failure behavior:** If retrieval returns no chunks, respond with: "I couldn't find relevant information in the uploaded documents." No hallucination of sources.

### 4) Memory System (Selective)

- **High-signal memory:** User preferences (e.g., "prefers weekly summaries on Mondays"), roles (e.g., "Project Finance Analyst"), org learnings (e.g., "Asset Management interfaces with Project Finance")
- **Explicitly NOT stored:** Raw transcripts, PII, secrets, low-value chitchat
- **Decision logic:** LLM extracts candidates from each turn; only items with confidence ≥ 0.8 are written; duplicates are skipped
- **Format:** Bullet lists in `USER_MEMORY.md` and `COMPANY_MEMORY.md`

### 5) Optional: Safe Tooling (Open-Meteo)

Not implemented.

---

## Tradeoffs & Next Steps

- **Why this design:** ChromaDB and sentence-transformers allow local embeddings without API costs; Streamlit provides a fast, simple UI. The chat LLM can be local (Ollama) or cloud (OpenAI), controlled by `USE_OLLAMA`.
- **Improvements with more time:** Hybrid retrieval (BM25 + embeddings), reranking, streaming responses, multi-user support, and file management (re-index, delete documents).
