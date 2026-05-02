# Novel-RAG Backend

Retrieval-Augmented Generation (RAG) backend for classic novels using FastAPI, ChromaDB, LangChain, and OpenAI.

## Features
- End-to-end ingestion pipeline: download, process, chunk, embed, store.
- Retrieval with MMR and optional hybrid retrieval (dense + BM25 + RRF).
- Query routing between RAG mode and fallback LLM mode.
- Source attribution in responses (novel/chapter metadata).
- Docker Compose support for local ChromaDB.

## Tech Stack
- Python 3.12+
- FastAPI
- ChromaDB (HTTP server)
- LangChain + LangChain-Chroma
- HuggingFace sentence-transformer embeddings
- OpenAI chat model

## Canonical Embedding Pipeline
This project now uses a single canonical embedding pipeline:
- Model: `multi-qa-MiniLM-L6-cos-v1`
- Vector metric: `cosine`
- Entry point: `app.vectorstore.ingest.reingest`

The legacy `app.ingest.embed_and_store` module is retained only as a deprecated compatibility wrapper and forwards to the canonical pipeline.

## Project Structure
```text
backend/
├── app/
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── ingest.py
│   │       └── vectorstore.py
│   ├── config/
│   ├── core/
│   ├── ingest/
│   ├── pipeline/
│   ├── rag/
│   ├── schema/
│   ├── scripts/
│   └── vectorstore/
├── assets/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Setup
1. Create and activate your environment.
2. Install dependencies.
3. Start ChromaDB.
4. Configure environment variables.
5. Run the API.

### 1) Create environment and install
```powershell
uv sync
```

### 2) Start ChromaDB
```powershell
docker compose up -d
```

### 3) Configure environment
Copy `.env.example` to `.env` and set required values:
- `OPENAI_API_KEY` (required)

### 4) Run API
```powershell
uv run uvicorn app.main:app --reload
```

## API Endpoints
Base routes are mounted in `app/api/router.py`.

### Ingestion
- `POST /ingest/v1/download`
- `POST /ingest/v1/process`
- `POST /ingest/v1/make-chunks`
- `POST /ingest/v1/embed-store` (canonical rebuild via `reingest`)

### Chat
- `POST /chat/v1/query`

### Vectorstore Utilities
- `POST /vectorstore/v1/ingest`
- `POST /vectorstore/v1/reingest`
- `GET /vectorstore/v1/search`
- `GET /vectorstore/v1/hybrid-search`
- `GET /vectorstore/v1/retrieve`
- `GET /vectorstore/v1/collection-info`
- `GET /vectorstore/v1/diagnostics`
- `DELETE /vectorstore/v1/collection`

## Recommended Ingestion Flow
Run in order:
1. `POST /ingest/v1/download`
2. `POST /ingest/v1/process`
3. `POST /ingest/v1/make-chunks`
4. `POST /ingest/v1/embed-store`

## Security Notes for Public Repositories
- Never commit `.env` or secrets.
- Rotate any key that was ever committed.
- Use `.env.example` with placeholders only.
- Keep tracing/telemetry disabled by default unless intentionally enabled.

## Troubleshooting
- If retrieval quality drops after chunking/model changes, run `POST /vectorstore/v1/reingest`.
- If Chroma appears stale, recreate collection using `DELETE /vectorstore/v1/collection` then reingest.
- If API startup fails, confirm required env vars are set.

## License and Data
Source texts are pulled from Project Gutenberg and are expected to be public-domain works.