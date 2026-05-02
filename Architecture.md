## Steps in RAG development

## 1. Text preprocessing: Remove boilerplate, Normalize text, Attach metadata early such as title, author, source
## 2. Document loading layer
### 2.1 This stage converts raw processed files into structured objects that downstream systems (LangChain, vector DB) can work with. What actually happens
    - Read each `.txt` file from `processed/`
    - Associate it with its metadata (from `metadata.json`)
    - Convert it into a Document abstraction
### 2.2 Why this layer exists
    - Separates: file system concerns, NLP / embedding logic
    - This makes it easy later to: switch from local files → S3, add more novels without changing pipeline
## 3. Chunking
### 3.1 LLMs and embedding models cannot process entire novels efficiently, perform better on semantically coherent chunks
### 3.2 Each chunk becomes: independently searchable, independently embedded
### 3.3 Chunk size: 
    - Typical range: 500–1000 characters (or tokens)
    - Trade-off: Too small → loss of context, Too large → poor retrieval precision
    - For novels: ~800 characters is a good starting point
### Overlap: Chunks should overlap: Chunk 1: [0–800], Chunk 2: [650–1450] to preserves continuity and avoids losing context at boundaries
### Semantic coherence: Naive chunking splits by character count. Better chunking respects:
    - paragraphs, sentence boundaries
    - improves: retrieval accuracy, answer quality
### Metadata propagation: Each chunk must inherit metadata. This enables: filtering (e.g., only one book), traceability (which text generated answer)
## 4. Embedding generation
## 5. Vector storage
## 6. Retriever setup
## 7. Query routing using Embedding similarity threshold or other approach
## 8. RAG chain
## 9. General LLM fallback
## 10. Unified response service
## 11. API layer (FastAPI) - ingestion, chat
## 12. Frontend (Next.js)
## 13. Enhancements
### 13.1 Metadata filtering
### 13.2 Multi-query retrieval
### 13.3 Hybrid search
### 13.4 Streaming responses
### 14 Evaluation
### 15 Deployment - Backend (Docker, Uvicorn), Frontend (Vercel)
### Storage - Persistent volume for Chroma

## Final pipeline summary

```text
Download → Clean → Load → Chunk → Embed → Store → Retrieve → Route → Generate → Serve
```

```test
assets/
   ↓
processed/
   ↓
chapters/
   ↓
chunking (local)
   ↓
embeddings (local/API)
   ↓
Chroma DB (Docker)
```

### Key focus

* Chunking quality > model choice
* Metadata design impacts retrieval accuracy
* Query routing determines system correctness
