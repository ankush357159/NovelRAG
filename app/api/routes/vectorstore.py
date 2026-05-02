import math
import os
import json
from typing import Optional

import chromadb
from fastapi import APIRouter, HTTPException, Query
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config.settings import CHUNK_DIR
from app.vectorstore.embeddings import get_embeddings
from app.vectorstore.ingest import ingest, load_chunks, reingest
from app.vectorstore.store import drop_collection, get_vectorstore, list_collections
from app.schema.vectorstore import (
    CollectionInfoResponse,
    DiagnosticsResponse,
    DiskChunkSample,
    DropCollectionResponse,
    IngestResponse,
    ReIngestResponse,
    RetrieveResponse,
    RetrieveResult,
    SearchResponse,
    SearchResult,
    StoredDocumentSample,
)

COLLECTION_NAME = "novels_collection"

router = APIRouter()


def _doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('novel', '')}::{meta.get('chapter_index', '')}::{doc.page_content}"


def _load_all_documents_for_bm25(vectorstore, novel: Optional[str]) -> list[Document]:
    where = {"novel": novel} if novel else None
    kwargs: dict = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    raw = vectorstore._collection.get(**kwargs)
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    return [
        Document(page_content=doc, metadata=(meta or {}))
        for doc, meta in zip(documents, metadatas)
    ]


@router.post("/ingest", response_model=IngestResponse)
def ingest_to_vectorstore():
    """Load all chunks from disk and upsert them into the Chroma vector store."""
    try:
        count = ingest()
        return IngestResponse(
            status="success",
            message=f"Ingested chunks into vector store. Collection now contains {count} documents.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reingest", response_model=ReIngestResponse)
def reingest_vectorstore():
    """
    Drop the existing collection and re-ingest all chunks from disk in one step.
    Use this after re-chunking or re-processing novels to replace stale vectors.
    """
    try:
        dropped = drop_collection()
        count = ingest()
        return ReIngestResponse(
            status="success",
            dropped=dropped,
            documents_ingested=count,
            message=f"Collection rebuilt with {count} documents.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection-info", response_model=CollectionInfoResponse)
def collection_info():
    """Return collection name and total document count from Chroma."""
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        return CollectionInfoResponse(
            status="success",
            collection_name=COLLECTION_NAME,
            document_count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics():
    """
    Inspect the state of ChromaDB and the chunk files on disk.

    Returns:
    - all_collections_in_chroma: every collection currently in the ChromaDB server
    - chroma_document_count: number of documents stored in novels_collection (0 if absent)
    - stored_document_samples: first 5 documents actually stored in ChromaDB
    - chunk_files_on_disk: total number of JSON chunk files found under CHUNK_DIR
    - disk_chunk_samples: content preview of the first 5 chunk files on disk

    Use this to verify that delete and ingest operations are working correctly.
    """
    try:
        import chromadb
        client = chromadb.HttpClient(host="localhost", port=9001)

        all_col_names = [c.name for c in client.list_collections()]
        collection_exists = COLLECTION_NAME in all_col_names

        # ----- ChromaDB side -----
        chroma_count = 0
        stored_samples: list[StoredDocumentSample] = []

        if collection_exists:
            col = client.get_collection(COLLECTION_NAME)
            chroma_count = col.count()

            if chroma_count > 0:
                raw = col.get(limit=5, include=["documents", "metadatas"])
                for doc_id, doc, meta in zip(
                    raw["ids"], raw["documents"], raw["metadatas"]
                ):
                    stored_samples.append(
                        StoredDocumentSample(
                            id=doc_id,
                            novel=meta.get("novel", ""),
                            chapter_label=meta.get("chapter_label", ""),
                            content_preview=doc[:120],
                        )
                    )

        # ----- Disk side -----
        disk_count = 0
        disk_samples: list[DiskChunkSample] = []

        for novel_dir in sorted(os.listdir(CHUNK_DIR)):
            novel_path = os.path.join(CHUNK_DIR, novel_dir)
            if not os.path.isdir(novel_path):
                continue
            for fname in sorted(os.listdir(novel_path)):
                if not fname.endswith(".json"):
                    continue
                disk_count += 1
                if len(disk_samples) < 5:
                    fpath = os.path.join(novel_path, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    disk_samples.append(
                        DiskChunkSample(
                            file=f"{novel_dir}/{fname}",
                            novel=data["metadata"].get("novel", ""),
                            chapter_label=data["metadata"].get("chapter_label", ""),
                            content_preview=data["page_content"][:120],
                        )
                    )

        return DiagnosticsResponse(
            status="success",
            all_collections_in_chroma=all_col_names,
            novels_collection_exists=collection_exists,
            chroma_document_count=chroma_count,
            chunk_files_on_disk=disk_count,
            stored_document_samples=stored_samples,
            disk_chunk_samples=disk_samples,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection", response_model=DropCollectionResponse)
def delete_collection():
    """
    Drop the Chroma collection entirely (idempotent — safe to call even if already absent).
    The collection is recreated with the correct L2 metric on the next ingest.
    """
    try:
        dropped = drop_collection()
        msg = (
            f"Collection '{COLLECTION_NAME}' dropped. Re-run POST /reingest to rebuild."
            if dropped
            else f"Collection '{COLLECTION_NAME}' did not exist — nothing to drop."
        )
        return DropCollectionResponse(status="success", message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse)
def similarity_search(
    query: str = Query(..., description="Query string to search for"),
    k: int = Query(6, ge=1, le=50, description="Number of results to return"),
    novel: Optional[str] = Query(None, description="Filter by novel slug, e.g. 'frankenstein'"),
):
    """
    Similarity search using LangChain's similarity_search_with_relevance_scores.
    Returns normalised relevance scores in [0, 1] (higher = more relevant), content and metadata.
    Optionally filter to a specific novel with the 'novel' parameter.
    """
    try:
        vectorstore = get_vectorstore()
        novel_filter = {"novel": novel} if novel else None

        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=novel_filter
        )

        results = [
            SearchResult(
                relevance_score=round(score, 6),
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc, score in scored_docs
        ]

        return SearchResponse(
            status="success",
            query=query,
            total_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid-search", response_model=SearchResponse)
def hybrid_search(
    query: str = Query(..., description="Query string to search for"),
    k: int = Query(6, ge=1, le=50, description="Number of results to return"),
    fetch_k: int = Query(50, ge=5, le=500, description="Candidate pool size from each retriever"),
    rrf_k: int = Query(60, ge=1, le=200, description="RRF smoothing constant (higher reduces rank dominance)"),
    novel: Optional[str] = Query(None, description="Filter by novel slug, e.g. 'pride_and_prejudice'"),
):
    """
    Hybrid retrieval: dense vector similarity + BM25 lexical retrieval,
    combined with Reciprocal Rank Fusion (RRF).

    This is more robust for long analytical questions that include specific
    textual cues (character names, scene actions) and abstract framing.
    """
    try:
        vectorstore = get_vectorstore()
        novel_filter = {"novel": novel} if novel else None

        # Dense semantic retrieval via LangChain + Chroma
        semantic_scored = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=novel_filter
        )
        semantic_docs = [doc for doc, _ in semantic_scored]
        semantic_rank: dict[str, int] = {
            _doc_key(doc): i + 1 for i, doc in enumerate(semantic_docs)
        }
        semantic_score: dict[str, float] = {
            _doc_key(doc): score for doc, score in semantic_scored
        }

        # Lexical retrieval via BM25 over stored chunk documents
        bm25_corpus = _load_all_documents_for_bm25(vectorstore, novel)
        if not bm25_corpus:
            return SearchResponse(status="success", query=query, total_results=0, results=[])

        bm25 = BM25Retriever.from_documents(bm25_corpus)
        bm25.k = min(fetch_k, len(bm25_corpus))
        lexical_docs = bm25.invoke(query)
        lexical_rank: dict[str, int] = {
            _doc_key(doc): i + 1 for i, doc in enumerate(lexical_docs)
        }

        # RRF merge of semantic + lexical rankings.
        doc_lookup: dict[str, Document] = {
            _doc_key(doc): doc for doc in semantic_docs
        }
        for doc in lexical_docs:
            doc_lookup.setdefault(_doc_key(doc), doc)

        fused: list[tuple[float, Document]] = []
        for key, doc in doc_lookup.items():
            score = 0.0
            sr = semantic_rank.get(key)
            lr = lexical_rank.get(key)
            if sr is not None:
                score += 1.0 / (rrf_k + sr)
            if lr is not None:
                score += 1.0 / (rrf_k + lr)

            # Small semantic tie-breaker for same RRF score buckets.
            score += 0.05 * max(0.0, semantic_score.get(key, 0.0))
            fused.append((score, doc))

        fused.sort(key=lambda x: x[0], reverse=True)
        top = fused[:k]

        results = [
            SearchResult(
                relevance_score=round(score, 6),
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for score, doc in top
        ]

        return SearchResponse(
            status="success",
            query=query,
            total_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve", response_model=RetrieveResponse)
def retrieve(
    query: str = Query(..., description="Query string for MMR retrieval"),
    k: int = Query(6, ge=1, le=50, description="Number of results to return"),
    fetch_k: int = Query(20, ge=1, le=200, description="Candidate pool size before MMR re-ranking"),
    lambda_mult: float = Query(
        0.5, ge=0.0, le=1.0,
        description="MMR diversity factor (0=max diversity, 1=max relevance)",
    ),
    novel: Optional[str] = Query(None, description="Filter by novel slug, e.g. 'frankenstein'"),
):
    """
    MMR retrieval with relevance scores.
    Uses max-marginal relevance to balance relevance and diversity.
    Optionally filter to a specific novel with the 'novel' parameter.
    """
    try:
        vectorstore = get_vectorstore()
        novel_filter = {"novel": novel} if novel else None

        # Collect relevance scores from a plain similarity pass over the candidate pool.
        # Full page_content is used as the key to avoid collisions on short prefixes.
        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=novel_filter
        )
        score_map: dict[str, float] = {
            doc.page_content: round(score, 6)
            for doc, score in scored_docs
        }

        # Re-rank the pool with MMR for diversity
        mmr_docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=novel_filter,
        )

        results = [
            RetrieveResult(
                relevance_score=score_map.get(doc.page_content, 0.0),
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in mmr_docs
        ]

        return RetrieveResponse(
            status="success",
            query=query,
            total_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug-search")
def debug_search(
    query: str = Query(..., description="Query to debug"),
    k: int = Query(50, ge=1, le=1000, description="How many top results to inspect"),
    keyword: Optional[str] = Query(
        None,
        description="Keyword to scan ALL stored documents for (e.g. 'universally acknowledged'). "
                    "Reports the rank and distance of the first matching document.",
    ),
):
    """
    Raw diagnostic search that bypasses all LangChain wrappers.

    Useful for debugging why the correct chunk is not appearing in search results.
    Reports:
    - Collection state (total documents, embedding dimension)
    - Query embedding norm (should be ~1.0 for normalised MiniLM vectors)
    - Top-k results with raw cosine distance and similarity score
    - If 'keyword' is provided: scans ALL stored documents and reports the
      rank and distance of the first document whose content contains that keyword.
      If the rank is > k, it means it is outside the normal search window.
    """
    try:
        client = chromadb.HttpClient(host="localhost", port=9001)
        all_col_names = [c.name for c in client.list_collections()]

        if COLLECTION_NAME not in all_col_names:
            raise HTTPException(status_code=404, detail=f"Collection '{COLLECTION_NAME}' not found in ChromaDB.")

        col = client.get_collection(COLLECTION_NAME)
        total = col.count()

        if total == 0:
            return {
                "status": "error",
                "detail": "Collection exists but contains 0 documents. Run POST /reingest first.",
                "total_in_collection": 0,
            }

        # Embed the query using the same model used at ingest time
        embedding_fn = get_embeddings()
        query_vector: list[float] = embedding_fn.embed_query(query)
        query_norm = round(math.sqrt(sum(v * v for v in query_vector)), 6)

        # Top-k raw query
        n = min(k, total)
        raw = col.query(
            query_embeddings=[query_vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        ids: list[str] = raw["ids"][0]
        distances: list[float] = raw["distances"][0]
        documents: list[str] = raw["documents"][0]
        metadatas: list[dict] = raw["metadatas"][0]

        top_results = [
            {
                "rank": rank,
                "distance": round(dist, 6),
                "relevance_score": round(1.0 - dist, 6),
                "novel": meta.get("novel"),
                "chapter_label": meta.get("chapter_label"),
                "content_preview": doc[:120],
            }
            for rank, (dist, doc, meta) in enumerate(
                zip(distances, documents, metadatas), start=1
            )
        ]

        # Keyword scan: retrieve all docs and find where the keyword appears
        keyword_hit: dict | None = None
        if keyword:
            # Fetch all documents (no embeddings needed)
            batch_size = 500
            offset = 0
            all_ids_kw: list[str] = []
            all_docs_kw: list[str] = []
            all_meta_kw: list[dict] = []

            while offset < total:
                batch = col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                all_ids_kw.extend(batch["ids"])
                all_docs_kw.extend(batch["documents"])
                all_meta_kw.extend(batch["metadatas"])
                offset += batch_size

            # Find the first doc matching the keyword
            matched_id: str | None = None
            matched_content: str | None = None
            matched_meta: dict | None = None
            for doc_id, doc, meta in zip(all_ids_kw, all_docs_kw, all_meta_kw):
                if keyword.lower() in doc.lower():
                    matched_id = doc_id
                    matched_content = doc
                    matched_meta = meta
                    break

            if matched_id is None:
                keyword_hit = {
                    "found": False,
                    "detail": f"No stored document contains the keyword '{keyword}'.",
                }
            else:
                # Find its rank in the top-k result IDs
                rank_in_top_k = next(
                    (i + 1 for i, doc_id in enumerate(ids) if doc_id == matched_id),
                    None,
                )

                # Get its actual distance by querying just for this ID
                hit_raw = col.query(
                    query_embeddings=[query_vector],
                    n_results=total,
                    where={"novel": matched_meta.get("novel")},
                    include=["distances"],
                )
                hit_all_ids = hit_raw["ids"][0]
                hit_all_distances = hit_raw["distances"][0]
                actual_rank = next(
                    (i + 1 for i, doc_id in enumerate(hit_all_ids) if doc_id == matched_id),
                    None,
                )
                actual_dist = next(
                    (d for doc_id, d in zip(hit_all_ids, hit_all_distances) if doc_id == matched_id),
                    None,
                )

                keyword_hit = {
                    "found": True,
                    "keyword": keyword,
                    "doc_id": matched_id,
                    "novel": matched_meta.get("novel"),
                    "chapter_label": matched_meta.get("chapter_label"),
                    "content_preview": matched_content[:200],
                    "rank_in_top_k_results": rank_in_top_k,
                    "rank_within_same_novel": actual_rank,
                    "distance": round(actual_dist, 6) if actual_dist is not None else None,
                    "relevance_score": round(1.0 - actual_dist, 6) if actual_dist is not None else None,
                }

        return {
            "status": "success",
            "query": query,
            "total_in_collection": total,
            "query_embedding_norm": query_norm,
            "note": "Norm should be ~1.0 for normalised MiniLM vectors. "
                    "A very different value indicates an embedding problem.",
            "top_results": top_results,
            "keyword_hit": keyword_hit,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
