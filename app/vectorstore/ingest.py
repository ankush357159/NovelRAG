import os
import json
import hashlib

from langchain_core.documents import Document
from app.config.settings import CHUNK_DIR
from app.vectorstore.store import drop_collection, get_vectorstore


def load_chunks() -> list[dict]:
    chunks = []

    for novel_dir in sorted(os.listdir(CHUNK_DIR)):
        novel_path = os.path.join(CHUNK_DIR, novel_dir)

        if not os.path.isdir(novel_path):
            continue

        for file in sorted(os.listdir(novel_path)):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(novel_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks.append(data)

    return chunks


def _chunk_id(page_content: str, metadata: dict) -> str:
    """
    Deterministic ID derived from novel, chapter index, and content.
    Allows ChromaDB to upsert (rather than blindly append) when
    re-ingesting unchanged chunks.
    """
    key = f"{metadata.get('novel', '')}::{metadata.get('chapter_index', '')}::{page_content}"
    return hashlib.md5(key.encode()).hexdigest()


def _clean_content(text: str) -> str:
    """
    Collapse the double newlines that the text normalizer inserts between every
    sentence into a single space.  The chapter heading on the first line is also
    stripped because it adds structural noise to the embedding.

    The original text is preserved in the on-disk JSON; only the embedded
    representation is cleaned here.
    """
    import re
    # Remove a leading chapter heading line (e.g. "CHAPTER I.\n")
    text = re.sub(r"^CHAPTER\s+[IVXLCDM\d]+\.?\s*\n", "", text, flags=re.IGNORECASE)
    # Replace double (or more) newlines with a single space
    text = re.sub(r"\n{2,}", " ", text)
    # Replace remaining single newlines with a space
    text = text.replace("\n", " ")
    # Collapse any resulting multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def build_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=_clean_content(chunk["page_content"]),
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]


def ingest() -> int:
    """
    Load chunks from disk and upsert them into the vector store.

    Uses deterministic, content-derived IDs so calling this on an empty
    collection is safe.  If the collection already contains documents with
    different content (e.g. after re-chunking), call reingest() instead,
    which drops the collection first.

    Returns the number of documents ingested.
    """
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from disk")

    if chunks:
        first = chunks[0]
        print(
            f"  First chunk — novel: {first['metadata'].get('novel')!r}, "
            f"chapter: {first['metadata'].get('chapter_label')!r}, "
            f"content: {first['page_content'][:80]!r}"
        )

    docs = build_documents(chunks)
    ids = [_chunk_id(c["page_content"], c["metadata"]) for c in chunks]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=docs, ids=ids)

    count = vectorstore._collection.count()
    print(f"Collection now contains {count} documents")
    return count


def reingest() -> int:
    """
    Atomically drop the existing collection and re-ingest all chunks from disk.

    Use this whenever chunks have been regenerated (e.g. after sanitization
    or chunking code changes) to avoid stale vectors accumulating alongside
    new ones.

    Returns the number of documents ingested.
    """
    dropped = drop_collection()
    print(f"Collection dropped: {dropped}")
    return ingest()


if __name__ == "__main__":
    reingest()