import os
import json
import hashlib
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import PROCESSED_DIR, CHUNK_DIR


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_dir_hash(directory: str) -> str:
    """
    Compute a single SHA-256 digest that covers every JSON file in *directory*
    (sorted by name for determinism).  Used to detect when chapter files change.
    """
    h = hashlib.sha256()
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        h.update(fname.encode())
        with open(fpath, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading / document creation
# ---------------------------------------------------------------------------

def _load_chapters_for_novel(chapters_dir: str, novel_dir: str) -> list[dict]:
    chapters: list[dict] = []
    for fname in sorted(os.listdir(chapters_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(chapters_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
            data["novel_dir"] = novel_dir
            chapters.append(data)
    return chapters


def load_chapter_files() -> list[dict]:
    """Load all chapter JSON files from every novel under PROCESSED_DIR."""
    chapters: list[dict] = []

    for novel_dir in os.listdir(PROCESSED_DIR):
        novel_path = os.path.join(PROCESSED_DIR, novel_dir)

        if not os.path.isdir(novel_path):
            continue

        chapters_dir = os.path.join(novel_path, "chapters")
        if not os.path.exists(chapters_dir):
            continue

        chapters.extend(_load_chapters_for_novel(chapters_dir, novel_dir))

    return chapters


def create_documents(chapters: list[dict]) -> list[Document]:
    documents: list[Document] = []

    for chapter in chapters:
        doc = Document(
            page_content=chapter["page_content"],
            metadata={
                "title": chapter["title"],
                "author": chapter["author"],
                "chapter_index": chapter["chapter_index"],
                "chapter_label": chapter["chapter_label"],
                "novel": chapter["novel_dir"],
            },
        )
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_novel(novel_dir: str, chapters_dir: str) -> list[str]:
    """Chunk all chapters for one novel and write to CHUNK_DIR/<novel_dir>/."""
    chapters = _load_chapters_for_novel(chapters_dir, novel_dir)
    documents = create_documents(chapters)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    novel_chunk_dir = os.path.join(CHUNK_DIR, novel_dir)
    os.makedirs(novel_chunk_dir, exist_ok=True)

    saved_files: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        file_path = os.path.join(novel_chunk_dir, f"chunk_{idx}.json")
        data = {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        saved_files.append(file_path)

    return saved_files


def chunk_and_store(documents: list[Document]) -> list[str]:
    """
    Chunk *documents* and write JSON files to CHUNK_DIR.
    Kept for backward-compatibility; prefer ``prepare_chunks`` for new code.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    saved_files: list[str] = []

    os.makedirs(CHUNK_DIR, exist_ok=True)

    for idx, chunk in enumerate(chunks, start=1):
        novel = chunk.metadata.get("novel", "unknown")
        novel_dir = os.path.join(CHUNK_DIR, novel)
        os.makedirs(novel_dir, exist_ok=True)

        file_path = os.path.join(novel_dir, f"chunk_{idx}.json")
        data = {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        saved_files.append(file_path)

    return saved_files


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_chunks() -> list[str]:
    """
    For each novel, compare the SHA-256 hash of its chapter files against the
    previously stored hash.  Re-chunk only when chapters have changed.
    """
    os.makedirs(CHUNK_DIR, exist_ok=True)
    all_saved_files: list[str] = []

    for novel_dir in os.listdir(PROCESSED_DIR):
        novel_path = os.path.join(PROCESSED_DIR, novel_dir)

        if not os.path.isdir(novel_path):
            continue

        chapters_dir = os.path.join(novel_path, "chapters")
        if not os.path.exists(chapters_dir):
            continue

        novel_chunk_dir = os.path.join(CHUNK_DIR, novel_dir)
        hash_file = os.path.join(novel_chunk_dir, ".chapters_hash")

        # ---- Change detection ------------------------------------------------
        current_hash = compute_dir_hash(chapters_dir)
        stored_hash: str | None = None
        if os.path.exists(hash_file):
            with open(hash_file) as fh:
                stored_hash = fh.read().strip()

        if stored_hash == current_hash and os.path.exists(novel_chunk_dir):
            print(f"Skipping chunks (chapters unchanged): {novel_dir}")
            existing = [
                os.path.join(novel_chunk_dir, f)
                for f in os.listdir(novel_chunk_dir)
                if f.endswith(".json")
            ]
            all_saved_files.extend(existing)
            continue

        # ---- (Re-)chunk ------------------------------------------------------
        print(f"Re-chunking: {novel_dir}")

        # Wipe stale chunk files before regenerating
        if os.path.exists(novel_chunk_dir):
            shutil.rmtree(novel_chunk_dir)

        saved_files = _chunk_novel(novel_dir, chapters_dir)

        # Persist hash so the next unchanged run is skipped
        with open(hash_file, "w") as fh:
            fh.write(current_hash)

        all_saved_files.extend(saved_files)

    print(f"Chapters loaded: {len(load_chapter_files())}")
    print(f"Total chunks: {len(all_saved_files)}")
    return all_saved_files


if __name__ == "__main__":
    prepare_chunks()