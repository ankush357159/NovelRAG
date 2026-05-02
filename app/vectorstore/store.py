import chromadb
from langchain_chroma import Chroma
from app.vectorstore.embeddings import get_embeddings

# Cosine similarity is the native metric for sentence-transformer models.
# ChromaDB stores the cosine *distance* (= 1 - cosine_similarity, range [0, 2]).
# LangChain's built-in relevance function for cosine collections returns
# cosine_similarity directly (= 1 - distance), so scores are in [0, 1]
# (higher = more relevant) with no manual formula needed.
# NOTE: changing this metric requires dropping and re-ingesting the collection.
_COLLECTION_NAME = "novels_collection"
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host="localhost", port=9001)


def get_vectorstore():
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        client=_get_client(),
        collection_metadata=_COLLECTION_METADATA,
    )

    return vectorstore


def list_collections() -> list[str]:
    """Return the names of every collection currently in ChromaDB."""
    client = _get_client()
    return [c.name for c in client.list_collections()]


def drop_collection() -> bool:
    """
    Delete the novels collection from ChromaDB.

    Returns True if the collection was deleted, False if it did not exist
    (idempotent — safe to call multiple times in a row).
    """
    client = _get_client()
    existing = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME not in existing:
        return False
    client.delete_collection(_COLLECTION_NAME)
    return True


if __name__ == "__main__":
    get_vectorstore()