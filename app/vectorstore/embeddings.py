from langchain_huggingface import HuggingFaceEmbeddings

# multi-qa-MiniLM-L6-cos-v1 is trained for asymmetric semantic search:
# query (question) → passage (answer).  It significantly outperforms the
# symmetric all-MiniLM-L6-v2 for RAG retrieval use cases.
# cos-v1 means the model is optimised for cosine similarity, so we must also
# set hnsw:space=cosine in ChromaDB and enable embedding normalisation.
_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    get_embeddings()
