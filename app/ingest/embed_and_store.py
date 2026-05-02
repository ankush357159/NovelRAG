import warnings

from app.vectorstore.ingest import reingest


def embed_and_store() -> int:
    """
    Deprecated compatibility wrapper.

    This function now forwards to the canonical vectorstore pipeline to keep
    embedding model and distance metric consistent across the project.
    """
    warnings.warn(
        "app.ingest.embed_and_store.embed_and_store is deprecated. "
        "Use app.vectorstore.ingest.reingest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reingest()


if __name__ == "__main__":
    embed_and_store()