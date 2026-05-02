from app.vectorstore.store import get_vectorstore


def get_retriever(novel_filter: str | None = None):
    vectorstore = get_vectorstore()

    search_kwargs: dict = {
        "k": 6,
        "lambda_mult": 0.5,
    }

    if novel_filter:
        search_kwargs["filter"] = {"novel": novel_filter}

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    return retriever


if __name__ == "__main__":
    get_retriever()