import logging

from langsmith import traceable

from app.config.settings import ROUTING_THRESHOLD
from app.core.llm import get_fallback_chain
from app.rag import get_rag_chain
from app.vectorstore.store import get_vectorstore

logger = logging.getLogger(__name__)


@traceable(name="query_routing")
def route_query(
    query: str,
    threshold: float = ROUTING_THRESHOLD,
    novel_filter: str | None = None,
) -> str:
    vectorstore = get_vectorstore()

    # L2 relevance scores: 1 - distance / sqrt(2), range [0, 1] for
    # normalised MiniLM vectors (higher = more relevant).
    search_kwargs: dict = {"k": 1}
    if novel_filter:
        search_kwargs["filter"] = {"novel": novel_filter}

    logger.debug(
        "Routing query | query=%r | novel_filter=%r | threshold=%.3f",
        query,
        novel_filter,
        threshold,
    )

    results = vectorstore.similarity_search_with_relevance_scores(
        query, **search_kwargs
    )

    if not results:
        logger.warning("No documents returned from vectorstore — routing to llm")
        return "llm"

    top_doc, relevance_score = results[0]
    logger.info(
        "Top result | score=%.4f | threshold=%.3f | passes=%s | "
        "novel=%r | chapter=%r | content_preview=%r",
        relevance_score,
        threshold,
        relevance_score >= threshold,
        top_doc.metadata.get("novel"),
        top_doc.metadata.get("chapter_label"),
        top_doc.page_content[:120],
    )

    route = "rag" if relevance_score >= threshold else "llm"
    logger.info("Route decision: %s", route)
    return route


@traceable(name="rag_pipeline")
def run_rag(query: str, novel_filter: str | None = None) -> dict:
    logger.debug("Running RAG pipeline | novel_filter=%r", novel_filter)
    rag_chain = get_rag_chain(novel_filter=novel_filter)
    result = rag_chain.invoke({"query": query})

    logger.info(
        "RAG pipeline complete | docs_retrieved=%d",
        len(result["context_docs"]),
    )
    return {
        "route": "rag",
        "answer": result["answer"],
        "sources": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["context_docs"]
        ],
    }


@traceable(name="fallback_llm")
def run_llm(query: str) -> dict:
    logger.debug("Running LLM fallback")
    fallback_chain = get_fallback_chain()
    answer = fallback_chain.invoke({"query": query})
    return {
        "route": "llm",
        "answer": answer,
        "sources": [],
    }


@traceable(name="full_query_pipeline", tags=["rag", "routing"])
def handle_query(query: str, novel_filter: str | None = None) -> dict:
    route = route_query(query, novel_filter=novel_filter)

    if route == "rag":
        return run_rag(query, novel_filter=novel_filter)
    else:
        return run_llm(query)


if __name__ == "__main__":
    handle_query("Who is Elizabeth Bennet?")