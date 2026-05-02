import logging

from fastapi import APIRouter, HTTPException

from app.pipeline.orchestrator import handle_query
from app.schema.chat import ChatRequest, ChatResponse, SourceChunk

router = APIRouter()
logger = logging.getLogger(__name__)


def _normalise_novel(novel: str | None) -> str | None:
    """Convert user-friendly novel names to the directory-slug stored in metadata.

    e.g. 'Pride and Prejudice' -> 'pride_and_prejudice'
         'Frankenstein'        -> 'frankenstein'
    """
    if novel is None:
        return None
    return novel.strip().lower().replace(" ", "_")


@router.post("/query", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.
    Routes the query to the RAG pipeline (novel-related) or the LLM fallback (general).
    Optionally filter retrieval to a specific novel via the 'novel' field
    (accepts friendly names like 'Pride and Prejudice' or slugs like 'pride_and_prejudice').
    """
    try:
        novel_filter = _normalise_novel(request.novel)
        logger.info(
            "Incoming query | query=%r | novel=%r | novel_filter=%r",
            request.query,
            request.novel,
            novel_filter,
        )

        result = handle_query(request.query, novel_filter=novel_filter)

        logger.info(
            "Query handled | route=%s | sources_count=%d",
            result["route"],
            len(result["sources"]),
        )

        return ChatResponse(
            status="success",
            query=request.query,
            route=result["route"],
            answer=result["answer"],
            sources=[SourceChunk(**s) for s in result["sources"]],
        )
    except Exception as e:
        logger.exception("Unhandled error in /query endpoint")
        raise HTTPException(status_code=500, detail=str(e))
