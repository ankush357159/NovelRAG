from typing import Optional

from pydantic import BaseModel

from app.schema.vectorstore import ChunkMetadata


class SourceChunk(BaseModel):
    page_content: str
    metadata: ChunkMetadata


class ChatRequest(BaseModel):
    query: str
    novel: Optional[str] = None  # e.g. "frankenstein" or "pride_and_prejudice"


class ChatResponse(BaseModel):
    status: str
    query: str
    route: str  # "rag" | "llm"
    answer: str
    sources: list[SourceChunk]
