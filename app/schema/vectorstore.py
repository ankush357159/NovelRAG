from pydantic import BaseModel, Field


class StoredDocumentSample(BaseModel):
    id: str
    novel: str
    chapter_label: str
    content_preview: str = Field(description="First 120 characters of stored content")


class DiskChunkSample(BaseModel):
    file: str
    novel: str
    chapter_label: str
    content_preview: str = Field(description="First 120 characters of chunk content on disk")


class DiagnosticsResponse(BaseModel):
    status: str
    all_collections_in_chroma: list[str]
    novels_collection_exists: bool
    chroma_document_count: int
    chunk_files_on_disk: int
    stored_document_samples: list[StoredDocumentSample]
    disk_chunk_samples: list[DiskChunkSample]


class ReIngestResponse(BaseModel):
    status: str
    dropped: bool
    documents_ingested: int
    message: str


class ChunkMetadata(BaseModel):
    title: str
    author: str
    chapter_index: int
    chapter_label: str
    novel: str


class SearchResult(BaseModel):
    relevance_score: float = Field(description="Normalised relevance score from LangChain (higher = more relevant, range [0, 1])")
    page_content: str
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    status: str
    query: str
    total_results: int
    results: list[SearchResult]


class RetrieveResult(BaseModel):
    relevance_score: float = Field(description="Relevance score returned by LangChain (higher = better)")
    page_content: str
    metadata: ChunkMetadata


class RetrieveResponse(BaseModel):
    status: str
    query: str
    total_results: int
    results: list[RetrieveResult]


class CollectionInfoResponse(BaseModel):
    status: str
    collection_name: str
    document_count: int


class IngestResponse(BaseModel):
    status: str
    message: str


class DropCollectionResponse(BaseModel):
    status: str
    message: str
