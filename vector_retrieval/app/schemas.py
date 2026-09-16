from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
    )

    limit: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class SearchResult(BaseModel):
    text: str
    source: str
    date: str
    classification: str

    score: float

    document_id: str
    chunk_index: int


class SearchResponse(BaseModel):
    qdrant_context: str
    results: list[SearchResult]