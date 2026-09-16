from fastapi import FastAPI

from .schemas import (
    SearchRequest,
    SearchResponse,
)
from .search import search_documents


app = FastAPI(
    title="kAIser Vector Retrieval",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "vector_retrieval",
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
) -> SearchResponse:
    return search_documents(
        query=request.query,
        limit=request.limit,
    )