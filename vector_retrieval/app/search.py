from .config import settings
from .database import search_vector
from .ingestion.embeddings import embed_query
from .schemas import (
    SearchResponse,
    SearchResult,
)


def format_qdrant_context(
    results: list[SearchResult],
) -> str:
    if not results:
        return ""

    blocks: list[str] = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        block = (
            f"[{index}] {result.text}\n"
            f"Quelle: {result.source}\n"
            f"Datum: {result.date}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def search_documents(
    query: str,
    limit: int | None = None,
) -> SearchResponse:
    if limit is None:
        limit = settings.vector_top_k

    query_embedding = embed_query(query)

    hits = search_vector(
        vector=query_embedding,
        limit=limit,
    )

    results: list[SearchResult] = []

    for hit in hits:
        payload = hit.payload or {}

        results.append(
            SearchResult(
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                date=payload.get("date", ""),
                classification=payload.get(
                    "classification",
                    "",
                ),
                score=float(hit.score),
                document_id=payload.get(
                    "document_id",
                    "",
                ),
                chunk_index=int(
                    payload.get(
                        "chunk_index",
                        0,
                    )
                ),
            )
        )

    return SearchResponse(
        qdrant_context=format_qdrant_context(
            results
        ),
        results=results,
    )