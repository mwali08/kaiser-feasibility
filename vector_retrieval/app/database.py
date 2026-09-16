import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .config import settings


VECTOR_SIZE = 1024


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
    )


def ensure_collection() -> None:
    client = get_qdrant_client()

    if client.collection_exists(
        collection_name=settings.qdrant_collection
    ):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def store_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(
            "Anzahl der Chunks und Vektoren stimmt nicht überein."
        )

    client = get_qdrant_client()

    points: list[PointStruct] = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"{chunk['document_id']}:"
                    f"{chunk['chunk_index']}"
                ),
            )
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk,
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )


def search_vector(
    vector: list[float],
    limit: int,
):
    client = get_qdrant_client()

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    return response.points