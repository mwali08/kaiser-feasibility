from ..database import (
    ensure_collection,
    store_chunks,
)
from .chunker import chunk_documents
from .embeddings import embed_texts
from .loader import load_documents


def run_pipeline() -> None:
    print("1. Dokumente laden ...")

    documents = load_documents()

    print(
        f"   {len(documents)} Dokumente geladen."
    )

    print("2. Dokumente chunken ...")

    chunks = chunk_documents(documents)

    print(
        f"   {len(chunks)} Chunks erzeugt."
    )

    print("3. BGE-M3 Embeddings erzeugen ...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = embed_texts(texts)

    print(
        f"   {len(vectors)} Embeddings erzeugt."
    )

    if vectors:
        print(
            f"   Vektordimension: {len(vectors[0])}"
        )

    print("4. Qdrant Collection vorbereiten ...")

    ensure_collection()

    print("5. Chunks in Qdrant speichern ...")

    store_chunks(
        chunks=chunks,
        vectors=vectors,
    )

    print("Ingestion erfolgreich abgeschlossen.")


if __name__ == "__main__":
    run_pipeline()