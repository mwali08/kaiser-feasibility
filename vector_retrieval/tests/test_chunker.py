from app.ingestion.chunker import chunk_documents


def test_chunk_documents():
    documents = [
        {
            "document_id": "test-001",
            "title": "Test",
            "source": "Testquelle",
            "date": "2026-09-01",
            "classification": "OFFEN",
            "text": "Ein kurzer Testtext über Einheit X und Brigade Süd.",
        }
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) >= 1

    chunk = chunks[0]

    assert chunk["document_id"] == "test-001"
    assert chunk["source"] == "Testquelle"
    assert chunk["classification"] == "OFFEN"
    assert chunk["chunk_index"] == 0