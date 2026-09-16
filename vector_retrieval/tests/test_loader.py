from app.ingestion.loader import load_documents


def test_load_documents():
    documents = load_documents()

    assert len(documents) >= 5

    first_document = documents[0]

    assert "document_id" in first_document
    assert "source" in first_document
    assert "date" in first_document
    assert "classification" in first_document
    assert "text" in first_document