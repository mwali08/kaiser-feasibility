from app.schemas import SearchResult
from app.search import format_qdrant_context


def test_format_qdrant_context():
    results = [
        SearchResult(
            text="Einheit X untersteht Brigade Süd.",
            source="Testquelle",
            date="2026-09-01",
            classification="OFFEN",
            score=0.9,
            document_id="doc-001",
            chunk_index=0,
        ),
        SearchResult(
            text="Brigade Süd führt Einheit X.",
            source="Testquelle 2",
            date="2026-09-02",
            classification="OFFEN",
            score=0.8,
            document_id="doc-002",
            chunk_index=0,
        ),
    ]

    context = format_qdrant_context(
        results
    )

    assert "[1]" in context
    assert "[2]" in context

    assert "Quelle: Testquelle" in context
    assert "Datum: 2026-09-01" in context

    assert "\n\n" in context