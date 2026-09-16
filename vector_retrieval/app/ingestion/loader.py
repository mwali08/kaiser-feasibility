import json

from ..config import DATA_DIR


REQUIRED_FIELDS = {
    "document_id",
    "title",
    "source",
    "date",
    "classification",
    "text",
}


def load_documents() -> list[dict]:
    documents: list[dict] = []

    for file_path in sorted(DATA_DIR.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        missing_fields = REQUIRED_FIELDS - document.keys()

        if missing_fields:
            raise ValueError(
                f"{file_path.name}: fehlende Felder: "
                f"{sorted(missing_fields)}"
            )

        documents.append(document)

    if not documents:
        raise RuntimeError(
            f"Keine Dokumente in {DATA_DIR} gefunden."
        )

    return documents