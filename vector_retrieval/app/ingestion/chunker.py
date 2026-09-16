from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def chunk_documents(
    documents: list[dict],
) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        text_chunks = text_splitter.split_text(
            document["text"]
        )

        for chunk_index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "date": document["date"],
                    "classification": document[
                        "classification"
                    ],
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

    return chunks