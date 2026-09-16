from FlagEmbedding import BGEM3FlagModel


def main() -> None:
    print("Lade BGE-M3 ...")

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
        devices="cpu",
    )

    texts = [
        "Die Einheit X ist der Brigade Süd unterstellt.",
        "Der Standort Alpha befindet sich in Wien.",
    ]

    print("Erzeuge Embeddings ...")

    output = model.encode(
        texts,
        batch_size=2,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vectors = output["dense_vecs"]

    print("Anzahl:", len(vectors))
    print("Dimension:", len(vectors[0]))

    assert len(vectors[0]) == 1024

    print("BGE-M3 funktioniert.")


if __name__ == "__main__":
    main()