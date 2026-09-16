from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

from ..config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> BGEM3FlagModel:
    return BGEM3FlagModel(
        settings.bge_model,
        use_fp16=False,
        devices=settings.bge_device,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()

    output = model.encode(
        texts,
        batch_size=4,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    return output["dense_vecs"].tolist()


def embed_query(query: str) -> list[float]:
    vectors = embed_texts([query])

    return vectors[0]