from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


VECTOR_RETRIEVAL_ROOT = Path(__file__).resolve().parents[1]
KAISER_ROOT = VECTOR_RETRIEVAL_ROOT.parent

DATA_DIR = VECTOR_RETRIEVAL_ROOT / "data"


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "kaiser_documents"

    bge_model: str = "BAAI/bge-m3"
    bge_device: str = "cpu"

    vector_top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=KAISER_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()