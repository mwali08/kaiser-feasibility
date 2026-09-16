from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

KAISER_ROOT= Path(__file__).resolve().parents[2]
VECTOR_RETRIEVAL_ROOT= Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    qdrant_url: str= "http://localhost:6333"
    qdrant_collection: str= "kaiser_documents"

    bge_model: str= "BAAI/bge-m3"
    bge_device: str= "cpu"
    vector_top_k: int=3

    model_config= SettingsConfigDict(
        env_file=KAISER_ROOT / ".env",
        extra="ignore",
    )
settings = Settings()
DATA_DIR= VECTOR_RETRIEVAL_ROOT / "data"