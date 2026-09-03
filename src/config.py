from pydantic_settings import BaseSettings , SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
        protected_namespaces=()
    )

    #gemini settings
    gemini_api_key: str
    generation_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = 3072

    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    collection_name: str = "rag_docs"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

