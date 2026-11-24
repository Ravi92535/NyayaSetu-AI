from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """
    Central application settings defined using Pydantic Settings.
    Reads values from environment variables or .env file.
    """

    # General App Configuration
    APP_NAME: str = "NyayaSetu AI"
    ENV: str = "development"
    DEBUG: bool = False

    # Storage and File Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = Path("data/raw")
    DATA_PROCESSED_DIR: Path = Path("data/processed")
    VECTORSTORE_DIR: Path = Path("vectorstore")

    # Ingestion & Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval & Ranking
    TOP_K: int = 5
    EMBEDDING_PROVIDER: str = "huggingface"  # "huggingface" or "openai"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    WEAVIATE_COLLECTION: str = "LegalStatute"
    WEAVIATE_HYBRID_ALPHA: float = 0.25  # 0.0 = pure BM25, 1.0 = pure dense vector, 0.25 = optimal hybrid
    WEAVIATE_URL: Optional[str] = None   # None = embedded local instance; or e.g. "http://localhost:8080"
    # LLM & Generation
    LLM_PROVIDER: str = "groq"  # "groq", "gemini", "openai"
    LLM_MODEL: str = "qwen/qwen3.8-27b"
    LLM_TEMPERATURE: float = 0.0

    # API Keys
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Export global settings instance
settings = Settings()

__all__ = ["Settings", "settings"]

