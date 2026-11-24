from typing import List, Optional

from langchain_core.embeddings import Embeddings

from ..config import settings
from ..utils import logger


from typing import List, Optional

from langchain_core.embeddings import Embeddings

from ..config import settings
from ..utils import logger


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Factory resolving LangChain-compatible Embeddings based on configuration.
    """
    selected_provider = (provider or settings.EMBEDDING_PROVIDER).lower()
    selected_model = model_name or settings.EMBEDDING_MODEL

    logger.info(f"Initializing embedding provider '{selected_provider}' with model '{selected_model}'")

    if selected_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=selected_model)
        except Exception as e:
            logger.error(f"Failed to load HuggingFaceEmbeddings ({e}).")
            raise

    if selected_provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbeddings ({e}).")
            raise

    raise ValueError(f"Unsupported embedding provider: '{selected_provider}'. Choose 'huggingface' or 'openai'.")


__all__ = ["get_embedding_model"]

