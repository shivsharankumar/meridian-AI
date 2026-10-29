"""Text -> vector conversion,used both when indexing documents and when searching.
the same model must be used for indexing and for querying so the model follows APP_MODE:
local -> uses OpenRouter embeddings (requires OPENROUTER_API_KEY in .env)
cloud -> uses the text-embedding-005 Vertex AI embedding model (requires GCP_PROJECT_ID in .env)

both produce 768 dimensional vectors in cloud mode that number is fixed by the vector search index (see github/workflows/deploy.yml)

LangChain uses Google embeddings for cloud mode and the OpenAI-compatible client for OpenRouter local mode.
"""

from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from exceptions import ConfigurationError

@lru_cache(maxsize=1) #least recently used cache, only one instance of the model is created and reused
def get_embeddings_model() -> GoogleGenerativeAIEmbeddings | OpenAIEmbeddings:
    """Create the embeddings model on first use (not at import time) and reuse it afterwards."""
    if settings.is_cloud:
        if not settings.GCP_PROJECT:
            raise ConfigurationError("APP_MODE=cloud needs GCP_PROJECT_ID for Vertex AI embeddings")
        #cloud part
        return GoogleGenerativeAIEmbeddings(
            model=settings.embeddings_model_name,
            vertexai=True,
            project=settings.GCP_PROJECT,
            location=settings.GCP_REGION,
            output_dimensionality=settings.embedding_dimensions,
        )
    if not settings.OPENROUTER_API_KEY:
        raise ConfigurationError(
            "APP_MODE=local needs OPENROUTER_API_KEY in .env for embeddings"
        )
    return OpenAIEmbeddings(
        model=settings.openrouter_embedding_model_name,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.openrouter_base_url,
        dimensions=settings.embedding_dimensions,
    )