"""The vector database that store our embeddings"""

from functools import lru_cache
from langchain_core.vectorstores import VectorStore
from config.settings import settings
from exceptions import ConfigurationError
from logger import GLOBAL_LOGGER as log
from rag.embeddings import get_embeddings_model

def _local_store() -> VectorStore:
    from langchain_chroma import Chroma
    log.info("Using local chroma vector store ",path=settings.chroma_dir)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings_model(),
        persist_directory=settings.chroma_dir,
    )
def _cloud_store() -> VectorStore:
    missing=settings.missing_cloud_settings()
    if missing:
        raise ConfigurationError(f"APP_MODE=cloud but these settings are empty:{','.join(missing)}")
    from google.cloud import apiplatform
    from langchain_google_vertexai import VectorSearchVectorStore
    log.info("Using Vertex AI vector search ",index_id=settings.vector_search_index_id)
    apiplatform.init(project=settings.GCP_PROJECT,location=settings.GCP_REGION)
    return VectorSearchVectorStore.from_components(
        project_id=settings.GCP_PROJECT,
        region=settings.GCP_REGION,
        embedding=get_embeddings_model,
        index_id=settings.vector_search_index_id,
        endpoint_id=settings.vector_search_index_endpoint_id,
        gcs_bucket_name=settings.GCS_BUCKET_NAME,
        stream_update=True
    )

@lru_cache(maxsize=1)
def get_vector_store()-> VectorStore:
    """connect on first use so the api can start before everything is configured"""
    return _cloud_store() if settings.is_cloud else _local_store()
