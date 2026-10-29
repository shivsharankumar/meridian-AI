import os
from pathlib import Path
from typing import Literal

from pydantic import Field,model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    # --App--
    app_name: str = "meridian-AI -Procurement AI Assistant"
    app_env:str = Field("local", validation_alias="Environment")
    app_version:str="1.1.0"

    # --- The toggle ---
    app_mode:Literal["local","cloud"]=Field("local", validation_alias="APP_MODE")

    # --Provider API keys--
    GOOGLE_API_KEY:str=Field("", validation_alias="GOOGLE_API_KEY")
    OPENROUTER_API_KEY:str=Field("", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url:str=Field(
        "https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    openrouter_model_name:str=Field(
        "google/gemini-3.8-flash",
        validation_alias="OPENROUTER_MODEL",
    )
    openrouter_embedding_model_name:str=Field(
        "openai/text-embedding-3-small",
        validation_alias="OPENROUTER_EMBEDDING_MODEL",
    )
    llm_model_name:str=Field("gemini-3.8-flash", validation_alias="VERTEX_LLM_MODEL_NAME")
    llm_temperature:float=Field(0.0, validation_alias="LLM_TEMPERATURE")
    # --Embedding model--
    embedding_model_name:str=Field("text-embedding-005", validation_alias="VERTEX_EMBEDDING_MODEL_NAME")
    local_embedding_model_name:str=Field("gemini-embedding-001", validation_alias="LOCAL_EMBEDDING_MODEL_NAME")
    embedding_dimensions:int=768
    #--local mode storage--
    local_data_dir:Path=Field(PROJECT_ROOT/".meridian", validation_alias="LOCAL_DATA_DIR")
    chroma_collection:str=Field("meridian_docs", validation_alias="CHROMA_COLLECTION")

    # --cloud mode(GCP)--
    GCP_PROJECT:str=Field(..., validation_alias="GCP_PROJECT_ID")
    GCP_REGION:str=Field("us-central1", validation_alias="GCP_REGION")
    GCS_BUCKET_NAME:str=Field(..., validation_alias="GCS_BUCKET_NAME")
    GCS_PREFIX:str=Field("uploads/", validation_alias="GCS_PREFIX")
    gcp_service_account_path:Path=Field("", validation_alias="GCP_SERVICE_ACCOUNT_PATH")
    vector_search_index_id:str=Field(..., validation_alias="VECTOR_SEARCH_INDEX_ID")
    vector_search_index_endpoint_id:str=Field(..., validation_alias="VECTOR_SEARCH_INDEX_ENDPOINT_ID")

    # --agent middleware limits--
    agent_max_model_calls:int=Field(8, validation_alias="AGENT_MAX_MODEL_CALLS")
    agent_max_tool_calls:int=Field(6, validation_alias="AGENT_MAX_TOOL_CALLS")
    agent_max_retries:int=Field(2, validation_alias="AGENT_MAX_RETRIES")

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT/'.env', env_file_encoding='utf-8', extra="ignore")
    @model_validator(mode="after")
    def _normalise(self)->"Settings":
        #vertex ai vector search expects the prefix to end with "/"
        if self.GCS_PREFIX and not self.GCS_PREFIX.endswith("/"):
            self.GCS_PREFIX += "/"
        # if not self.local_data_dir.exists():
        #     self.local_data_dir.mkdir(parents=True, exist_ok=True)
        return self
    # --convenience ---
    @property
    def is_cloud(self)->bool:
        return self.app_mode=="cloud"
    @property
    def chroma_dir(self)-> str:
        return os.path.join(self.local_data_dir, "chroma")
    @property 
    def local_upload_dir(self)->str:
        return os.path.join(self.local_data_dir, "uploads")

    def missing_cloud_settings(self)->list[str]:
        #what is missing on shared object
        """Names of cloud settings that are still empty (used for clear error messages)"""
        required={
            "GCP_PROJECT_ID": self.GCP_PROJECT,
            "GCP_BUCKET_NAME": self.GCS_BUCKET_NAME,
            "VECTOR_SEARCH_INDEX_ID": self.vector_search_index_id,
            "VECTOR_SEARCH_INDEX_ENDPOINT_ID": self.vector_search_index_endpoint_id,
        }
        return [name for name,value in required.items() if not value]
settings=Settings()
# cloud mode on laptop:point google's client libraries at service account key
#(on Cloud run nothing set here ;the service's own identity is used.)
if settings.is_cloud and settings.gcp_service_account_path:
    key_path=Path(settings.gcp_service_account_path)
    if not key_path.is_absolute():
        key_path=PROJECT_ROOT/key_path
    if key_path.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))


     