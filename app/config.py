import os
import json
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from firebase_admin import credentials, initialize_app

class Settings(BaseSettings):
    # Inference via OpenRouter
    OPENROUTER_API_KEY: str
    HF_MODEL: str = "anthropic/claude-3-haiku"

    # Backward compatibility
    HF_API_TOKEN: Optional[str] = None

    # Firebase Configuration
    FIREBASE_KEY_JSON: Optional[str] = None  # JSON string from Railway env
    FIREBASE_KEY_PATH: str = "./firebase_key.json"  # fallback for local dev
    FIREBASE_WEB_API_KEY: str  # Required for login endpoint

    # ChromaDB Configuration
    CHROMA_DB_DIR: str = "./chroma_db"
    CHROMA_CLOUD_HOST: str = "api.trychroma.com"
    CHROMA_CLOUD_API_KEY: str
    CHROMA_CLOUD_TENANT: Optional[str] = None
    CHROMA_CLOUD_DATABASE: Optional[str] = None
    USE_CHROMA_CLOUD: bool = True

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    UPLOAD_DIR: str = "./uploads"

    # Text chunking settings
    MAX_CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Memory optimization
    EMBEDDING_BATCH_SIZE: int = 32
    VECTOR_DB_BATCH_SIZE: int = 16
    PDF_PROCESSING_BATCH_SIZE: int = 3

    # File size limits
    MAX_FILE_SIZE_MB: int = 50

    # Processing settings
    ENABLE_BACKGROUND_PROCESSING: bool = True
    ENABLE_DUPLICATE_DETECTION: bool = True
    RETRAIN_DATASET_PATH: str = "./retrain/dataset.jsonl"

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

# Initialize Firebase dynamically
if settings.FIREBASE_KEY_JSON:
    firebase_cred = credentials.Certificate(json.loads(settings.FIREBASE_KEY_JSON))
    initialize_app(firebase_cred)
else:
    # Fallback for local dev
    cred_path = settings.FIREBASE_KEY_PATH
    firebase_cred = credentials.Certificate(cred_path)
    initialize_app(firebase_cred)
