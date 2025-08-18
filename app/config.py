import os
import json
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import firebase_admin
from firebase_admin import credentials

class Settings(BaseSettings):
    # Inference via OpenRouter
    OPENROUTER_API_KEY: str
    HF_MODEL: str = "anthropic/claude-3-haiku"
    HF_API_TOKEN: Optional[str] = None  # backward compatibility

    # Firebase Configuration
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

# ----------------------------
# Firebase Initialization
# ----------------------------
firebase_json_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if firebase_json_env:
    # Write JSON to file if it exists in environment
    firebase_path = "/app/firebase-config.json"
    with open(firebase_path, "w") as f:
        f.write(firebase_json_env)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = firebase_path
    cred = credentials.Certificate(firebase_path)
else:
    # Fallback for local dev
    local_path = "./firebase_key.json"
    cred = credentials.Certificate(local_path)

# Initialize Firebase only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
