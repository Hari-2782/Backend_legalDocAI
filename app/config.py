import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HF_API_TOKEN: str
    HF_MODEL: str = "microsoft/DialoGPT-medium"  # Better model for text generation
    FIREBASE_KEY_PATH: str = "./firebase_key.json"
    
    # ChromaDB Configuration - Choose between local and cloud
    CHROMA_DB_DIR: str = "./chroma_db"  # Local storage path
    CHROMA_CLOUD_HOST: str = "api.trychroma.com"  # ChromaDB Cloud host
    CHROMA_CLOUD_API_KEY: str = "ck-7vpLMM8BhPQp8SKd7qsu2zvAZqif9cA2jrDES86GxJsv"  # ChromaDB Cloud API key
    
    # ChromaDB Cloud settings - required for CloudClient
    CHROMA_CLOUD_TENANT: str = "57565ed5-fc18-490b-b764-1a343d5627d6"  # ChromaDB Cloud tenant
    CHROMA_CLOUD_DATABASE: str = "legal_doc"  # ChromaDB Cloud database name
    
    # Use cloud if configured, otherwise fallback to local
    USE_CHROMA_CLOUD: bool = True
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    UPLOAD_DIR: str = "./uploads"
    
    # Text chunking settings
    MAX_CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # Memory optimization settings
    EMBEDDING_BATCH_SIZE: int = 32
    VECTOR_DB_BATCH_SIZE: int = 16
    PDF_PROCESSING_BATCH_SIZE: int = 3
    # MAX_PDF_PAGES: int = 100  # Removed - now process all pages
    # MAX_TEXT_PER_PAGE: int = 50000  # Removed - no text limit per page
    
    # File size limits
    MAX_FILE_SIZE_MB: int = 50
    
    # Processing settings
    ENABLE_BACKGROUND_PROCESSING: bool = True
    ENABLE_DUPLICATE_DETECTION: bool = True
    
    RETRAIN_DATASET_PATH: str = "./retrain/dataset.jsonl"

    class Config:
        env_file = ".env"

settings = Settings()
