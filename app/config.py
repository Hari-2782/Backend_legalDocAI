import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
	# Inference via OpenRouter
	OPENROUTER_API_KEY: str
	HF_MODEL: str = "anthropic/claude-3-haiku"  # Used via OpenRouter
	
	# Backward compatibility: keep optional HF token to avoid validation errors
	HF_API_TOKEN: Optional[str] = None
	
	# Firebase Configuration
	FIREBASE_KEY_PATH: str = "./firebase_key.json"
	FIREBASE_WEB_API_KEY: str  # Required for login endpoint
	
	# ChromaDB Configuration - Choose between local and cloud
	CHROMA_DB_DIR: str = "./chroma_db"  # Local storage path
	CHROMA_CLOUD_HOST: str = "api.trychroma.com"  # ChromaDB Cloud host
	CHROMA_CLOUD_API_KEY: str  # Required for ChromaDB Cloud
	
	# ChromaDB Cloud settings - optional; if unset, defaults are used server-side
	CHROMA_CLOUD_TENANT: Optional[str] = None
	CHROMA_CLOUD_DATABASE: Optional[str] = None
	
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

	# Pydantic v2 settings configuration
	model_config = SettingsConfigDict(
		env_file=".env",
		extra="ignore",  # Ignore extra env vars like HF_API_TOKEN if present
	)

settings = Settings()
