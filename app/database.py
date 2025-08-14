import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from sentence_transformers import SentenceTransformer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase init
try:
    cred = credentials.Certificate(settings.FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
    firestore_db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    firestore_db = None

# ChromaDB init - Support both local and cloud
try:
    if settings.USE_CHROMA_CLOUD and settings.CHROMA_CLOUD_HOST and settings.CHROMA_CLOUD_API_KEY:
        # Use ChromaDB Cloud with CloudClient
        logger.info("Initializing ChromaDB Cloud connection...")
        logger.info(f"Host: {settings.CHROMA_CLOUD_HOST}")
        logger.info(f"Tenant: {settings.CHROMA_CLOUD_TENANT}")
        logger.info(f"Database: {settings.CHROMA_CLOUD_DATABASE}")
        
        # Create cloud client using CloudClient
        chroma_client = chromadb.CloudClient(
            api_key=settings.CHROMA_CLOUD_API_KEY,
            tenant=settings.CHROMA_CLOUD_TENANT if settings.CHROMA_CLOUD_TENANT else None,
            database=settings.CHROMA_CLOUD_DATABASE if settings.CHROMA_CLOUD_DATABASE else None
        )
        logger.info("ChromaDB Cloud client created successfully")
        
    else:
        # Use local ChromaDB
        logger.info("Initializing local ChromaDB...")
        
        # Ensure local directory exists
        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
        
        # Create local persistent client
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        logger.info("Local ChromaDB client created successfully")
    
    # Try to get existing collection or create new one
    collection_name = "legal_chunks"
    
    # First, try to get the existing collection
    try:
        vector_collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Using existing ChromaDB collection: {collection_name}")
        
        # Verify the collection has data
        try:
            count = vector_collection.count()
            logger.info(f"Collection contains {count} documents")
            
            if count == 0:
                logger.warning("Collection exists but is empty - this might indicate a connection issue")
            else:
                logger.info("✅ Collection is properly connected and contains data")
                
        except Exception as e:
            logger.warning(f"Could not count documents in collection: {e}")
            
    except Exception as e:
        logger.info(f"Collection '{collection_name}' not found, creating new one: {e}")
        
        # Create new collection
        vector_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Legal document chunks for RAG"}
        )
        logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    logger.info("ChromaDB initialized successfully")
    
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    vector_collection = None

# Embedding model
try:
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info(f"Embedding model loaded: {settings.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Embedding model loading failed: {e}")
    embedder = None
