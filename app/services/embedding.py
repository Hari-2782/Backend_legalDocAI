from app.database import embedder, vector_collection
import logging
import gc
from typing import List, Dict, Any
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedEmbeddingService:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts in batches to manage memory usage.
        """
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    # Embed batch
                    batch_embeddings = embedder.encode(
                        batch_texts, 
                        show_progress_bar=False, 
                        convert_to_numpy=True,
                        batch_size=min(len(batch_texts), self.batch_size)
                    )
                    
                    # Convert to list and add to results
                    if hasattr(batch_embeddings, 'tolist'):
                        batch_embeddings_list = batch_embeddings.tolist()
                    else:
                        batch_embeddings_list = batch_embeddings
                    
                    all_embeddings.extend(batch_embeddings_list)
                    
                    # Force garbage collection after each batch
                    del batch_embeddings
                    gc.collect()
                    
                    logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                    
                except Exception as e:
                    logger.error(f"Error embedding batch {i//self.batch_size + 1}: {e}")
                    # Add empty embeddings for failed batch
                    all_embeddings.extend([[0.0] * 384] * len(batch_texts))  # Assuming 384-dim embeddings
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise
    
    def add_vectors_batch(self, ids: List[str], documents: List[str], 
                          metadatas: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Add vectors to database in batches to manage memory.
        """
        try:
            if not all([ids, documents, metadatas, embeddings]):
                logger.warning("Missing required parameters for add_vectors_batch")
                return False
            
            logger.info(f"Starting to add {len(ids)} vectors to ChromaDB Cloud...")
            
            # Process in smaller batches for vector database
            vector_batch_size = min(self.batch_size, 16)  # Smaller batches for vector DB
            
            for i in range(0, len(ids), vector_batch_size):
                batch_ids = ids[i:i + vector_batch_size]
                batch_docs = documents[i:i + vector_batch_size]
                batch_metas = metadatas[i:i + vector_batch_size]
                batch_embs = embeddings[i:i + vector_batch_size]
                
                try:
                    logger.info(f"Adding batch {i//vector_batch_size + 1}/{(len(ids) + vector_batch_size - 1)//vector_batch_size}")
                    logger.info(f"Batch size: {len(batch_ids)} vectors")
                    
                    # Debug: Print first item details
                    if i == 0:
                        logger.info(f"First vector ID: {batch_ids[0]}")
                        logger.info(f"First document preview: {batch_docs[0][:100]}...")
                        logger.info(f"First metadata: {batch_metas[0]}")
                        logger.info(f"First embedding length: {len(batch_embs[0])}")
                    
                    vector_collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas,
                        embeddings=batch_embs
                    )
                    
                    logger.info(f"✓ Successfully added batch {i//vector_batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"❌ Error adding batch {i//vector_batch_size + 1}: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    return False
                
                # Clean up batch variables
                del batch_ids, batch_docs, batch_metas, batch_embs
                gc.collect()
            
            logger.info(f"✅ Successfully added {len(ids)} vectors to ChromaDB Cloud")
            
            # Verify storage by counting
            try:
                total_count = vector_collection.count()
                logger.info(f"✅ Total vectors in collection after addition: {total_count}")
            except Exception as e:
                logger.warning(f"Could not verify vector count: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding vectors in batches: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False
    
    def process_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Process chunks and store them efficiently in the vector database.
        """
        try:
            if not chunks:
                logger.warning("No chunks to process")
                return False
            
            # Extract data for processing
            texts = [chunk["text"] for chunk in chunks]
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings in batches
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embed_texts_batch(texts)
            
            if not embeddings or len(embeddings) != len(texts):
                logger.error("Embedding generation failed or mismatch")
                return False
            
            # Store vectors in batches
            logger.info(f"Storing {len(embeddings)} vectors in database...")
            success = self.add_vectors_batch(chunk_ids, texts, metadatas, embeddings)
            
            # Clean up
            del texts, chunk_ids, metadatas, embeddings
            gc.collect()
            
            return success
            
        except Exception as e:
            logger.error(f"Error in process_and_store_chunks: {e}")
            return False

# Create global instance
embedding_service = OptimizedEmbeddingService()

# Backward compatibility functions
def embed_texts(texts):
    """Legacy function for backward compatibility"""
    return embedding_service.embed_texts_batch(texts)

def add_vectors(ids, documents, metadatas, embeddings):
    """Legacy function for backward compatibility"""
    return embedding_service.add_vectors_batch(ids, documents, metadatas, embeddings)

def query_vectors(query_text, file_id=None, top_k=5):
    try:
        filter = {"file_hash": file_id} if file_id else {}
        
        if not query_text:
            logger.info(f"Getting all documents for file_hash: {file_id}")
            # When no query text, get all documents for the file
            # Use a generic query to get all documents
            results = vector_collection.query(
                query_texts=["document"],  # Generic query to get all
                n_results=top_k, 
                where=filter
            )
        else:
            results = vector_collection.query(
                query_texts=[query_text], 
                n_results=top_k, 
                where=filter
            )
        
        return results
    except Exception as e:
        logger.error(f"Error querying vectors: {e}")
        return {"documents": [], "metadatas": [], "distances": []}
