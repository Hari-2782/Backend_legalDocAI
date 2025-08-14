import fitz
import hashlib
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os
import gc

class OptimizedPDFParser:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content for duplicate detection"""
        return hashlib.sha256(content).hexdigest()
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, max_pages: int = None) -> Dict:
        """
        Extract text from PDF bytes with optimized memory management.
        Returns dict with text, hash, and metadata.
        """
        try:
            # Create temporary file to avoid memory issues
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Use PyMuPDFLoader for better memory management
                loader = PyMuPDFLoader(temp_file_path)
                pages = loader.load()
                
                # Process all pages if no limit specified, otherwise use the limit
                if max_pages is not None:
                    pages = pages[:max_pages]
                    print(f"Processing limited to {len(pages)} pages")
                else:
                    print(f"Processing all {len(pages)} pages")
                
                # Extract text and metadata
                extracted_data = {
                    "hash": self.calculate_file_hash(pdf_bytes),
                    "pages": [],
                    "total_pages": len(pages),
                    "file_size": len(pdf_bytes)
                }
                
                for i, page in enumerate(pages):
                    page_data = {
                        "page": i + 1,
                        "text": page.page_content,  # Remove the 50000 character limit
                        "metadata": page.metadata
                    }
                    extracted_data["pages"].append(page_data)
                    
                    # Force garbage collection every few pages
                    if i % 10 == 0:
                        gc.collect()
                
                return extracted_data
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {
                "hash": self.calculate_file_hash(pdf_bytes),
                "error": str(e),
                "chunks": [],
                "total_chunks": 0
            }
    
    def chunk_text_optimized(self, text: str) -> List[str]:
        """
        Use LangChain's optimized text splitting for better memory management.
        """
        try:
            if not text or len(text.strip()) < 50:
                return []
            
            # Use LangChain's text splitter
            chunks = self.text_splitter.split_text(text)
            
            # Filter out very short chunks
            filtered_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
            
            return filtered_chunks
            
        except Exception as e:
            print(f"Error chunking text: {e}")
            return []
    
    def process_pdf_in_batches(self, pdf_bytes: bytes, batch_size: int = 3) -> Dict:
        """
        Process PDF in small batches to minimize memory usage.
        """
        try:
            # Process ALL pages without limit
            extracted_data = self.extract_text_from_pdf_bytes(pdf_bytes, max_pages=None)
            
            if "error" in extracted_data:
                return extracted_data
            
            all_chunks = []
            total_chunks = 0
            
            # Process pages in batches
            for i in range(0, len(extracted_data["pages"]), batch_size):
                batch_pages = extracted_data["pages"][i:i + batch_size]
                
                for page_data in batch_pages:
                    page_text = page_data["text"]
                    page_num = page_data["page"]
                    
                    # Chunk the page text
                    chunks = self.chunk_text_optimized(page_text)
                    
                    for idx, chunk in enumerate(chunks):
                        chunk_data = {
                            "text": chunk,
                            "page": page_num,
                            "chunk_id": f"{extracted_data['hash']}::p{page_num}::c{idx}",
                            "metadata": {
                                "file_hash": extracted_data["hash"],
                                "page": page_num,
                                "chunk_index": idx
                            }
                        }
                        all_chunks.append(chunk_data)
                        total_chunks += 1
                
                # Force garbage collection after each batch
                gc.collect()
            
            extracted_data["chunks"] = all_chunks
            extracted_data["total_chunks"] = total_chunks
            
            return extracted_data
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return {
                "hash": self.calculate_file_hash(pdf_bytes),
                "error": str(e),
                "chunks": [],
                "total_chunks": 0
            }

# Create global instance
pdf_parser = OptimizedPDFParser()

# Backward compatibility functions
def extract_text_from_pdf_bytes(pdf_bytes: bytes):
    """Legacy function for backward compatibility"""
    return pdf_parser.extract_text_from_pdf_bytes(pdf_bytes)

def chunk_text(text: str, chunk_size=800, overlap=100):
    """Legacy function for backward compatibility"""
    return pdf_parser.chunk_text_optimized(text)
