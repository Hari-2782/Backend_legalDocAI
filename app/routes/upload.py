from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from app.services.pdf_parser import pdf_parser
from app.services.embedding import embedding_service
from app.database import firestore_db, vector_collection, embedder
import os
import firebase_admin.firestore as firestore
import asyncio
from typing import Optional
import gc

from app.config import settings
from app.auth import get_current_user
from app.models import UploadResponse

router = APIRouter()

async def check_duplicate_file(file_hash: str) -> Optional[dict]:
    """
    Check if a file with the same hash already exists in the database.
    """
    try:
        if not firestore_db:
            return None
        
        # Check documents collection for existing hash
        docs_ref = firestore_db.collection("documents")
        query = docs_ref.where("file_hash", "==", file_hash).limit(1)
        docs = query.stream()
        
        for doc in docs:
            return {
                "exists": True,
                "file_id": doc.id,
                "filename": doc.get("filename"),
                "upload_time": doc.get("upload_time")
            }
        
        return {"exists": False}
        
    except Exception as e:
        print(f"Error checking duplicate: {e}")
        return None

async def process_pdf_background(file_hash: str, file_content: bytes, original_filename: str):
    """
    Background task to process PDF and store in vector database.
    This prevents blocking the upload response.
    """
    try:
        print(f"Starting background processing for file: {original_filename}")
        
        # Process PDF in batches
        result = pdf_parser.process_pdf_in_batches(file_content, batch_size=3)
        
        if "error" in result:
            print(f"Error processing PDF: {result['error']}")
            return
        
        # Store chunks in vector database
        if result["chunks"]:
            success = embedding_service.process_and_store_chunks(result["chunks"])
            if success:
                print(f"Successfully processed {len(result['chunks'])} chunks for {original_filename}")
            else:
                print(f"Failed to store chunks for {original_filename}")
        
        # Update document status in Firestore
        if firestore_db:
            try:
                firestore_db.collection("documents").document(file_hash).update({
                    "processing_status": "completed",
                    "total_chunks": result.get("total_chunks", 0),
                    "processing_completed": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f"Error updating document status: {e}")
        
        # Clean up
        del file_content, result
        gc.collect()
        
    except Exception as e:
        print(f"Background processing error for {original_filename}: {e}")
        # Update status to failed
        if firestore_db:
            try:
                firestore_db.collection("documents").document(file_hash).update({
                    "processing_status": "failed",
                    "error": str(e)
                })
            except Exception as update_error:
                print(f"Error updating failed status: {update_error}")

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if required services are initialized
        if not vector_collection:
            raise HTTPException(500, "Vector database not available")
        if not embedder:
            raise HTTPException(500, "Embedding model not available")
        if not firestore_db:
            raise HTTPException(500, "Document database not available")
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDFs are supported")
        
        # Read file content
        content = await file.read()
        
        # Check file size to prevent memory issues
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(400, "File too large. Maximum size is 50MB.")
        
        # Calculate file hash for duplicate detection
        file_hash = pdf_parser.calculate_file_hash(content)
        
        # Check for duplicate files
        duplicate_check = await check_duplicate_file(file_hash)
        if duplicate_check and duplicate_check.get("exists"):
            existing_file = duplicate_check
            return UploadResponse(
                file_id=existing_file["file_id"],
                filename=existing_file["filename"],
                pages=0,
                message=f"File already exists (uploaded on {existing_file['upload_time']})",
                is_duplicate=True
            )
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file with hash-based name for consistency
        file_extension = os.path.splitext(file.filename)[1]
        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_hash}{file_extension}")
        
        # Save file content
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Get basic file info for immediate response
        basic_info = pdf_parser.extract_text_from_pdf_bytes(content, max_pages=None)  # Process ALL pages
        
        # Save initial document metadata
        doc_data = {
            "filename": file.filename,
            "file_hash": file_hash,
            "file_path": file_path,
            "file_size": len(content),
            "pages": basic_info.get("total_pages", 0),
            "upload_time": firestore.SERVER_TIMESTAMP,
            "processing_status": "processing",
            "original_filename": file.filename,
            "owner_id": current_user.get("uid")
        }
        
        firestore_db.collection("documents").document(file_hash).set(doc_data)
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(
                process_pdf_background,
                file_hash,
                content,
                file.filename
            )
        
        # Clean up content from memory
        del content
        gc.collect()
        
        return UploadResponse(
            file_id=file_hash,
            filename=file.filename,
            pages=basic_info.get("total_pages", 0),
            message="File uploaded successfully. Processing in background.",
            is_duplicate=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on any unexpected error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, f"Upload failed: {str(e)}")

@router.get("/upload/status/{file_hash}")
async def get_upload_status(file_hash: str, current_user: dict = Depends(get_current_user)):
    """
    Get the processing status of an uploaded file.
    """
    try:
        if not firestore_db:
            raise HTTPException(500, "Database not available")
        
        doc_ref = firestore_db.collection("documents").document(file_hash)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(404, "File not found")
        
        data = doc.to_dict()
        # Enforce ownership
        if data.get("owner_id") != current_user.get("uid"):
            raise HTTPException(404, "File not found")
        return {
            "file_hash": file_hash,
            "filename": data.get("filename"),
            "status": data.get("processing_status", "unknown"),
            "pages": data.get("pages", 0),
            "chunks": data.get("total_chunks", 0),
            "upload_time": data.get("upload_time"),
            "error": data.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error getting status: {str(e)}")

@router.delete("/upload/{file_hash}")
async def delete_file(file_hash: str, current_user: dict = Depends(get_current_user)):
    """
    Delete an uploaded file and its associated data.
    """
    try:
        if not firestore_db:
            raise HTTPException(500, "Database not available")
        
        # Get file info
        doc_ref = firestore_db.collection("documents").document(file_hash)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(404, "File not found")
        
        data = doc.to_dict()
        # Enforce ownership
        if data.get("owner_id") != current_user.get("uid"):
            raise HTTPException(404, "File not found")
        file_path = data.get("file_path")
        
        # Delete file from disk
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from vector database (this would need to be implemented)
        # For now, just delete from Firestore
        
        # Delete document
        doc_ref.delete()
        
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error deleting file: {str(e)}")
