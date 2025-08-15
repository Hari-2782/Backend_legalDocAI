from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.pdf_parser import pdf_parser
from app.services.embedding import embedding_service
from app.database import firestore_db, vector_collection, embedder
import os
import firebase_admin.firestore as firestore
import asyncio
from typing import Optional
import gc
import uuid

from app.config import settings
from app.models import GuestUploadResponse, QARequest, QAResponse, SummarizeRequest

router = APIRouter()

async def process_pdf_background_guest(file_hash: str, file_content: bytes, original_filename: str):
    """Background task for guest PDF processing - no user tracking."""
    try:
        print(f"Starting guest background processing for file: {original_filename}")
        
        # Process PDF in batches
        result = pdf_parser.process_pdf_in_batches(file_content, batch_size=3)
        
        if "error" in result:
            print(f"Error processing PDF: {result['error']}")
            return
        
        # Store chunks in vector database
        if result["chunks"]:
            success = embedding_service.process_and_store_chunks(result["chunks"])
            if success:
                print(f"Successfully processed {len(result['chunks'])} chunks for guest file {original_filename}")
            else:
                print(f"Failed to store chunks for guest file {original_filename}")
        
        # Update document status in Firestore (guest documents)
        if firestore_db:
            try:
                firestore_db.collection("guest_documents").document(file_hash).update({
                    "processing_status": "completed",
                    "total_chunks": result.get("total_chunks", 0),
                    "processing_completed": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f"Error updating guest document status: {e}")
        
        # Clean up
        del file_content, result
        gc.collect()
        
    except Exception as e:
        print(f"Guest background processing error for {original_filename}: {e}")
        # Update status to failed
        if firestore_db:
            try:
                firestore_db.collection("guest_documents").document(file_hash).update({
                    "processing_status": "failed",
                    "error": str(e)
                })
            except Exception as update_error:
                print(f"Error updating failed status: {update_error}")

@router.post("/guest/upload", response_model=GuestUploadResponse)
async def upload_pdf_guest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload PDF for guest users - no authentication required."""
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
        
        # Calculate file hash for guest document
        file_hash = pdf_parser.calculate_file_hash(content)
        guest_hash = f"guest_{file_hash}"  # Prefix to distinguish from user documents
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file with guest hash-based name
        file_extension = os.path.splitext(file.filename)[1]
        file_path = os.path.join(settings.UPLOAD_DIR, f"{guest_hash}{file_extension}")
        
        # Save file content
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Get basic file info for immediate response
        basic_info = pdf_parser.extract_text_from_pdf_bytes(content, max_pages=None)
        
        # Save guest document metadata (no user tracking)
        doc_data = {
            "filename": file.filename,
            "file_hash": guest_hash,
            "original_hash": file_hash,
            "file_path": file_path,
            "file_size": len(content),
            "pages": basic_info.get("total_pages", 0),
            "upload_time": firestore.SERVER_TIMESTAMP,
            "processing_status": "processing",
            "original_filename": file.filename,
            "is_guest": True,
            "session_id": str(uuid.uuid4())  # Temporary session tracking
        }
        
        firestore_db.collection("guest_documents").document(guest_hash).set(doc_data)
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(
                process_pdf_background_guest,
                guest_hash,
                content,
                file.filename
            )
        
        # Clean up content from memory
        del content
        gc.collect()
        
        return GuestUploadResponse(
            file_hash=guest_hash,
            filename=file.filename,
            pages=basic_info.get("total_pages", 0),
            message="Guest file uploaded successfully. Processing in background.",
            is_guest=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on any unexpected error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, f"Guest upload failed: {str(e)}")

@router.post("/guest/qa", response_model=QAResponse)
async def query_legal_doc_guest(req: QARequest):
    """Query legal document for guest users - no authentication required."""
    if not req.file_hash.startswith("guest_"):
        raise HTTPException(400, "Invalid guest file hash")
    
    # Query vectors using guest file_hash
    res = query_vectors(req.question, file_id=req.file_hash, top_k=req.top_k)
    
    if not res["documents"] or not res["documents"][0]:
        raise HTTPException(404, "No relevant documents found for this guest file")
    
    docs = []
    snippets = []
    
    # Safely iterate over results
    for i in range(len(res["documents"][0])):
        doc_id = res["ids"][0][i] if res["ids"] and res["ids"][0] else f"doc_{i}"
        doc_text = res["documents"][0][i] if res["documents"] and res["documents"][0] else ""
        meta = res["metadatas"][0][i] if res["metadatas"] and res["metadatas"][0] else {}
        score = res["distances"][0][i] if res["distances"] and res["distances"][0] else 0.0
        
        docs.append({"chunk_id": doc_id, "text": doc_text, "meta": meta, "score": score})
        snippets.append(doc_text)

    prompt = build_rag_prompt(req.question, snippets)
    answer, conf = call_hf_inference(prompt)

    # No history saved for guest users
    return QAResponse(answer=answer, evidence=docs, confidence=conf)

@router.post("/guest/summarize")
async def summarize_contract_guest(req: SummarizeRequest):
    """Summarize legal document for guest users - no authentication required."""
    if not req.file_hash.startswith("guest_"):
        raise HTTPException(400, "Invalid guest file hash")
    
    # Get document chunks
    res = query_vectors("", file_id=req.file_hash, top_k=20)
    docs = res.get("documents", [[]])
    if not docs or not docs[0]:
        raise HTTPException(404, "Guest document not found")
    
    # Join all chunks for comprehensive summary
    joined_text = "\n\n".join(docs[0])
    prompt = f"""Analyze the following legal document and provide a comprehensive summary:

{joined_text}

Please provide a structured summary with:
1. Document Type and Purpose
2. Key Parties Involved
3. Main Terms and Conditions
4. Important Deadlines or Dates
5. Financial Obligations
6. Termination Conditions

Summary:"""
    
    answer, confidence = call_hf_inference(prompt)
    return {"summary": answer, "confidence": confidence}

@router.get("/guest/status/{file_hash}")
async def get_guest_upload_status(file_hash: str):
    """Get the processing status of a guest uploaded file."""
    try:
        if not firestore_db:
            raise HTTPException(500, "Database not available")
        
        if not file_hash.startswith("guest_"):
            raise HTTPException(400, "Invalid guest file hash")
        
        doc_ref = firestore_db.collection("guest_documents").document(file_hash)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(404, "Guest file not found")
        
        data = doc.to_dict()
        return {
            "file_hash": file_hash,
            "filename": data.get("filename"),
            "status": data.get("processing_status", "unknown"),
            "pages": data.get("pages", 0),
            "chunks": data.get("total_chunks", 0),
            "upload_time": data.get("upload_time"),
            "error": data.get("error"),
            "is_guest": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error getting guest status: {str(e)}")
