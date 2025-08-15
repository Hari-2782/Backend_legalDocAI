from fastapi import APIRouter, HTTPException, Depends
from app.models import (
    SummarizeRequest, CompareRequest, CompareResponse, 
    SimplifyResponse, HighlightEvidenceRequest, HighlightEvidenceResponse
)
from app.services.embedding import query_vectors
from app.services.inference import call_hf_inference
from app.auth import get_current_user
from app.database import firestore_db
from typing import List, Dict, Any
import firebase_admin.firestore as firestore

router = APIRouter()

@router.post("/summarize")
async def summarize_contract(
    req: SummarizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Summarize a legal document with ownership verification."""
    user_id = current_user.get("uid")
    
    # Verify document ownership
    doc_ref = firestore_db.collection("documents").document(req.file_hash)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = doc.to_dict()
    owner_id = doc_data.get("owner_id")
    
    # If no owner_id is set (old documents), allow access but log it
    if owner_id and owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found or access denied.")
    
    # Get document chunks
    res = query_vectors("", file_id=req.file_hash, top_k=20)  # Get more chunks for better summary
    docs = res.get("documents", [[]])
    if not docs or not docs[0]:
        raise HTTPException(404, "Document not found")
    
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

@router.post("/simplify", response_model=SimplifyResponse)
async def simplify_document(
    req: SummarizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Transform legal document into plain English with structured breakdown."""
    user_id = current_user.get("uid")
    
    # Verify document ownership
    doc_ref = firestore_db.collection("documents").document(req.file_hash)
    doc = doc_ref.get()
    if not doc.exists or doc.to_dict().get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Document not found or access denied.")
    
    # Get all document chunks
    res = query_vectors("", file_id=req.file_hash, top_k=50)  # Get all chunks
    docs = res.get("documents", [[]])
    if not docs or not docs[0]:
        raise HTTPException(404, "Document not found")
    
    # Join all chunks
    full_text = "\n\n".join(docs[0])
    
    prompt = f"""Analyze the entire legal document provided below. Your task is to "translate" it into plain English. Create a structured summary with the following markdown sections:

### 🔑 Key Terms
(Define important terms like 'Lessee', 'Force Majeure', etc.)

### ✅ Your Obligations
(A bulleted list of everything the user MUST do.)

### ⚠️ Potential Risks & Red Flags
(A bulleted list of penalties, auto-renewals, or unfair terms.)

### 📈 Financial Summary
(List all costs, deposits, and fees mentioned.)

### 📅 Important Dates
(All deadlines, renewal dates, and time-sensitive items.)

Document Text:
{full_text}

Structured Analysis:"""
    
    answer, confidence = call_hf_inference(prompt)
    return SimplifyResponse(simplified=answer, confidence=confidence)

@router.post("/compare", response_model=CompareResponse)
async def compare_clauses(
    req: CompareRequest,
    current_user: dict = Depends(get_current_user)
):
    """Compare specific clauses across multiple documents."""
    user_id = current_user.get("uid")
    file_hashes = req.file_hashes
    clause_query = req.clause_query
    
    if len(file_hashes) < 2:
        raise HTTPException(400, "At least 2 documents required for comparison")
    
    if not clause_query:
        raise HTTPException(400, "Clause query is required")
    
    # Verify ownership of all documents
    for file_hash in file_hashes:
        doc_ref = firestore_db.collection("documents").document(file_hash)
        doc = doc_ref.get()
        if not doc.exists or doc.to_dict().get("owner_id") != user_id:
            raise HTTPException(status_code=404, detail=f"Document {file_hash} not found or access denied.")
    
    # Get relevant chunks from each document
    document_texts = []
    for i, file_hash in enumerate(file_hashes):
        res = query_vectors(clause_query, file_id=file_hash, top_k=5)
        docs = res.get("documents", [[]])
        if docs and docs[0]:
            doc_text = "\n\n".join(docs[0])
            document_texts.append(f"Document {i+1}:\n{doc_text}")
    
    if not document_texts:
        raise HTTPException(404, "No relevant clauses found in the specified documents")
    
    # Build comparison prompt
    comparison_text = "\n\n".join(document_texts)
    prompt = f"""You are a legal analyst. Compare and contrast the following clauses from different documents. Explain the key differences in simple terms.

Query: "{clause_query}"

{comparison_text}

Comparison Analysis:
### Key Similarities
### Key Differences
### Recommendations
### Risk Assessment"""
    
    answer, confidence = call_hf_inference(prompt)
    return CompareResponse(comparison=answer, confidence=confidence)

@router.post("/highlight-evidence", response_model=HighlightEvidenceResponse)
async def get_evidence_highlights(
    req: HighlightEvidenceRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get evidence highlights with page coordinates for frontend rendering."""
    user_id = current_user.get("uid")
    file_hash = req.file_hash
    question = req.question
    
    # Verify document ownership
    doc_ref = firestore_db.collection("documents").document(file_hash)
    doc = doc_ref.get()
    if not doc.exists or doc.to_dict().get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Document not found or access denied.")
    
    # Get relevant chunks with metadata
    res = query_vectors(question, file_id=file_hash, top_k=5)
    
    if not res.get("documents") or not res["documents"][0]:
        raise HTTPException(404, "No relevant evidence found")
    
    # Format evidence with highlighting info
    evidence = []
    for i in range(len(res["documents"][0])):
        doc_text = res["documents"][0][i]
        metadata = res["metadatas"][0][i] if res["metadatas"] and res["metadatas"][0] else {}
        score = res["distances"][0][i] if res["distances"] and res["distances"][0] else 0.0
        
        evidence.append({
            "text": doc_text,
            "page": metadata.get("page", 1),
            "chunk_index": metadata.get("chunk_index", i),
            "score": score,
            "highlight_coords": metadata.get("bbox", None)  # For future PDF highlighting
        })
    
    return HighlightEvidenceResponse(
        evidence=evidence,
        question=question,
        file_hash=file_hash
    )


# Temporary endpoint to fix existing documents (remove in production)
@router.post("/fix-ownership")
async def fix_document_ownership(
    req: SummarizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Temporary endpoint to fix ownership of existing documents."""
    user_id = current_user.get("uid")
    
    doc_ref = firestore_db.collection("documents").document(req.file_hash)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Update the document with owner_id
    doc_ref.update({"owner_id": user_id})
    
    return {"message": f"Document {req.file_hash} ownership fixed for user {user_id}"}
