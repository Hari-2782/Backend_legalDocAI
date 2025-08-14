from fastapi import APIRouter, HTTPException
from app.models import QARequest, QAResponse
from app.services.embedding import query_vectors
from app.services.inference import call_hf_inference, build_rag_prompt
from app.database import firestore_db
import firebase_admin.firestore as firestore

router = APIRouter()

@router.post("/qa", response_model=QAResponse)
async def query_legal_doc(req: QARequest):
    if not req.file_hash:
        raise HTTPException(400, "file_hash is required")
    
    # Query vectors using file_hash
    res = query_vectors(req.question, file_id=req.file_hash, top_k=req.top_k)
    
    if not res["documents"] or not res["documents"][0]:
        raise HTTPException(404, "No relevant documents found for this file")
    
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

    # Save to history
    firestore_db.collection("history").add({
        "file_hash": req.file_hash,
        "question": req.question,
        "answer": answer,
        "confidence": conf,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    return QAResponse(answer=answer, evidence=docs, confidence=conf)
