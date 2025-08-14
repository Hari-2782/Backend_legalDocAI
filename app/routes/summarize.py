from fastapi import APIRouter, HTTPException
from app.models import SummarizeRequest
from app.services.embedding import query_vectors
from app.services.inference import call_hf_inference

router = APIRouter()

@router.post("/summarize")
async def summarize_contract(req: SummarizeRequest):
    res = query_vectors("", file_id=req.file_hash, top_k=10)
    docs = res.get("documents", [])
    if not docs:
        raise HTTPException(404, "Document not found")

    joined_text = "\n\n".join(docs)
    prompt = f"Summarize the following contract in simple terms:\n\n{joined_text}\n\nSummary:"
    answer, confidence = call_hf_inference(prompt)
    return {"summary": answer, "confidence": confidence}
