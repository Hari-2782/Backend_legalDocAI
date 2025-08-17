from fastapi import APIRouter, HTTPException, Depends
from app.models import QARequest, QAResponse, HighlightItem, HighlightDisplayItem, QAResponseV2
from app.services.embedding import query_vectors
from app.services.inference import call_hf_inference, build_rag_prompt
import json
from app.database import firestore_db
from app.auth import get_current_user
import firebase_admin.firestore as firestore

router = APIRouter()

@router.post("/qa", response_model=QAResponseV2)
async def query_legal_doc(
    req: QARequest,
    current_user: dict = Depends(get_current_user)
):
    if not req.file_hash:
        raise HTTPException(400, "file_hash is required")
    
    # Verify document access
    user_id = current_user.get("uid")
    doc_ref = firestore_db.collection("documents").document(req.file_hash)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="File not found.")

    doc_data = doc.to_dict()
    if user_id not in doc_data.get("authorized_users", []):
        raise HTTPException(status_code=403, detail="Access denied.")
    
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
    raw_answer, conf = call_hf_inference(prompt)

    parsed_answer = None
    highlights: list = []  # legacy string list
    highlight_items: list[HighlightItem] = []
    category = None
    suggestions = []
    answer = raw_answer

    # Try to parse strict JSON response
    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", answer)
            conf = float(parsed.get("confidence", conf))
            # suggestions
            suggestions = parsed.get("suggestions", []) or []
            # parse highlights: accept array of objects or strings
            parsed_highlights = parsed.get("highlights", []) or []
            if parsed_highlights and isinstance(parsed_highlights, list):
                if isinstance(parsed_highlights[0], dict):
                    for item in parsed_highlights:
                        hi = HighlightItem(
                            text=item.get("text", "").strip(),
                            category=item.get("category", "Clause"),
                            page=item.get("page"),
                            chunk_id=item.get("chunk_id"),
                            score=item.get("score"),
                        )
                        highlight_items.append(hi)
                else:
                    # list of strings; keep legacy highlights
                    highlights = [str(h) for h in parsed_highlights]
            # optional overall category
            category = parsed.get("category")
    except Exception:
        # Fallback: derive minimal highlights and suggestions
        for i, (s, d) in enumerate(zip(snippets[:4], docs[:4])):
            s_clean = (s or "").strip().replace("\n", " ")
            if s_clean:
                words = s_clean.split()
                text = " ".join(words[:15])
                # Heuristic category
                lowered = text.lower()
                if any(k in lowered for k in ["penalty", "breach", "risk", "liable", "terminate without", "late fee"]):
                    cat = "Risk"
                elif any(k in lowered for k in ["favorable", "benefit", "waive", "cap", "limit liability"]):
                    cat = "Favorable"
                elif any(k in lowered for k in ["may", "general", "standard", "typical"]):
                    cat = "General"
                else:
                    cat = "Clause"
                highlight_items.append(HighlightItem(
                    text=text,
                    category=cat,
                    page=(d.get("meta", {}) or {}).get("page"),
                    chunk_id=d.get("chunk_id"),
                    score=d.get("score"),
                ))
        if not highlight_items:
            highlights = ["No clear highlights found in document excerpts."]
        # Heuristic category
        lowered = answer.lower()
        if any(k in lowered for k in ["penalty", "breach", "risk", "liable", "terminate without", "late fee"]):
            category = "Risk"
        elif any(k in lowered for k in ["may", "general", "standard", "typical"]):
            category = "General"
        elif any(k in lowered for k in ["favorable", "benefit", "waive", "cap", "limit liability"]):
            category = "Favorable"
        else:
            category = "Clause"
        suggestions = [
            "Consult a lawyer for clause-specific advice.",
            "Prepare a reply letter seeking clarification or amendments.",
            "Negotiate terms to reduce risk or add protective language.",
            "Document internal deadlines and compliance steps.",
        ]

    # Save to history
    firestore_db.collection("history").add({
        "user_id": user_id,
        "file_hash": req.file_hash,
        "question": req.question,
        "answer": answer,
        "confidence": conf,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    # Build slim highlights for frontend: include suggestion only for Risk
    display_highlights: list[HighlightDisplayItem] = []
    # Prefer structured highlight_items; else fallback to legacy strings
    if highlight_items:
        for hi in highlight_items:
            suggestion = None
            if hi.category == "Risk":
                # Sri Lanka-oriented suggestion
                suggestion = "Consult a Sri Lanka attorney; consider a tailored reply or amendment."
            display_highlights.append(HighlightDisplayItem(
                text=hi.text,
                category=hi.category,
                suggestion=suggestion,
            ))
    else:
        for h in highlights[:4]:
            text = str(h).strip()
            cat = category or "Clause"
            suggestion = None
            if cat == "Risk":
                suggestion = "Consult a Sri Lanka attorney; consider a tailored reply or amendment."
            display_highlights.append(HighlightDisplayItem(text=text, category=cat, suggestion=suggestion))

    return QAResponseV2(
        answer=answer,
        evidence=docs,
        highlights=display_highlights,
        confidence=conf,
    )
