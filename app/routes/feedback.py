from fastapi import APIRouter
from app.models import FeedbackRequest
from app.database import firestore_db

router = APIRouter()

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    doc = feedback.dict()
    doc["timestamp"] = firestore.SERVER_TIMESTAMP
    # Note: file_hash is already in the feedback object from the model
    firestore_db.collection("feedback").add(doc)
    return {"status": "feedback recorded"}
