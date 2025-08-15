from fastapi import APIRouter, Depends
from app.models import FeedbackRequest
from app.database import firestore_db
from app.auth import get_current_user
import firebase_admin.firestore as firestore
from app.services.retrain import trigger_retrain_for_user

router = APIRouter()

FEEDBACK_THRESHOLD = 5  # Minimum feedback entries to trigger retrain

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for AI responses to improve the model."""
    user_id = current_user.get("uid")
    
    # Add user_id and timestamp to feedback
    doc = feedback.dict()
    doc["user_id"] = user_id
    doc["timestamp"] = firestore.SERVER_TIMESTAMP
    
    # Optional: Respect confidential flag to exclude from training
    # If client includes {'confidential': true} in feedback, mark it
    if "confidential" in doc and doc["confidential"]:
        doc["not_for_training"] = True
    
    # Store feedback in Firestore
    firestore_db.collection("feedback").add(doc)
    
    # Count non-confidential feedback for this user
    feedback_query = (
        firestore_db.collection("feedback")
        .where("user_id", "==", user_id)
        .where("not_for_training", "!=", True)
    )
    count = len(list(feedback_query.stream()))
    
    triggered = False
    if count >= FEEDBACK_THRESHOLD:
        # Trigger background retrain for user
        trigger_retrain_for_user(user_id)
        triggered = True
    
    return {
        "status": "feedback recorded",
        "user_id": user_id,
        "file_hash": feedback.file_hash,
        "chunk_id": feedback.chunk_id,
        "feedback_count": count,
        "retrain_triggered": triggered
    }
