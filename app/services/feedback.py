from typing import Dict, Any
from app.database import firestore_db

COLL = "feedback"


def record_feedback(doc: Dict[str, Any]):
    doc["timestamp"] = firestore_db.SERVER_TIMESTAMP
    firestore_db.collection(COLL).add(doc)


def record_history(item: Dict[str, Any]):
    item["timestamp"] = firestore_db.SERVER_TIMESTAMP
    firestore_db.collection("history").add(item)