import os
import json
from app.database import firestore_db
from app.config import settings


def build_retrain_dataset_for_user(user_id: str) -> str:
    """Build a JSONL dataset from feedback belonging to a specific user."""
    feedback_docs = (
        firestore_db
        .collection("feedback")
        .where("user_id", "==", user_id)
        .stream()
    )

    dataset = []
    for doc in feedback_docs:
        data = doc.to_dict()
        # Use file_hash consistently; include minimal prompt structure
        corrected = data.get("corrected_output")
        chunk_id = data.get("chunk_id")
        if corrected and chunk_id:
            dataset.append({
                "prompt": f"Improve answer for chunk {chunk_id} based on legal context:",
                "completion": corrected
            })

    # Persist dataset
    os.makedirs(os.path.dirname(settings.RETRAIN_DATASET_PATH), exist_ok=True)
    dataset_path = settings.RETRAIN_DATASET_PATH.replace(".jsonl", f"_{user_id}.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return dataset_path


def trigger_retrain_for_user(user_id: str) -> None:
    """Trigger a user-scoped retrain job using LoRA script with provided dataset."""
    dataset_path = build_retrain_dataset_for_user(user_id)
    # This call should ideally be delegated to a job runner
    os.system(f"python retrain/train_lora.py --user_id {user_id} --dataset_path {dataset_path}")
