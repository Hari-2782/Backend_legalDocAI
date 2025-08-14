import os
import json
from app.database import firestore_db
from app.config import settings
from subprocess import Popen

def build_retrain_dataset():
    feedback_docs = firestore_db.collection("feedback").stream()
    dataset = []
    for doc in feedback_docs:
        data = doc.to_dict()
        # Simplified example to convert feedback into training samples
        if data.get("corrected_output") and data.get("chunk_id") and data.get("file_id"):
            dataset.append({
                "prompt": f"Question on chunk {data['chunk_id']}: ",
                "completion": data["corrected_output"]
            })

    dataset_path = settings.RETRAIN_DATASET_PATH
    with open(dataset_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

def trigger_retrain():
    build_retrain_dataset()
    # Call external LoRA train script here (can be a shell command or python script)
    os.system("python retrain/train_lora.py")
