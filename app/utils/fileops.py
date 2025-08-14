import os
import uuid
from app.config import settings

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

def save_upload(content: bytes, original_name: str) -> str:
    file_id = f"{uuid.uuid4()}.pdf"
    path = os.path.join(settings.UPLOAD_DIR, file_id)
    with open(path, "wb") as f:
        f.write(content)
    return file_id