from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    pages: int
    message: Optional[str] = None
    is_duplicate: Optional[bool] = False

class QARequest(BaseModel):
    file_hash: str  # Changed from file_id to file_hash for consistency
    question: str
    top_k: Optional[int] = 5

class QAResponse(BaseModel):
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float

class FeedbackRequest(BaseModel):
    file_hash: str  # Changed from file_id to file_hash for consistency
    chunk_id: str
    user_id: Optional[str]
    rating: Optional[int]
    corrected_output: Optional[str]

class SummarizeRequest(BaseModel):
    file_hash: str  # Changed from file_id to file_hash for consistency
