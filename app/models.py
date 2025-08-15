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

class UserProfile(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None
    created_at: str

class UserProfileResponse(BaseModel):
    message: str
    user_data: UserProfile

# Authentication models
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class RegisterResponse(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None
    message: str

# Login models
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    id_token: str
    refresh_token: str
    expires_in: int
    uid: str
    email: str
    message: str

# Google Sign-in models
class GoogleSignInRequest(BaseModel):
    id_token: str  # Google ID token

class GoogleSignInResponse(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None
    message: str

# Advanced feature models
class CompareRequest(BaseModel):
    file_hashes: List[str]
    clause_query: str

class CompareResponse(BaseModel):
    comparison: str
    confidence: float

class SimplifyResponse(BaseModel):
    simplified: str
    confidence: float

class EvidenceHighlight(BaseModel):
    text: str
    page: int
    chunk_index: int
    score: float
    highlight_coords: Optional[Dict[str, float]] = None

class HighlightEvidenceRequest(BaseModel):
    file_hash: str
    question: str

class HighlightEvidenceResponse(BaseModel):
    evidence: List[EvidenceHighlight]
    question: str
    file_hash: str

# Guest mode models
class GuestUploadRequest(BaseModel):
    filename: str
    file_size: int

class GuestUploadResponse(BaseModel):
    file_hash: str
    filename: str
    pages: int
    message: str
    is_guest: bool = True

# History and chat models
class ChatHistoryRequest(BaseModel):
    file_hash: Optional[str] = None
    limit: Optional[int] = 20

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total_count: int

class ConfidentialReportRequest(BaseModel):
    file_hash: str
    report_type: str  # "financial", "legal_risks", "compliance"

class ConfidentialReportResponse(BaseModel):
    report: str
    confidence: float
    is_confidential: bool = True