# app/routes/users.py
from fastapi import APIRouter, Depends, HTTPException
from app.database import firestore_db
from app.auth import get_current_user
from firebase_admin import auth as fb_auth
from app.models import (
    RegisterRequest, RegisterResponse, LoginRequest, LoginResponse,
    GoogleSignInRequest, GoogleSignInResponse, ChatHistoryRequest, 
    ChatHistoryResponse, ConfidentialReportRequest, ConfidentialReportResponse
)
import datetime
import os
import requests
from app.services.inference import call_hf_inference
import firebase_admin.firestore as firestore
from google.api_core.exceptions import FailedPrecondition

router = APIRouter()

@router.get("/users/profile", tags=["Users"])
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile details."""
    user_id = current_user.get("uid")
    user_ref = firestore_db.collection("users").document(user_id)
    snapshot = user_ref.get()

    if snapshot.exists:
        data = snapshot.to_dict()
    else:
        # Fallback to token info
        data = {
            "uid": user_id,
            "email": current_user.get("email"),
            "name": current_user.get("name"),
            "created_at": None,
            "auth_provider": current_user.get("firebase", {}).get("sign_in_provider")
        }

    return {"status": "success", "user": data}

@router.post("/users/register", response_model=RegisterResponse, tags=["Users"])
async def register_user(payload: RegisterRequest):
    """Register a new user via Firebase Auth and create a profile in Firestore."""
    try:
        user_record = fb_auth.create_user(email=payload.email, password=payload.password)
        uid = user_record.uid

        user_doc = {
            "uid": uid,
            "email": payload.email,
            "name": payload.name,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        firestore_db.collection("users").document(uid).set(user_doc)

        return RegisterResponse(uid=uid, email=payload.email, name=payload.name, message="User registered successfully")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

@router.post("/users/login", response_model=LoginResponse, tags=["Users"])
async def login_user(payload: LoginRequest):
    """Sign in a user using Firebase Identity Toolkit REST API and return ID token."""
    try:
        from app.config import settings
        api_key = os.getenv("FIREBASE_WEB_API_KEY") or getattr(settings, "FIREBASE_WEB_API_KEY", None)
        if not api_key:
            raise RuntimeError("FIREBASE_WEB_API_KEY is not configured")

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        resp = requests.post(url, json={
            "email": payload.email,
            "password": payload.password,
            "returnSecureToken": True
        }, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail=resp.json().get("error", {}).get("message", "LOGIN_FAILED"))

        data = resp.json()
        uid = data.get("localId")

        return LoginResponse(
            id_token=data.get("idToken"),
            refresh_token=data.get("refreshToken"),
            expires_in=int(data.get("expiresIn", 3600)),
            uid=uid,
            email=payload.email,
            message="Login successful"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Login failed: {e}")

@router.post("/users/google-signin", response_model=GoogleSignInResponse, tags=["Users"])
async def google_signin(payload: GoogleSignInRequest):
    """Sign in a user using Google ID token."""
    try:
        decoded_token = fb_auth.verify_id_token(payload.id_token)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name")

        user_ref = firestore_db.collection("users").document(uid)
        if not user_ref.get().exists:
            user_doc = {
                "uid": uid,
                "email": email,
                "name": name,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "auth_provider": "google"
            }
            user_ref.set(user_doc)
        else:
            user_ref.update({
                "last_login": datetime.datetime.utcnow().isoformat(),
                "auth_provider": "google"
            })

        return GoogleSignInResponse(
            uid=uid,
            email=email,
            name=name,
            message="Google sign-in successful"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google sign-in failed: {e}")

@router.post("/users/profile", status_code=201, tags=["Users"])
async def create_user_profile(current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("uid")
    user_ref = firestore_db.collection("users").document(user_id)

    if user_ref.get().exists:
        return {"status": "success", "message": "User profile already exists."}

    user_data = {
        "uid": user_id,
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    user_ref.set(user_data)
    return {"status": "success", "message": "User profile created.", "user": user_data}

@router.get("/users/chat-history", response_model=ChatHistoryResponse, tags=["Users"])
async def get_chat_history(
    req: ChatHistoryRequest = Depends(),
    current_user: dict = Depends(get_current_user)
):
    """Get user's chat history with optional file filtering. Falls back if Firestore index is missing."""
    user_id = current_user.get("uid")

    try:
        query = firestore_db.collection("history").where("user_id", "==", user_id)
        if req.file_hash:
            query = query.where("file_hash", "==", req.file_hash)
        query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)
        docs = list(query.limit(req.limit).stream())
    except FailedPrecondition:
        # Missing composite index; fallback to basic fetch and local filter/sort
        base = firestore_db.collection("history").where("user_id", "==", user_id)
        docs_stream = base.stream()
        items = []
        for d in docs_stream:
            data = d.to_dict()
            if req.file_hash and data.get("file_hash") != req.file_hash:
                continue
            items.append((d.id, data))
        # Sort locally by timestamp desc
        items.sort(key=lambda x: x[1].get("timestamp", datetime.datetime.min), reverse=True)
        items = items[: req.limit]
        docs = items

    history = []
    if docs and isinstance(docs[0], tuple):
        # Fallback path: docs are (id, data)
        for doc_id, data in docs:
            history.append({
                "id": doc_id,
                "file_hash": data.get("file_hash"),
                "question": data.get("question"),
                "answer": data.get("answer"),
                "confidence": data.get("confidence"),
                "timestamp": data.get("timestamp")
            })
    else:
        for doc in docs:
            data = doc.to_dict()
            history.append({
                "id": doc.id,
                "file_hash": data.get("file_hash"),
                "question": data.get("question"),
                "answer": data.get("answer"),
                "confidence": data.get("confidence"),
                "timestamp": data.get("timestamp")
            })

    return ChatHistoryResponse(history=history, total_count=len(history))

@router.post("/users/confidential-report", response_model=ConfidentialReportResponse, tags=["Users"])
async def generate_confidential_report(
    req: ConfidentialReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate confidential reports that are NOT used for training."""
    user_id = current_user.get("uid")

    doc_ref = firestore_db.collection("documents").document(req.file_hash)
    doc = doc_ref.get()
    if not doc.exists or doc.to_dict().get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Document not found or access denied.")

    from app.services.embedding import query_vectors
    res = query_vectors("", file_id=req.file_hash, top_k=50)
    docs = res.get("documents", [[]])
    if not docs or not docs[0]:
        raise HTTPException(404, "Document not found")

    full_text = "\n\n".join(docs[0])

    if req.report_type == "financial":
        prompt = f"""Generate a confidential financial analysis report for this legal document. Focus on:

1. All financial obligations and costs
2. Payment terms and schedules
3. Penalties and late fees
4. Tax implications
5. Financial risks and liabilities

Document: {full_text}

Confidential Financial Report:"""
    elif req.report_type == "legal_risks":
        prompt = f"""Generate a confidential legal risk assessment for this document. Identify:

1. Potential legal liabilities
2. Compliance risks
3. Contractual obligations
4. Termination risks
5. Dispute resolution procedures
6. Regulatory compliance issues

Document: {full_text}

Confidential Legal Risk Assessment:"""
    elif req.report_type == "compliance":
        prompt = f"""Generate a confidential compliance analysis for this document. Assess:

1. Regulatory compliance requirements
2. Industry-specific regulations
3. Data protection and privacy
4. Reporting obligations
5. Audit requirements
6. Compliance deadlines

Document: {full_text}

Confidential Compliance Report:"""
    else:
        raise HTTPException(400, "Invalid report type. Use: financial, legal_risks, or compliance")

    answer, confidence = call_hf_inference(prompt)

    firestore_db.collection("confidential_reports").add({
        "user_id": user_id,
        "file_hash": req.file_hash,
        "report_type": req.report_type,
        "report": answer,
        "confidence": confidence,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "is_confidential": True,
        "not_for_training": True
    })

    return ConfidentialReportResponse(report=answer, confidence=confidence, is_confidential=True)