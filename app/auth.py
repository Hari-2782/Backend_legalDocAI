# app/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth, credentials
import firebase_admin
from app.config import settings

security = HTTPBearer()

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(settings.FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )