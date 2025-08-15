## LegalDoc AI Backend

FastAPI backend for uploading, parsing, embedding, and querying legal PDFs with:
- ChromaDB Cloud for vector storage
- Sentence-Transformers for embeddings
- OpenRouter (Claude 3 Haiku) for generation
- Firebase Auth and Firestore for auth, profiles, history, feedback
- Guest-mode (no-login) usage with no persistence

### Requirements
- Python 3.10+
- Firebase project (service account JSON + Web API key)
- ChromaDB Cloud account and API key
- OpenRouter API key

### Setup

1) Install deps
```bash
pip install -r requirements.txt
```

2) Required files
- Place your Firebase service account JSON at `firebase_key.json`

3) Environment (.env)
```env
# OpenRouter (required)
OPENROUTER_API_KEY=sk-or-...

# Firebase (required)
FIREBASE_WEB_API_KEY=AIzaSy...
# Path to service account JSON
FIREBASE_KEY_PATH=./firebase_key.json

# ChromaDB Cloud (required for cloud mode)
CHROMA_CLOUD_API_KEY=ck-...

# Optional Chroma Cloud scoping
CHROMA_CLOUD_HOST=api.trychroma.com
# Optional; leave empty to use defaults on server side
CHROMA_CLOUD_TENANT=
CHROMA_CLOUD_DATABASE=

# General
USE_CHROMA_CLOUD=true
UPLOAD_DIR=./uploads
```

4) Run
```bash
uvicorn app.app:app --reload
# Swagger UI → http://localhost:8000/docs
```

### Auth Model
- Bearer tokens (Firebase ID tokens) via Swagger “Authorize”.
- Email/password login → `/api/users/login` returns `id_token`.
- Google sign-in → `/api/users/google-signin` (verify Google ID token via Firebase).

### Guest vs Registered Users
- Guest users:
  - Can upload and use QA/summarize
  - No history, no feedback, no training
  - File hashes prefixed with `guest_`, metadata in `guest_documents`
- Registered users:
  - Full features: history, feedback, confidential reports
  - Document ownership enforced via `owner_id`
  - Retrain is per-user using non-confidential feedback only

### Endpoints

#### Users & Auth
- GET `/api/users/profile` (Bearer)
- POST `/api/users/profile` (Bearer)
- POST `/api/users/register`
```json
{ "email": "user@example.com", "password": "StrongPass123", "name": "User" }
```
- POST `/api/users/login`
```json
{ "email": "user@example.com", "password": "StrongPass123" }
```
- POST `/api/users/google-signin`
```json
{ "id_token": "GOOGLE_ID_TOKEN" }
```
- GET `/api/users/chat-history?file_hash=optional&limit=20` (Bearer)
- POST `/api/users/confidential-report` (Bearer)
```json
{ "file_hash": "HASH", "report_type": "financial" }
```
  - report_type: `financial` | `legal_risks` | `compliance`

#### Upload & Status
- POST `/api/upload` (Bearer)
- GET `/api/upload/status/{file_hash}` (Bearer)
- DELETE `/api/upload/{file_hash}` (Bearer)

#### Guest (no auth)
- POST `/api/guest/upload`
- GET `/api/guest/status/{file_hash}`
- POST `/api/guest/qa`
```json
{ "file_hash": "guest_HASH", "question": "What is termination?", "top_k": 5 }
```
- POST `/api/guest/summarize`
```json
{ "file_hash": "guest_HASH" }
```

#### QA & Summaries (registered)
- POST `/api/qa` (Bearer)
```json
{ "file_hash": "HASH", "question": "What is termination?", "top_k": 5 }
```
- POST `/api/summarize` (Bearer)
```json
{ "file_hash": "HASH" }
```
- POST `/api/simplify` (Bearer)
```json
{ "file_hash": "HASH" }
```
- POST `/api/compare` (Bearer)
```json
{ "file_hashes": ["HASH1", "HASH2"], "clause_query": "termination clause" }
```
- POST `/api/highlight-evidence` (Bearer)
```json
{ "file_hash": "HASH", "question": "termination clause" }
```

#### Feedback & Retrain
- POST `/api/feedback` (Bearer)
```json
{
  "file_hash": "HASH",
  "chunk_id": "HASH::p4::c2",
  "rating": 4,
  "corrected_output": "Better phrasing",
  "confidential": false
}
```
  - Saves feedback with `user_id` and timestamp
  - Auto-triggers retrain when non-confidential feedback count ≥ 5
- POST `/api/retrain` (Bearer)
  - Triggers user-scoped LoRA retrain

### Notes
- Chroma Cloud Client ensures collection persistence.
- Summarize/Simplify fetch context even when query text is empty.
- Chat history endpoint falls back when Firestore composite index is missing.
- Confidential reports are stored with `not_for_training: true` and excluded from training.
