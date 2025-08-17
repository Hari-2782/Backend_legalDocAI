from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware  # ✅ Import
from app.routes import upload, qa, feedback, summarize, retrain, users, guest

app = FastAPI(title="LegalDoc AI Backend")

# ✅ Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

app.include_router(upload.router, prefix="/api")
app.include_router(qa.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(retrain.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(guest.router, prefix="/api")
