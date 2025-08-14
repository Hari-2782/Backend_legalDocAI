from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.routes import upload, qa, feedback, summarize, retrain

app = FastAPI(title="LegalDoc AI Backend")

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

app.include_router(upload.router, prefix="/api")
app.include_router(qa.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(retrain.router, prefix="/api")

