from fastapi import APIRouter, BackgroundTasks
from app.services.retrain import trigger_retrain

router = APIRouter()

@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(trigger_retrain)
    return {"status": "retrain job started"}
