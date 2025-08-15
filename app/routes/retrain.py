from fastapi import APIRouter, BackgroundTasks, Depends
from app.services.retrain import trigger_retrain_for_user
from app.auth import get_current_user

router = APIRouter()

@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("uid")
    background_tasks.add_task(trigger_retrain_for_user, user_id)
    return {"status": "retrain job started", "user_id": user_id}
