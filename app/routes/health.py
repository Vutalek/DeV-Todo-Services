from fastapi import APIRouter

router = APIRouter()


@router.get("/app/v1/heartbeat")
def heartbeat():
    return {"status": "alive"}
