"""
Health check and user/device management routes.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.ml.model_registry import ModelRegistry
from app.schemas.schemas import HealthOut, UserCreate, UserOut
from app.services.sensor_service import get_or_create_user
from app.core.config import settings
from app.utils.ws_manager import ws_manager

router = APIRouter(tags=["Health & Devices"])


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Health check",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthOut:
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return HealthOut(
        status="ok",
        version=settings.VERSION,
        db=db_status,
        models_loaded=ModelRegistry.models_loaded(),
    )


@router.post(
    "/devices",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new wearable device",
)
async def register_device(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    user = await get_or_create_user(db, body.device_id)
    if body.name:
        user.name = body.name
        await db.flush()
    return UserOut.model_validate(user)


@router.get(
    "/ws/stats",
    summary="Active WebSocket connections",
)
async def ws_stats() -> dict:
    return {
        "active_devices": ws_manager.active_devices(),
        "total_connections": ws_manager.connection_count(),
    }
