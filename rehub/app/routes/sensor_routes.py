"""
POST /sensors/{device_id}/readings
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.schemas import IngestResponse, SensorPayload
from app.services.sensor_service import ingest_reading

router = APIRouter(prefix="/sensors", tags=["Sensors"])


@router.post(
    "/{device_id}/readings",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a sensor reading from the wearable device",
    description=(
        "The mobile app calls this endpoint after receiving a payload from the "
        "Arduino wearable. The backend stores the raw reading, feeds it into the "
        "sliding-window buffer, and—once the window is full—runs the ML pipelines "
        "and pushes analysis + alerts back via WebSocket."
    ),
)
async def post_reading(
    device_id: str,
    payload: SensorPayload,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    if not device_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id must not be empty",
        )
    return await ingest_reading(db, device_id.strip(), payload)
