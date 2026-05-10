"""
GET  /alerts/{device_id}
POST /alerts/{device_id}/acknowledge
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Alert, User
from app.database.session import get_db
from app.schemas.schemas import AlertAckRequest, AlertOut

router = APIRouter(prefix="/alerts", tags=["Alerts"])


async def _get_user_or_404(db: AsyncSession, device_id: str) -> User:
    result = await db.execute(
        select(User).where(User.device_id == device_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found",
        )
    return user


@router.get(
    "/{device_id}",
    response_model=List[AlertOut],
    summary="Get alerts for a device",
)
async def get_alerts(
    device_id: str,
    unacknowledged_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> List[AlertOut]:
    user = await _get_user_or_404(db, device_id)
    q = select(Alert).where(Alert.user_id == user.id)
    if unacknowledged_only:
        q = q.where(Alert.is_acknowledged == False)  # noqa: E712
    q = q.order_by(Alert.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return [AlertOut.model_validate(a) for a in result.scalars().all()]


@router.post(
    "/{device_id}/acknowledge",
    summary="Acknowledge one or more alerts",
    status_code=status.HTTP_200_OK,
)
async def acknowledge_alerts(
    device_id: str,
    body: AlertAckRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    user = await _get_user_or_404(db, device_id)
    await db.execute(
        update(Alert)
        .where(Alert.user_id == user.id, Alert.id.in_(body.alert_ids))
        .values(is_acknowledged=True)
    )
    return {"acknowledged": len(body.alert_ids)}
