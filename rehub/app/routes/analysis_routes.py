"""
GET /analysis/{device_id}/history
GET /analysis/{device_id}/latest
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ExerciseAnalysis, User
from app.database.session import get_db
from app.schemas.schemas import AnalysisOut

router = APIRouter(prefix="/analysis", tags=["Analysis"])


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
    "/{device_id}/history",
    response_model=List[AnalysisOut],
    summary="Get analysis history for a device",
)
async def get_history(
    device_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[AnalysisOut]:
    user = await _get_user_or_404(db, device_id)
    result = await db.execute(
        select(ExerciseAnalysis)
        .where(ExerciseAnalysis.user_id == user.id)
        .order_by(ExerciseAnalysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    analyses = result.scalars().all()
    return [AnalysisOut.model_validate(a) for a in analyses]


@router.get(
    "/{device_id}/latest",
    response_model=AnalysisOut,
    summary="Get the most recent analysis for a device",
)
async def get_latest(
    device_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisOut:
    user = await _get_user_or_404(db, device_id)
    result = await db.execute(
        select(ExerciseAnalysis)
        .where(ExerciseAnalysis.user_id == user.id)
        .order_by(ExerciseAnalysis.created_at.desc())
        .limit(1)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis found for this device yet",
        )
    return AnalysisOut.model_validate(analysis)
