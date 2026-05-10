"""
Sensor ingestion service.

Responsibilities:
  1. Resolve or create a User record for the incoming device_id.
  2. Persist the raw SensorReading to the DB.
  3. Push the reading into the sliding window buffer.
  4. If the window is complete → run ML pipelines.
  5. Persist ExerciseAnalysis + Alerts.
  6. Push results over WebSocket.
  7. Return a rich IngestResponse.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.models import (
    Alert, ExerciseAnalysis, ExerciseType, SensorReading, User,
)
from app.ml.model_registry import ModelRegistry
from app.schemas.schemas import AlertOut, AnalysisOut, IngestResponse, SensorPayload
from app.services.alert_service import generate_alerts
from app.utils.features import (
    extract_exercise_features,
    extract_injury_features,
    window_stats,
)
from app.utils.window_buffer import window_buffer
from app.utils.ws_manager import ws_manager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Resolve user
# ---------------------------------------------------------------------------

async def get_or_create_user(db: AsyncSession, device_id: str) -> User:
    result = await db.execute(
        select(User).where(User.device_id == device_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(device_id=device_id)
        db.add(user)
        await db.flush()
        logger.info("Created new user for device_id=%s", device_id)
    return user


# ---------------------------------------------------------------------------
# Ingest a single reading
# ---------------------------------------------------------------------------

async def ingest_reading(
    db: AsyncSession,
    device_id: str,
    payload: SensorPayload,
) -> IngestResponse:
    # 1. Resolve user
    user = await get_or_create_user(db, device_id)

    # 2. Persist raw reading
    reading = SensorReading(
        user_id=user.id,
        emg=payload.emg,
        flex=payload.flex,
        flow=payload.flow,
        ax=payload.ax,
        ay=payload.ay,
        az=payload.az,
        gx=payload.gx,
        gy=payload.gy,
        gz=payload.gz,
        temp=payload.temp,
        movement_intensity=payload.movement_intensity,
    )
    db.add(reading)
    await db.flush()  # get reading.id

    # 3. Push into sliding window
    window: Optional[list] = await window_buffer.add(device_id, reading)
    buf_size = await window_buffer.buffer_size(device_id)

    analysis_out: Optional[AnalysisOut] = None
    alert_outs: List[AlertOut] = []
    analysis_triggered = False

    # 4. ML pipeline (only when window is full)
    if window:
        analysis_triggered = True
        analysis_out, alert_outs = await _run_ml_pipeline(db, user, window)

        # 5. Push over WebSocket
        await _push_ws(device_id, analysis_out, alert_outs)

    return IngestResponse(
        reading_id=reading.id,
        device_id=device_id,
        movement_intensity=payload.movement_intensity,
        buffer_size=buf_size,
        analysis_triggered=analysis_triggered,
        analysis=analysis_out,
        alerts=alert_outs,
    )


# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------

async def _run_ml_pipeline(
    db: AsyncSession,
    user: User,
    window: list,
) -> tuple[AnalysisOut, List[AlertOut]]:
    """Feature extraction → classification → prediction → alert generation."""

    # Feature extraction
    ex_feats = extract_exercise_features(window)
    inj_feats = extract_injury_features(window)
    stats = window_stats(window)

    # ML inference
    exercise_label, confidence = ModelRegistry.classify_exercise(ex_feats)
    injury_scores = ModelRegistry.predict_injury(inj_feats)

    # Map string label → Enum
    try:
        exercise_enum = ExerciseType(exercise_label)
    except ValueError:
        exercise_enum = ExerciseType.RESTING

    # Persist analysis
    analysis = ExerciseAnalysis(
        user_id=user.id,
        window_size=len(window),
        window_start=window[0].timestamp,
        window_end=window[-1].timestamp,
        exercise_type=exercise_enum,
        exercise_confidence=confidence,
        fatigue_score=injury_scores["fatigue_score"],
        strain_risk=injury_scores["strain_risk"],
        injury_risk=injury_scores["injury_risk"],
        overexertion_score=injury_scores["overexertion_score"],
        **stats,
    )
    db.add(analysis)
    await db.flush()

    # Generate alerts
    avg_temp = stats.get("avg_temp", 0.0) or 0.0
    alerts: List[Alert] = await generate_alerts(db, analysis, avg_temp)

    logger.info(
        "Analysis id=%d exercise=%s conf=%.2f fatigue=%.2f alerts=%d",
        analysis.id, exercise_label, confidence,
        injury_scores["fatigue_score"], len(alerts),
    )

    analysis_out = AnalysisOut.model_validate(analysis)
    alert_outs = [AlertOut.model_validate(a) for a in alerts]
    return analysis_out, alert_outs


# ---------------------------------------------------------------------------
# WebSocket push
# ---------------------------------------------------------------------------

async def _push_ws(
    device_id: str,
    analysis: AnalysisOut,
    alerts: List[AlertOut],
) -> None:
    payload = {
        "event": "analysis",
        "device_id": device_id,
        "timestamp": datetime.utcnow(),
        "analysis": analysis.model_dump(),
        "alerts": [a.model_dump() for a in alerts],
    }
    await ws_manager.send_to_device(device_id, payload)
