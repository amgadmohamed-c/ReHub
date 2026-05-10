"""
Alert generation service.

Evaluates ML predictions and raw window statistics to produce
structured Alert objects stored in PostgreSQL and pushed via WebSocket.
"""

from __future__ import annotations

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.database.models import Alert, AlertSeverity, AlertType, ExerciseAnalysis

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _score_to_severity(score: float) -> AlertSeverity:
    if score >= 0.85:
        return AlertSeverity.CRITICAL
    if score >= 0.70:
        return AlertSeverity.HIGH
    if score >= 0.55:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


# ---------------------------------------------------------------------------
# Main alert builder
# ---------------------------------------------------------------------------

async def generate_alerts(
    db: AsyncSession,
    analysis: ExerciseAnalysis,
    avg_temp: float,
) -> List[Alert]:
    """
    Inspect an ExerciseAnalysis row and create Alert rows if thresholds
    are exceeded. Persists to DB and returns the created alert objects.
    """
    alerts: list[Alert] = []

    checks = [
        (
            AlertType.FATIGUE,
            analysis.fatigue_score,
            settings.FATIGUE_THRESHOLD,
            "High fatigue detected. Consider resting or reducing intensity.",
        ),
        (
            AlertType.STRAIN_RISK,
            analysis.strain_risk,
            settings.STRAIN_THRESHOLD,
            "Muscle strain risk elevated. Check form and reduce load.",
        ),
        (
            AlertType.INJURY_RISK,
            analysis.injury_risk,
            settings.INJURY_THRESHOLD,
            "Injury risk detected. Stop activity if pain is present.",
        ),
        (
            AlertType.OVEREXERTION,
            analysis.overexertion_score,
            settings.OVEREXERTION_THRESHOLD,
            "Overexertion warning. Heart rate and movement intensity are high.",
        ),
    ]

    for alert_type, score, threshold, message in checks:
        if score >= threshold:
            severity = _score_to_severity(score)
            alert = Alert(
                user_id=analysis.user_id,
                analysis_id=analysis.id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                score=score,
            )
            db.add(alert)
            alerts.append(alert)
            logger.info(
                "Alert created: type=%s severity=%s score=%.2f user=%d",
                alert_type.value, severity.value, score, analysis.user_id,
            )

    # Temperature alert (raw value, not ML-derived)
    if avg_temp >= settings.TEMP_HIGH_THRESHOLD:
        severity = AlertSeverity.HIGH if avg_temp >= 39.5 else AlertSeverity.MEDIUM
        alert = Alert(
            user_id=analysis.user_id,
            analysis_id=analysis.id,
            alert_type=AlertType.HIGH_TEMPERATURE,
            severity=severity,
            message=(
                f"Elevated skin temperature ({avg_temp:.1f}°C). "
                "Risk of heat exhaustion. Hydrate and cool down."
            ),
            score=avg_temp,
        )
        db.add(alert)
        alerts.append(alert)

    if alerts:
        await db.flush()  # assign IDs without committing (caller commits)

    return alerts
