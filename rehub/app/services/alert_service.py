"""
Alert generation service.

Only generates alerts for MEDIUM, HIGH, or CRITICAL severity.
LOW severity alerts are silently ignored.
Alert messages are tailored per exercise type.
"""

from __future__ import annotations
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.database.models import Alert, AlertSeverity, AlertType, ExerciseAnalysis, ExerciseType

logger = get_logger(__name__)


def _score_to_severity(score: float) -> AlertSeverity:
    if score >= 0.85:
        return AlertSeverity.CRITICAL
    if score >= 0.70:
        return AlertSeverity.HIGH
    if score >= 0.55:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


# ---------------------------------------------------------------------------
# Exercise-specific messages
# ---------------------------------------------------------------------------

EXERCISE_MESSAGES = {
    ExerciseType.RUNNING: {
        AlertType.FATIGUE:      "Fatigue detected during running. Slow down or take a walking break.",
        AlertType.STRAIN_RISK:  "Muscle strain risk while running. Reduce pace and check your stride.",
        AlertType.INJURY_RISK:  "Injury risk during running. Stop if you feel any pain.",
        AlertType.OVEREXERTION: "Overexertion during running. Intensity too high — rest now.",
    },
    ExerciseType.WALKING: {
        AlertType.FATIGUE:      "Fatigue detected during walking. Consider sitting down and resting.",
        AlertType.STRAIN_RISK:  "Muscle strain risk during walking. Check your posture.",
        AlertType.INJURY_RISK:  "Injury risk during walking. Slow down and check for discomfort.",
        AlertType.OVEREXERTION: "Overexertion during walking. Reduce pace and hydrate.",
    },
    ExerciseType.PUSH_UP: {
        AlertType.FATIGUE:      "Fatigue detected during push-ups. Stop the set and rest.",
        AlertType.STRAIN_RISK:  "Strain risk during push-ups. Check your wrist and shoulder form.",
        AlertType.INJURY_RISK:  "Injury risk during push-ups. Stop if you feel shoulder or wrist pain.",
        AlertType.OVEREXERTION: "Overexertion during push-ups. Too many reps — take a break.",
    },
    ExerciseType.SIT_UP: {
        AlertType.FATIGUE:      "Fatigue detected during sit-ups. Rest your core.",
        AlertType.STRAIN_RISK:  "Strain risk during sit-ups. Avoid pulling your neck.",
        AlertType.INJURY_RISK:  "Injury risk during sit-ups. Stop if you feel lower back pain.",
        AlertType.OVEREXERTION: "Overexertion during sit-ups. Reduce reps and rest.",
    },
    ExerciseType.STANDING: {
        AlertType.FATIGUE:      "Fatigue detected while standing. Sit down and rest.",
        AlertType.STRAIN_RISK:  "Strain detected while standing. Shift weight or sit down.",
        AlertType.INJURY_RISK:  "Injury risk while standing. Check your posture.",
        AlertType.OVEREXERTION: "Overexertion while standing. Take a seated break.",
    },
    ExerciseType.RESTING: {
        AlertType.FATIGUE:      "High fatigue at rest. Ensure proper sleep and nutrition.",
        AlertType.STRAIN_RISK:  "Strain risk at rest. Muscles may need more recovery time.",
        AlertType.INJURY_RISK:  "Injury risk at rest. Consider consulting a physiotherapist.",
        AlertType.OVEREXERTION: "Overexertion signal at rest. Your body needs full recovery.",
    },
}

DEFAULT_MESSAGES = {
    AlertType.FATIGUE:      "High fatigue detected. Rest or reduce intensity.",
    AlertType.STRAIN_RISK:  "Muscle strain risk elevated. Check form and reduce load.",
    AlertType.INJURY_RISK:  "Injury risk detected. Stop if pain is present.",
    AlertType.OVEREXERTION: "Overexertion warning. Movement intensity too high — rest now.",
}


def _get_message(exercise_type: ExerciseType, alert_type: AlertType) -> str:
    return EXERCISE_MESSAGES.get(exercise_type, DEFAULT_MESSAGES).get(
        alert_type, DEFAULT_MESSAGES[alert_type]
    )


# ---------------------------------------------------------------------------
# Main alert builder
# ---------------------------------------------------------------------------

async def generate_alerts(
    db: AsyncSession,
    analysis: ExerciseAnalysis,
    avg_temp: float,
) -> List[Alert]:
    """
    Generate alerts for MEDIUM, HIGH, CRITICAL severity only.
    LOW severity is ignored. Messages are tailored to the exercise type.
    """
    alerts: list[Alert] = []
    exercise_type = analysis.exercise_type

    checks = [
        (AlertType.FATIGUE,      analysis.fatigue_score,      settings.FATIGUE_THRESHOLD),
        (AlertType.STRAIN_RISK,  analysis.strain_risk,        settings.STRAIN_THRESHOLD),
        (AlertType.INJURY_RISK,  analysis.injury_risk,        settings.INJURY_THRESHOLD),
        (AlertType.OVEREXERTION, analysis.overexertion_score, settings.OVEREXERTION_THRESHOLD),
    ]

    for alert_type, score, threshold in checks:
        if score < threshold:
            continue

        severity = _score_to_severity(score)

        # Only fire MEDIUM and above — skip LOW entirely
        if severity == AlertSeverity.LOW:
            continue

        alert = Alert(
            user_id=analysis.user_id,
            analysis_id=analysis.id,
            alert_type=alert_type,
            severity=severity,
            message=_get_message(exercise_type, alert_type),
            score=round(score, 4),
        )
        db.add(alert)
        alerts.append(alert)
        logger.info(
            "Alert fired: exercise=%s type=%s severity=%s score=%.2f user=%d",
            exercise_type.value, alert_type.value, severity.value, score, analysis.user_id,
        )

    # Temperature — only MEDIUM and above
    if avg_temp >= settings.TEMP_HIGH_THRESHOLD:
        severity = AlertSeverity.CRITICAL if avg_temp >= 39.5 else AlertSeverity.HIGH
        alert = Alert(
            user_id=analysis.user_id,
            analysis_id=analysis.id,
            alert_type=AlertType.HIGH_TEMPERATURE,
            severity=severity,
            message=(
                f"Elevated skin temperature ({avg_temp:.1f}°C) during "
                f"{exercise_type.value.replace('_', ' ')}. "
                "Stop, hydrate, and cool down."
            ),
            score=round(avg_temp, 2),
        )
        db.add(alert)
        alerts.append(alert)

    if alerts:
        await db.flush()

    return alerts