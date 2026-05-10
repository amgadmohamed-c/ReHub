"""
Pydantic v2 schemas for request/response validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field
import math

from app.database.models import AlertSeverity, AlertType, ExerciseType


# ---------------------------------------------------------------------------
# Sensor ingestion
# ---------------------------------------------------------------------------

class SensorPayload(BaseModel):
    """Payload sent by the mobile app from the Arduino wearable."""

    emg: float = Field(..., ge=0, description="EMG raw value")
    flex: float = Field(..., ge=0, le=1023, description="Flex sensor 0–1023")
    flow: float = Field(..., ge=0, description="Blood-flow / pulse sensor")
    ax: float = Field(..., description="Accelerometer X (m/s²)")
    ay: float = Field(..., description="Accelerometer Y (m/s²)")
    az: float = Field(..., description="Accelerometer Z (m/s²)")
    gx: float = Field(..., description="Gyroscope X (°/s)")
    gy: float = Field(..., description="Gyroscope Y (°/s)")
    gz: float = Field(..., description="Gyroscope Z (°/s)")
    temp: float = Field(..., description="Temperature (°C)")

    @computed_field  # type: ignore[misc]
    @property
    def movement_intensity(self) -> float:
        """I = sqrt(ax² + ay² + az²)"""
        return math.sqrt(self.ax ** 2 + self.ay ** 2 + self.az ** 2)

    model_config = {"json_schema_extra": {
        "example": {
            "emg": 114, "flex": 1023, "flow": 0,
            "ax": 0.299275, "ay": -1.482011, "az": 11.02051,
            "gx": -0.083403, "gy": -0.022649, "gz": 0.012257,
            "temp": 30.03588,
        }
    }}


class SensorReadingOut(BaseModel):
    id: int
    user_id: int
    emg: float
    flex: float
    flow: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    temp: float
    movement_intensity: float
    timestamp: datetime

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    """Returned immediately after a reading is stored."""
    reading_id: int
    device_id: str
    movement_intensity: float
    buffer_size: int          # current readings in the sliding window
    analysis_triggered: bool  # True when a window was full and ML ran
    analysis: Optional[AnalysisOut] = None
    alerts: List[AlertOut] = []


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalysisOut(BaseModel):
    id: int
    user_id: int
    window_size: int
    window_start: datetime
    window_end: datetime
    exercise_type: ExerciseType
    exercise_confidence: float
    fatigue_score: float
    strain_risk: float
    injury_risk: float
    overexertion_score: float
    avg_movement_intensity: Optional[float]
    avg_emg: Optional[float]
    avg_temp: Optional[float]
    max_movement_intensity: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertOut(BaseModel):
    id: int
    user_id: int
    analysis_id: Optional[int]
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    score: Optional[float]
    is_acknowledged: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertAckRequest(BaseModel):
    alert_ids: List[int]


# ---------------------------------------------------------------------------
# Users / Devices
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    device_id: str = Field(..., min_length=3, max_length=64)
    name: Optional[str] = None


class UserOut(BaseModel):
    id: int
    device_id: str
    name: Optional[str]
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# WebSocket messages
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    """Generic WebSocket push message."""
    event: str          # e.g. "analysis", "alert", "ack"
    device_id: str
    payload: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthOut(BaseModel):
    status: str
    version: str
    db: str
    models_loaded: bool


# Resolve forward reference
IngestResponse.model_rebuild()
