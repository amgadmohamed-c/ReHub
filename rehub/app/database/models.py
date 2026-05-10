"""
SQLAlchemy ORM models for ReHub wearable health monitoring system.
Covers raw sensor readings, ML analysis results, alerts, and user sessions.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, ForeignKey, Text, Enum as SAEnum
)
from sqlalchemy.orm import relationship, DeclarativeBase
import enum


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExerciseType(str, enum.Enum):
    WALKING = "walking"
    RUNNING = "running"
    PUSH_UP = "push_up"
    SIT_UP = "sit_up"
    STANDING = "standing"
    RESTING = "resting"


class AlertSeverity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, enum.Enum):
    FATIGUE = "fatigue"
    STRAIN_RISK = "strain_risk"
    INJURY_RISK = "injury_risk"
    OVEREXERTION = "overexertion"
    HIGH_TEMPERATURE = "high_temperature"
    ABNORMAL_MOVEMENT = "abnormal_movement"


# ---------------------------------------------------------------------------
# User / Device
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(64), unique=True, nullable=False, index=True,
                       comment="Unique Arduino / BLE device identifier")
    name = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    sensor_readings = relationship("SensorReading", back_populates="user")
    analyses = relationship("ExerciseAnalysis", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


# ---------------------------------------------------------------------------
# Raw sensor reading (one row per inbound payload)
# ---------------------------------------------------------------------------

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # EMG / flex / flow
    emg = Column(Float, nullable=False, comment="Electromyography raw value")
    flex = Column(Float, nullable=False, comment="Flex sensor (0–1023)")
    flow = Column(Float, nullable=False, comment="Blood-flow / pulse reading")

    # Accelerometer  (m/s²)
    ax = Column(Float, nullable=False)
    ay = Column(Float, nullable=False)
    az = Column(Float, nullable=False)

    # Gyroscope  (°/s)
    gx = Column(Float, nullable=False)
    gy = Column(Float, nullable=False)
    gz = Column(Float, nullable=False)

    # Skin / ambient temperature  (°C)
    temp = Column(Float, nullable=False)

    # Derived on ingestion
    movement_intensity = Column(Float, nullable=False,
                                comment="sqrt(ax²+ay²+az²)")

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="sensor_readings")


# ---------------------------------------------------------------------------
# ML analysis result (produced after every sliding-window batch)
# ---------------------------------------------------------------------------

class ExerciseAnalysis(Base):
    __tablename__ = "exercise_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Window metadata
    window_size = Column(Integer, nullable=False,
                         comment="Number of readings used for this analysis")
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)

    # Exercise classification
    exercise_type = Column(SAEnum(ExerciseType), nullable=False)
    exercise_confidence = Column(Float, nullable=False,
                                 comment="Model confidence 0–1")

    # Injury / health risk predictions
    fatigue_score = Column(Float, nullable=False, comment="0–1 severity")
    strain_risk = Column(Float, nullable=False)
    injury_risk = Column(Float, nullable=False)
    overexertion_score = Column(Float, nullable=False)

    # Aggregate stats for the window
    avg_movement_intensity = Column(Float)
    avg_emg = Column(Float)
    avg_temp = Column(Float)
    max_movement_intensity = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")
    alerts = relationship("Alert", back_populates="analysis")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    analysis_id = Column(Integer, ForeignKey("exercise_analyses.id"),
                         nullable=True, index=True)

    alert_type = Column(SAEnum(AlertType), nullable=False)
    severity = Column(SAEnum(AlertSeverity), nullable=False)
    message = Column(Text, nullable=False)
    score = Column(Float, nullable=True,
                   comment="Numeric score that triggered this alert")

    is_acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="alerts")
    analysis = relationship("ExerciseAnalysis", back_populates="alerts")
