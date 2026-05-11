"""
POST /api/v1/simulate/{device_id}

Accepts an exercise type + intensity level and generates a full realistic
sensor payload (including faked EMG) then runs it through the complete
ingestion + ML pipeline, returning analysis and alerts just like real data.

Useful for:
  - Testing without hardware
  - Bypassing broken EMG sensor readings
  - Demoing the system
"""

import random
import math
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.schemas import IngestResponse, SensorPayload
from app.services.sensor_service import ingest_reading

router = APIRouter(prefix="/simulate", tags=["Simulation"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

ExerciseTypeStr = Literal[
    "walking", "running", "push_up", "sit_up", "standing", "resting"
]

IntensityLevel = Literal["low", "medium", "high"]


class SimulateRequest(BaseModel):
    exercise_type: ExerciseTypeStr = Field(
        ..., description="The exercise being performed"
    )
    intensity: IntensityLevel = Field(
        ..., description="low / medium / high — controls EMG, movement, and temp"
    )
    num_readings: int = Field(
        default=1,
        ge=1,
        le=100,
        description="How many readings to inject (useful to fill the window faster)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "exercise_type": "running",
            "intensity": "high",
            "num_readings": 50
        }
    }}


# ---------------------------------------------------------------------------
# Sensor profiles per exercise × intensity
# ---------------------------------------------------------------------------

# Structure: exercise → intensity → { sensor: (mean, std) }
PROFILES = {
    "walking": {
        "low":    dict(ax=(0.2,0.5), ay=(-1.2,0.5), az=(9.6,0.5), gx=(0,.2), gy=(0,.2), gz=(0,.1), emg=(60,10),  flex=(350,50), flow=(55,8),  temp=(36.4,.2)),
        "medium": dict(ax=(0.3,1.2), ay=(-1.5,0.8), az=(9.5,1.0), gx=(0,.4), gy=(0,.3), gz=(0,.2), emg=(110,20), flex=(500,60), flow=(70,10), temp=(36.8,.3)),
        "high":   dict(ax=(0.5,2.0), ay=(-2.0,1.5), az=(9.3,1.8), gx=(0,.8), gy=(0,.6), gz=(0,.4), emg=(180,30), flex=(650,70), flow=(90,12), temp=(37.2,.3)),
    },
    "running": {
        "low":    dict(ax=(0.4,1.5), ay=(-1.8,1.2), az=(9.5,1.5), gx=(0,.6), gy=(0,.5), gz=(0,.3), emg=(140,25), flex=(550,60), flow=(95,15),  temp=(37.0,.3)),
        "medium": dict(ax=(0.6,2.5), ay=(-2.5,2.0), az=(9.8,2.5), gx=(0,1.2), gy=(0,1.0), gz=(0,.6), emg=(220,35), flex=(720,70), flow=(120,18), temp=(37.6,.4)),
        "high":   dict(ax=(1.0,3.5), ay=(-3.5,3.0), az=(9.8,3.5), gx=(0,1.8), gy=(0,1.5), gz=(0,.9), emg=(320,50), flex=(880,60), flow=(150,20), temp=(38.2,.4)),
    },
    "push_up": {
        "low":    dict(ax=(0.8,1.0), ay=(0.0,0.8), az=(5.5,2.0), gx=(.3,.5), gy=(0,.3), gz=(.1,.2), emg=(150,25), flex=(600,80), flow=(65,10),  temp=(36.6,.2)),
        "medium": dict(ax=(1.2,1.8), ay=(0.0,1.2), az=(5.0,2.5), gx=(.6,.8), gy=(0,.4), gz=(.2,.3), emg=(250,40), flex=(780,70), flow=(80,12),  temp=(37.0,.3)),
        "high":   dict(ax=(1.8,2.5), ay=(0.0,1.8), az=(4.5,3.0), gx=(1.0,1.2), gy=(0,.6), gz=(.3,.4), emg=(360,55), flex=(920,50), flow=(100,15), temp=(37.5,.4)),
    },
    "sit_up": {
        "low":    dict(ax=(1.5,1.2), ay=(-2.0,1.0), az=(7.5,2.0), gx=(.5,.6), gy=(0,.4), gz=(0,.2), emg=(130,20), flex=(580,70), flow=(60,10),  temp=(36.5,.2)),
        "medium": dict(ax=(2.0,2.0), ay=(-3.0,1.8), az=(7.0,2.5), gx=(1.0,1.0), gy=(0,.6), gz=(0,.3), emg=(220,35), flex=(750,65), flow=(75,12),  temp=(36.9,.3)),
        "high":   dict(ax=(2.8,2.8), ay=(-4.0,2.5), az=(6.5,3.0), gx=(1.5,1.4), gy=(0,.8), gz=(0,.4), emg=(330,50), flex=(900,55), flow=(95,15),  temp=(37.4,.4)),
    },
    "standing": {
        "low":    dict(ax=(0.05,.1), ay=(-0.3,.2), az=(9.8,.3), gx=(0,.05), gy=(0,.05), gz=(0,.05), emg=(30,8),  flex=(200,30), flow=(50,8),  temp=(36.3,.2)),
        "medium": dict(ax=(0.1,.2),  ay=(-0.5,.3), az=(9.8,.4), gx=(0,.1),  gy=(0,.1),  gz=(0,.08), emg=(50,10), flex=(280,40), flow=(55,8),  temp=(36.5,.2)),
        "high":   dict(ax=(0.15,.3), ay=(-0.7,.4), az=(9.8,.5), gx=(0,.15), gy=(0,.12), gz=(0,.1),  emg=(70,12), flex=(340,45), flow=(60,10), temp=(36.7,.2)),
    },
    "resting": {
        "low":    dict(ax=(0,.05),  ay=(0,.05),  az=(9.81,.1), gx=(0,.02), gy=(0,.02), gz=(0,.02), emg=(20,5),  flex=(150,20), flow=(45,5),  temp=(36.2,.1)),
        "medium": dict(ax=(0,.08),  ay=(0,.08),  az=(9.81,.15),gx=(0,.03), gy=(0,.03), gz=(0,.03), emg=(30,8),  flex=(180,25), flow=(48,6),  temp=(36.3,.1)),
        "high":   dict(ax=(0,.1),   ay=(0,.1),   az=(9.81,.2), gx=(0,.05), gy=(0,.05), gz=(0,.04), emg=(40,10), flex=(210,30), flow=(52,7),  temp=(36.4,.2)),
    },
}


def _r(mean: float, std: float) -> float:
    return round(random.gauss(mean, std), 6)


def generate_payload(exercise_type: str, intensity: str) -> SensorPayload:
    """Generate one realistic SensorPayload for the given exercise + intensity."""
    p = PROFILES[exercise_type][intensity]
    return SensorPayload(
        emg=max(0,   _r(*p["emg"])),
        flex=max(0,  min(1023, _r(*p["flex"]))),
        flow=max(0,  _r(*p["flow"])),
        ax=_r(*p["ax"]),
        ay=_r(*p["ay"]),
        az=_r(*p["az"]),
        gx=_r(*p["gx"]),
        gy=_r(*p["gy"]),
        gz=_r(*p["gz"]),
        temp=_r(*p["temp"]),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{device_id}",
    response_model=list[IngestResponse],
    summary="Simulate sensor readings for a given exercise and intensity",
    description=(
        "Generates realistic fake sensor data (including EMG) based on the "
        "exercise type and intensity level, then runs it through the full "
        "ingestion + ML pipeline. Use this to test or demo without hardware."
    ),
)
async def simulate_readings(
    device_id: str,
    body: SimulateRequest,
    db: AsyncSession = Depends(get_db),
) -> list[IngestResponse]:
    responses = []
    for _ in range(body.num_readings):
        payload = generate_payload(body.exercise_type, body.intensity)
        result = await ingest_reading(db, device_id.strip(), payload)
        responses.append(result)
    return responses