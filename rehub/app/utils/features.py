"""
Feature engineering utilities for ReHub ML pipelines.

Transforms a sliding window of raw SensorReading rows into:
  - exercise_features  → used by the exercise classifier
  - injury_features    → used by the injury/fatigue predictor
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from app.database.models import SensorReading


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Readings = Sequence[SensorReading]


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------

def _stats(values: np.ndarray) -> dict:
    """Return mean, std, min, max, range for a 1-D array."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def _zero_crossings(values: np.ndarray) -> int:
    """Count sign changes in a signal (useful for step detection)."""
    signs = np.sign(values - np.mean(values))
    return int(np.sum(np.diff(signs) != 0))


# ---------------------------------------------------------------------------
# Movement intensity
# ---------------------------------------------------------------------------

def movement_intensity(ax: float, ay: float, az: float) -> float:
    """I = sqrt(ax² + ay² + az²)"""
    return math.sqrt(ax ** 2 + ay ** 2 + az ** 2)


# ---------------------------------------------------------------------------
# Exercise features  (accelerometer + gyroscope focused)
# ---------------------------------------------------------------------------

def extract_exercise_features(readings: Readings) -> np.ndarray:
    """
    Returns a 1-D feature vector suitable for the exercise classifier.

    Features (30 total):
      ax, ay, az stats (5 each) = 15
      gx, gy, gz stats (5 each) = 15
      plus: zero_crossing_ax, zero_crossing_ay, zero_crossing_az,
            intensity_mean, intensity_std, intensity_max  → +6
    Total: 36
    """
    ax = np.array([r.ax for r in readings])
    ay = np.array([r.ay for r in readings])
    az = np.array([r.az for r in readings])
    gx = np.array([r.gx for r in readings])
    gy = np.array([r.gy for r in readings])
    gz = np.array([r.gz for r in readings])
    intensity = np.array([r.movement_intensity for r in readings])

    feats: list[float] = []
    for arr in (ax, ay, az, gx, gy, gz):
        s = _stats(arr)
        feats.extend([s["mean"], s["std"], s["min"], s["max"], s["range"]])

    feats.extend([
        _zero_crossings(ax),
        _zero_crossings(ay),
        _zero_crossings(az),
        float(np.mean(intensity)),
        float(np.std(intensity)),
        float(np.max(intensity)),
    ])

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Injury / fatigue features  (EMG + flex + flow + temp + intensity)
# ---------------------------------------------------------------------------

def extract_injury_features(readings: Readings) -> np.ndarray:
    """
    Returns a 1-D feature vector for the injury/fatigue predictor.

    Features:
      emg stats (5), flex stats (5), flow stats (5), temp stats (5),
      intensity stats (5)
      + trend slopes (5 signals × 1 slope) = 5
      Total: 30
    """
    emg = np.array([r.emg for r in readings])
    flex = np.array([r.flex for r in readings])
    flow = np.array([r.flow for r in readings])
    temp = np.array([r.temp for r in readings])
    intensity = np.array([r.movement_intensity for r in readings])

    feats: list[float] = []
    for arr in (emg, flex, flow, temp, intensity):
        s = _stats(arr)
        feats.extend([s["mean"], s["std"], s["min"], s["max"], s["range"]])

    # Linear trend (slope) for each signal — rising EMG over time = fatigue
    t = np.arange(len(readings), dtype=np.float32)
    for arr in (emg, flex, flow, temp, intensity):
        if len(t) > 1:
            slope = float(np.polyfit(t, arr, 1)[0])
        else:
            slope = 0.0
        feats.append(slope)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Window-level aggregate stats for DB storage
# ---------------------------------------------------------------------------

def window_stats(readings: Readings) -> dict:
    intensity = np.array([r.movement_intensity for r in readings])
    emg = np.array([r.emg for r in readings])
    temp = np.array([r.temp for r in readings])
    return {
        "avg_movement_intensity": float(np.mean(intensity)),
        "max_movement_intensity": float(np.max(intensity)),
        "avg_emg": float(np.mean(emg)),
        "avg_temp": float(np.mean(temp)),
    }
