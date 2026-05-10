"""
ML model loader and inference interface.

Two pipelines:
  1. ExerciseClassifier  – Random Forest trained on accel/gyro features
  2. InjuryPredictor     – Gradient Boosting trained on EMG/flex/flow/temp/intensity

Both are loaded once at startup and reused across requests.
If trained models are absent, lightweight stub models are used so the
service can run without pre-trained weights (for development/demo).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Exercise label mapping (must match training)
EXERCISE_LABELS = [
    "walking", "running", "push_up", "sit_up", "standing", "resting"
]

MODEL_DIR = Path(settings.MODEL_DIR)


# ---------------------------------------------------------------------------
# Stub models (used when real .pkl files are absent)
# ---------------------------------------------------------------------------

class _StubExerciseClassifier:
    """Rule-based heuristic fallback for exercise classification."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n_samples, 36)
        # Feature index 30 = intensity_mean  (see features.py)
        intensity_mean = X[:, 30]
        labels = []
        for i in intensity_mean:
            if i < 1.5:
                labels.append("resting")
            elif i < 3.0:
                labels.append("standing")
            elif i < 6.0:
                labels.append("walking")
            elif i < 10.0:
                labels.append("running")
            else:
                labels.append("running")
        return np.array(labels)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        proba = []
        for label in preds:
            row = [0.05] * len(EXERCISE_LABELS)
            idx = EXERCISE_LABELS.index(label)
            row[idx] = 0.70
            # distribute remainder
            remainder = 0.30 / (len(EXERCISE_LABELS) - 1)
            for j in range(len(EXERCISE_LABELS)):
                if j != idx:
                    row[j] = remainder
            proba.append(row)
        return np.array(proba)


class _StubInjuryPredictor:
    """Rule-based heuristic fallback for injury/fatigue prediction."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Returns shape (n_samples, 4): [fatigue, strain, injury, overexertion]
        results = []
        for row in X:
            emg_mean = row[0]       # feature index 0
            intensity_mean = row[20] # feature index 20
            temp_mean = row[15]      # feature index 15

            fatigue = min(emg_mean / 400.0, 1.0)
            strain = min(intensity_mean / 15.0, 1.0)
            injury = min((emg_mean / 500.0 + intensity_mean / 20.0) / 2, 1.0)
            overexertion = min(intensity_mean / 12.0, 1.0)

            results.append([fatigue, strain, injury, overexertion])
        return np.clip(np.array(results), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    _exercise_clf = None
    _injury_pred = None
    _loaded = False

    @classmethod
    def load(cls) -> None:
        cls._exercise_clf = cls._load_or_stub(
            MODEL_DIR / "exercise_classifier.pkl",
            _StubExerciseClassifier(),
            "Exercise Classifier",
        )
        cls._injury_pred = cls._load_or_stub(
            MODEL_DIR / "injury_predictor.pkl",
            _StubInjuryPredictor(),
            "Injury Predictor",
        )
        cls._loaded = True

    @staticmethod
    def _load_or_stub(path: Path, stub, name: str):
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded %s from %s", name, path)
            return model
        logger.warning(
            "%s model not found at %s — using stub heuristics. "
            "Run ml_training/train_models.py to generate real models.",
            name, path,
        )
        return stub

    @classmethod
    def models_loaded(cls) -> bool:
        return cls._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def classify_exercise(
        cls, features: np.ndarray
    ) -> Tuple[str, float]:
        """
        Returns (exercise_label, confidence).
        features shape: (36,) — will be reshaped to (1, 36).
        """
        X = features.reshape(1, -1)
        label = cls._exercise_clf.predict(X)[0]
        proba = cls._exercise_clf.predict_proba(X)[0]
        idx = EXERCISE_LABELS.index(label) if label in EXERCISE_LABELS else 0
        confidence = float(proba[idx])
        return str(label), confidence

    @classmethod
    def predict_injury(
        cls, features: np.ndarray
    ) -> dict:
        """
        Returns dict with fatigue, strain_risk, injury_risk, overexertion.
        features shape: (30,) — will be reshaped to (1, 30).
        """
        X = features.reshape(1, -1)
        proba = cls._injury_pred.predict_proba(X)[0]
        # proba shape: (4,) → [fatigue, strain, injury, overexertion]
        return {
            "fatigue_score": float(proba[0]),
            "strain_risk": float(proba[1]),
            "injury_risk": float(proba[2]),
            "overexertion_score": float(proba[3]),
        }


# ---------------------------------------------------------------------------
# Picklable wrapper used by the training script
# (must be importable from this module so unpickling works)
# ---------------------------------------------------------------------------

class WrappedInjuryPredictor:
    """Wraps scikit-learn MultiOutputClassifier → single predict_proba matrix."""
    def __init__(self, model):
        self._m = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [e.predict_proba(X)[:, 1] for e in self._m.estimators_]
        return np.stack(probas, axis=1)
