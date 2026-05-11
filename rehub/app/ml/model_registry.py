"""
ML model loader and inference interface.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EXERCISE_LABELS = [
    "walking", "running", "push_up", "sit_up", "standing", "resting"
]

MODEL_DIR = Path(settings.MODEL_DIR)


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------

class _StubExerciseClassifier:
    classes_ = np.array(EXERCISE_LABELS)

    def predict(self, X: np.ndarray) -> np.ndarray:
        intensity_mean = X[:, 30]
        labels = []
        for i in intensity_mean:
            if i < 1.5:   labels.append("resting")
            elif i < 3.0: labels.append("standing")
            elif i < 6.0: labels.append("walking")
            else:          labels.append("running")
        return np.array(labels)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        proba = []
        for label in preds:
            row = [0.05] * len(EXERCISE_LABELS)
            idx = list(self.classes_).index(label)
            row[idx] = 0.70
            remainder = 0.30 / (len(EXERCISE_LABELS) - 1)
            for j in range(len(EXERCISE_LABELS)):
                if j != idx:
                    row[j] = remainder
            proba.append(row)
        return np.array(proba)


class _StubInjuryPredictor:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            emg_mean       = row[0]
            intensity_mean = row[20]
            fatigue      = min(emg_mean / 400.0, 1.0)
            strain       = min(intensity_mean / 15.0, 1.0)
            injury       = min((emg_mean / 500.0 + intensity_mean / 20.0) / 2, 1.0)
            overexertion = min(intensity_mean / 12.0, 1.0)
            results.append([fatigue, strain, injury, overexertion])
        return np.clip(np.array(results), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    _exercise_clf = None
    _injury_pred  = None
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
        logger.warning("%s not found at %s — using stub.", name, path)
        return stub

    @classmethod
    def models_loaded(cls) -> bool:
        return cls._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def classify_exercise(cls, features: np.ndarray) -> Tuple[str, float]:
        X = features.reshape(1, -1)
        label = str(cls._exercise_clf.predict(X)[0])
        proba = cls._exercise_clf.predict_proba(X)[0]

        # Use the model's own classes_ order — NOT the hardcoded list
        # This is what caused confidence = 0 before
        classes = list(cls._exercise_clf.classes_)
        idx = classes.index(label) if label in classes else 0
        confidence = float(proba[idx])

        return label, confidence

    @classmethod
    def predict_injury(cls, features: np.ndarray) -> dict:
        X = features.reshape(1, -1)
        proba = cls._injury_pred.predict_proba(X)[0]
        return {
            "fatigue_score":    float(proba[0]),
            "strain_risk":      float(proba[1]),
            "injury_risk":      float(proba[2]),
            "overexertion_score": float(proba[3]),
        }


# ---------------------------------------------------------------------------
# Picklable wrapper — must stay in this module for unpickling to work
# ---------------------------------------------------------------------------

class WrappedInjuryPredictor:
    """Wraps MultiOutputClassifier → single (n_samples, 4) predict_proba."""
    def __init__(self, model):
        self._m = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [e.predict_proba(X)[:, 1] for e in self._m.estimators_]
        return np.stack(probas, axis=1)