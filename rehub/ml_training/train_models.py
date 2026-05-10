#!/usr/bin/env python3
"""
ReHub ML Training Script
========================
Generates synthetic sensor datasets for 6 exercise classes and trains:
  1. ExerciseClassifier  (RandomForestClassifier)
  2. InjuryPredictor     (GradientBoostingClassifier)

Usage:
    python ml_training/train_models.py

Output:
    app/ml/models/exercise_classifier.pkl
    app/ml/models/injury_predictor.pkl
    ml_training/datasets/exercise_dataset.csv
    ml_training/datasets/injury_dataset.csv
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

MODEL_DIR = Path(settings.MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path("ml_training/datasets")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = settings.WINDOW_SIZE
N_SAMPLES_PER_CLASS = 500
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# Import from model_registry so the pickled object unpickles correctly
from app.ml.model_registry import WrappedInjuryPredictor


# ---------------------------------------------------------------------------
# Sensor profiles per exercise (mean, std for each axis)
# ---------------------------------------------------------------------------

EXERCISE_PROFILES = {
    "walking": {
        "ax": (0.3, 1.5), "ay": (-1.5, 1.0), "az": (9.5, 1.5),
        "gx": (0.0, 0.5), "gy": (0.0, 0.4), "gz": (0.0, 0.3),
    },
    "running": {
        "ax": (0.5, 3.0), "ay": (-2.0, 2.5), "az": (9.8, 3.5),
        "gx": (0.0, 1.5), "gy": (0.0, 1.2), "gz": (0.0, 0.8),
    },
    "push_up": {
        "ax": (1.0, 2.0), "ay": (0.0, 1.5), "az": (5.0, 3.0),
        "gx": (0.5, 1.0), "gy": (0.0, 0.5), "gz": (0.2, 0.4),
    },
    "sit_up": {
        "ax": (2.0, 2.5), "ay": (-3.0, 2.0), "az": (7.0, 3.0),
        "gx": (1.0, 1.5), "gy": (0.0, 0.8), "gz": (0.0, 0.5),
    },
    "standing": {
        "ax": (0.1, 0.3), "ay": (-0.5, 0.4), "az": (9.8, 0.5),
        "gx": (0.0, 0.1), "gy": (0.0, 0.1), "gz": (0.0, 0.1),
    },
    "resting": {
        "ax": (0.0, 0.1), "ay": (0.0, 0.1), "az": (9.81, 0.2),
        "gx": (0.0, 0.05), "gy": (0.0, 0.05), "gz": (0.0, 0.05),
    },
}

INJURY_PROFILES = {
    "low": {
        "emg": (80, 20), "flex": (400, 100), "flow": (60, 10),
        "temp": (36.5, 0.3), "intensity": (2.0, 0.8),
    },
    "medium": {
        "emg": (200, 50), "flex": (700, 100), "flow": (100, 20),
        "temp": (37.5, 0.5), "intensity": (7.0, 2.0),
    },
    "high": {
        "emg": (380, 80), "flex": (950, 50), "flow": (140, 30),
        "temp": (38.8, 0.4), "intensity": (14.0, 3.0),
    },
}


# ---------------------------------------------------------------------------
# Simulate a reading window
# ---------------------------------------------------------------------------

def _sim_window(profile: dict, n: int = WINDOW_SIZE) -> dict:
    """Simulate N readings from a profile, return column arrays."""
    data = {}
    for key, (mu, sigma) in profile.items():
        data[key] = np.random.normal(mu, sigma, n)
    return data


def _stats_feats(arr: np.ndarray) -> list:
    return [arr.mean(), arr.std(), arr.min(), arr.max(), arr.max() - arr.min()]


def _zero_crossings(arr: np.ndarray) -> int:
    s = np.sign(arr - arr.mean())
    return int(np.sum(np.diff(s) != 0))


# ---------------------------------------------------------------------------
# Build exercise dataset
# ---------------------------------------------------------------------------

def build_exercise_dataset() -> pd.DataFrame:
    print("Generating exercise dataset ...")
    rows = []
    for label, profile in EXERCISE_PROFILES.items():
        for _ in range(N_SAMPLES_PER_CLASS):
            w = _sim_window(profile)
            ax, ay, az = w["ax"], w["ay"], w["az"]
            gx, gy, gz = w["gx"], w["gy"], w["gz"]
            intensity = np.sqrt(ax**2 + ay**2 + az**2)

            feats = []
            for arr in (ax, ay, az, gx, gy, gz):
                feats.extend(_stats_feats(arr))
            feats.extend([
                _zero_crossings(ax), _zero_crossings(ay), _zero_crossings(az),
                intensity.mean(), intensity.std(), intensity.max(),
            ])
            rows.append(feats + [label])

    cols = []
    for s in ("ax", "ay", "az", "gx", "gy", "gz"):
        for stat in ("mean", "std", "min", "max", "range"):
            cols.append(f"{s}_{stat}")
    cols += ["zc_ax", "zc_ay", "zc_az",
             "intensity_mean", "intensity_std", "intensity_max",
             "label"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Build injury dataset
# ---------------------------------------------------------------------------

def build_injury_dataset() -> pd.DataFrame:
    print("Generating injury dataset ...")
    rows = []
    for risk_level, profile in INJURY_PROFILES.items():
        for _ in range(N_SAMPLES_PER_CLASS):
            w = _sim_window(profile)
            t = np.arange(WINDOW_SIZE, dtype=float)

            feats = []
            for key in ("emg", "flex", "flow", "temp", "intensity"):
                arr = w[key]
                feats.extend(_stats_feats(arr))
                slope = float(np.polyfit(t, arr, 1)[0])
                feats.append(slope)

            # Labels: 4 binary targets
            is_high = risk_level == "high"
            is_med  = risk_level in ("medium", "high")
            rows.append(feats + [
                1 if is_med  else 0,  # fatigue
                1 if is_high else 0,  # strain_risk
                1 if is_high else 0,  # injury_risk
                1 if is_high else 0,  # overexertion
            ])

    cols = []
    for sig in ("emg", "flex", "flow", "temp", "intensity"):
        for stat in ("mean", "std", "min", "max", "range", "slope"):
            cols.append(f"{sig}_{stat}")
    cols += ["fatigue", "strain_risk", "injury_risk", "overexertion"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------

def train_exercise_classifier(df: pd.DataFrame) -> None:
    print("\n--- Training Exercise Classifier ---")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    out_path = MODEL_DIR / "exercise_classifier.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved → {out_path}")


def train_injury_predictor(df: pd.DataFrame) -> None:
    print("\n--- Training Injury Predictor ---")
    target_cols = ["fatigue", "strain_risk", "injury_risk", "overexertion"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].values
    Y = df[target_cols].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE
    )

    base = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, Y_train)

    preds = clf.predict(X_test)
    for i, col in enumerate(target_cols):
        from sklearn.metrics import f1_score
        f1 = f1_score(Y_test[:, i], preds[:, i])
        print(f"  {col}: F1={f1:.3f}")

    wrapped = WrappedInjuryPredictor(clf)
    out_path = MODEL_DIR / "injury_predictor.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(wrapped, f)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ex_df = build_exercise_dataset()
    ex_df.to_csv(DATASET_DIR / "exercise_dataset.csv", index=False)
    print(f"Exercise dataset: {ex_df.shape}")

    inj_df = build_injury_dataset()
    inj_df.to_csv(DATASET_DIR / "injury_dataset.csv", index=False)
    print(f"Injury dataset: {inj_df.shape}")

    train_exercise_classifier(ex_df)
    train_injury_predictor(inj_df)

    print("\n✅  Training complete. Models saved to", MODEL_DIR)
