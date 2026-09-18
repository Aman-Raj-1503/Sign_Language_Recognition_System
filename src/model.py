"""
model.py
--------
Defines the classifier used to map a normalized 63-value hand-landmark
feature vector to a gesture label, plus save/load helpers.

Three classical ML backends are supported (selected via config.MODEL_TYPE):
  - random_forest : robust default, handles non-linear boundaries well,
                     fast to train, interpretable feature importances.
  - svm            : strong baseline for small/medium datasets.
  - mlp            : small neural network, useful if the gesture set
                     grows large and boundaries become more complex.

A classical-ML approach (rather than a CNN on raw pixels) is a
deliberate design choice: because MediaPipe already solves hand
localization and gives us a compact 21-point skeleton, the remaining
classification problem is low-dimensional (63 features) and does not
need a deep network or GPU -- keeping the whole system runnable on a
CPU-only laptop from the command line, which is a project requirement.
"""

import json
import os
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def build_model(model_type: str = config.MODEL_TYPE) -> Any:
    """Factory function returning an unfitted sklearn estimator."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == "svm":
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                    random_state=config.RANDOM_STATE)
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"Unknown MODEL_TYPE '{model_type}'. "
                      "Expected one of: random_forest, svm, mlp.")


def save_model(model, label_encoder: LabelEncoder,
                model_path: str = config.MODEL_PATH,
                encoder_path: str = config.LABEL_ENCODER_PATH) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    logger.info("Model saved to %s", model_path)
    logger.info("Label encoder saved to %s", encoder_path)


def load_model(model_path: str = config.MODEL_PATH,
                encoder_path: str = config.LABEL_ENCODER_PATH):
    if not (os.path.isfile(model_path) and os.path.isfile(encoder_path)):
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.main train` first."
        )
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


def save_metrics(metrics: dict, path: str = config.METRICS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics report written to %s", path)
