"""
Unit tests for src/model.py using a tiny synthetic dataset, so the
full train -> save -> load -> predict path is checked without needing
the real webcam-collected dataset.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.model import build_model, save_model, load_model


def _toy_dataset():
    rng = np.random.default_rng(0)
    class_a = rng.normal(loc=0.0, scale=0.1, size=(30, 63))
    class_b = rng.normal(loc=5.0, scale=0.1, size=(30, 63))
    X = np.vstack([class_a, class_b]).astype(np.float32)
    y = np.array(["A"] * 30 + ["B"] * 30)
    return X, y


def test_build_model_default():
    model = build_model("random_forest")
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_train_save_load_predict_roundtrip(tmp_path):
    X, y = _toy_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = build_model("random_forest")
    model.fit(X, y_encoded)

    model_path = tmp_path / "model.joblib"
    encoder_path = tmp_path / "encoder.joblib"
    save_model(model, encoder, str(model_path), str(encoder_path))

    loaded_model, loaded_encoder = load_model(str(model_path), str(encoder_path))
    preds = loaded_model.predict(X)
    labels = loaded_encoder.inverse_transform(preds)

    accuracy = (labels == y).mean()
    assert accuracy > 0.9  # well-separated synthetic classes should be easy
