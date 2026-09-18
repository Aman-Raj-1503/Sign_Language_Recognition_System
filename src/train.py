"""
train.py
--------
Trains the gesture classifier on the collected landmark dataset and
saves the fitted model + label encoder + evaluation metrics to disk.
"""

import argparse
import sys

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config
from src.dataset import load_dataset
from src.model import build_model, save_model, save_metrics
from src.utils import get_logger

logger = get_logger(__name__)


def train_pipeline(model_type: str = config.MODEL_TYPE,
                    test_size: float = config.TEST_SPLIT_RATIO) -> dict:
    X, y = load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(set(y_encoded)) < 2:
        raise ValueError(
            "Need at least 2 distinct gesture classes to train. "
            "Collect more labels with `python -m src.main collect`."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size,
        random_state=config.RANDOM_STATE, stratify=y_encoded,
    )

    logger.info("Training %s on %d samples (test set: %d samples)",
                model_type, len(X_train), len(X_test))

    model = build_model(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    logger.info("Test accuracy: %.4f", accuracy)

    save_model(model, label_encoder)
    metrics = {
        "model_type": model_type,
        "accuracy": accuracy,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "classes": list(label_encoder.classes_),
        "classification_report": report,
    }
    save_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the sign-language gesture classifier.")
    parser.add_argument("--model-type", default=config.MODEL_TYPE,
                         choices=["random_forest", "svm", "mlp"])
    parser.add_argument("--test-size", type=float, default=config.TEST_SPLIT_RATIO)
    args = parser.parse_args()

    try:
        metrics = train_pipeline(args.model_type, args.test_size)
        print(f"Training complete. Test accuracy: {metrics['accuracy']:.2%}")
        print(f"Metrics report saved to {config.METRICS_PATH}")
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
