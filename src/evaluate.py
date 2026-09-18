"""
evaluate.py
-----------
Standalone evaluation of a trained model: reloads the saved model and
the full dataset, re-splits with the same seed for a held-out test
set, and prints/exports a confusion matrix and classification report.

Kept separate from train.py so an evaluator can inspect model quality
without retraining, and so this logic can be reused in unit tests.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CLI/server environments
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src import config
from src.dataset import load_dataset
from src.model import load_model
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate(output_dir: str = config.MODEL_DIR) -> dict:
    model, label_encoder = load_model()
    X, y = load_dataset()
    y_encoded = label_encoder.transform(y)

    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE, stratify=y_encoded,
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix as an image for the project report.
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_encoder.classes_)))
    ax.set_yticks(range(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_, rotation=90)
    ax.set_yticklabels(label_encoder.classes_)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Confusion matrix saved to %s", cm_path)
    logger.info("Evaluation report saved to %s", report_path)

    return {"confusion_matrix_path": cm_path, "report_path": report_path, "report": report}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained gesture classifier.")
    parser.add_argument("--output-dir", default=config.MODEL_DIR)
    args = parser.parse_args()

    try:
        result = evaluate(args.output_dir)
        print(json.dumps(result["report"]["weighted avg"], indent=2))
        print(f"Confusion matrix image: {result['confusion_matrix_path']}")
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
