"""
main.py
-------
Single command-line entry point for the whole system, exposed as:

    python -m src.main <command> [options]

Commands:
    collect    Capture labelled training samples from the webcam
    train      Train the classifier on the collected dataset
    evaluate   Evaluate the trained model (confusion matrix + report)
    recognize  Run real-time recognition from the webcam

Centralizing subcommands here (rather than requiring users to
remember five different scripts) is a usability requirement of the
project: a first-time evaluator only needs to remember one entry
point and `--help`.
"""

import argparse
import sys

from src import config
from src.collect_data import run_collection
from src.evaluate import evaluate as run_evaluate
from src.recognize import run_recognition
from src.train import train_pipeline
from src.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sign-language-recognition",
        description="Sign Language Recognition System (Computer Vision course project).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Collect labelled gesture samples")
    p_collect.add_argument("--label", required=True)
    p_collect.add_argument("--samples", type=int, default=150)
    p_collect.add_argument("--camera", type=int, default=config.CAMERA_INDEX)

    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--model-type", default=config.MODEL_TYPE,
                          choices=["random_forest", "svm", "mlp"])
    p_train.add_argument("--test-size", type=float, default=config.TEST_SPLIT_RATIO)

    p_eval = sub.add_parser("evaluate", help="Evaluate the trained classifier")
    p_eval.add_argument("--output-dir", default=config.MODEL_DIR)

    p_recognize = sub.add_parser("recognize", help="Run real-time recognition")
    p_recognize.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p_recognize.add_argument("--smoothing-window", type=int, default=5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "collect":
            n = run_collection(args.label, args.samples, args.camera)
            print(f"Captured {n} samples for label '{args.label.upper()}'.")

        elif args.command == "train":
            metrics = train_pipeline(args.model_type, args.test_size)
            print(f"Training complete. Test accuracy: {metrics['accuracy']:.2%}")

        elif args.command == "evaluate":
            result = run_evaluate(args.output_dir)
            print(f"Confusion matrix saved to: {result['confusion_matrix_path']}")

        elif args.command == "recognize":
            run_recognition(args.camera, args.smoothing_window)

    except Exception as exc:
        logger.exception("Command '%s' failed: %s", args.command, exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
