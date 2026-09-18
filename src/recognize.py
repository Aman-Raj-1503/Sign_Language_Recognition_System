"""
recognize.py
------------
Real-time inference loop: reads webcam frames, extracts and normalizes
hand landmarks, runs the trained classifier, and overlays the
predicted gesture (with confidence) on the video feed.

Predictions below CONFIDENCE_THRESHOLD are shown as "Uncertain" rather
than a potentially wrong label -- a deliberate usability/reliability
trade-off favouring fewer false positives.
"""

import argparse
import sys
from collections import deque

import cv2
import numpy as np

from src import config
from src.landmark_extractor import LandmarkExtractor
from src.model import load_model
from src.preprocessing import extract_feature_vector
from src.utils import FPSCounter, draw_hud, get_logger, open_camera

logger = get_logger(__name__)


def _predict_with_confidence(model, label_encoder, vector: np.ndarray):
    """Return (label, confidence) using predict_proba when available,
    falling back to a fixed confidence of 1.0 for models without it."""
    vector = vector.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        idx = int(model.predict(vector)[0])
        confidence = 1.0
    label = label_encoder.inverse_transform([idx])[0]
    return label, confidence


def run_recognition(camera_index: int = config.CAMERA_INDEX, smoothing_window: int = 5):
    model, label_encoder = load_model()
    logger.info("Loaded model with classes: %s", list(label_encoder.classes_))

    cap = open_camera(camera_index, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    fps_counter = FPSCounter()
    recent_predictions = deque(maxlen=smoothing_window)

    try:
        with LandmarkExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                hand = extractor.process(frame)
                extractor.draw(frame, hand)

                display_label = "No hand detected"
                if hand is not None:
                    vector = extract_feature_vector(hand.raw_landmarks)
                    if vector is not None:
                        label, confidence = _predict_with_confidence(model, label_encoder, vector)
                        recent_predictions.append(label)
                        # Majority vote over the recent window for temporal smoothing
                        stable_label = max(set(recent_predictions), key=recent_predictions.count)
                        if confidence < config.CONFIDENCE_THRESHOLD:
                            display_label = f"Uncertain ({confidence:.0%})"
                        else:
                            display_label = f"{stable_label} ({confidence:.0%})"

                fps = fps_counter.tick()
                draw_hud(frame, [
                    f"Prediction: {display_label}",
                    f"FPS: {fps:.1f}   Q = quit",
                ])
                cv2.imshow("Sign Language Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run real-time sign language recognition.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--smoothing-window", type=int, default=5,
                         help="Number of recent frames used for majority-vote smoothing")
    args = parser.parse_args()

    try:
        run_recognition(args.camera, args.smoothing_window)
    except Exception as exc:
        logger.exception("Recognition failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
