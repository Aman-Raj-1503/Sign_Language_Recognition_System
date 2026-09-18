"""
collect_data.py
----------------
CLI-driven data collection: opens the webcam, extracts hand landmarks
frame by frame, and appends normalized feature vectors labelled with a
gesture name to the training CSV.

Controls while the capture window is focused:
  SPACE : capture the current frame's landmarks as a labelled sample
  q     : quit collection for this label
"""

import argparse
import sys

import cv2

from src import config
from src.dataset import append_sample
from src.landmark_extractor import LandmarkExtractor
from src.preprocessing import extract_feature_vector
from src.utils import FPSCounter, draw_hud, get_logger, open_camera

logger = get_logger(__name__)


def run_collection(label: str, target_samples: int, camera_index: int = config.CAMERA_INDEX) -> int:
    """
    Run an interactive collection session for a single gesture label.
    Returns the number of samples successfully captured.
    """
    label = label.strip().upper()
    if not label:
        raise ValueError("Label must be a non-empty string.")

    logger.info("Starting data collection for label '%s' (target=%d)", label, target_samples)

    captured = 0
    cap = open_camera(camera_index, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    fps_counter = FPSCounter()

    try:
        with LandmarkExtractor() as extractor:
            while captured < target_samples:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                hand = extractor.process(frame)
                extractor.draw(frame, hand)

                fps = fps_counter.tick()
                draw_hud(frame, [
                    f"Label: {label}  Captured: {captured}/{target_samples}",
                    "SPACE = capture sample   Q = quit",
                    f"FPS: {fps:.1f}",
                ])
                cv2.imshow("Sign Language Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Collection stopped early by user.")
                    break
                if key == ord(" "):
                    if hand is None:
                        logger.warning("No hand detected -- sample not captured.")
                        continue
                    vector = extract_feature_vector(hand.raw_landmarks)
                    if vector is None:
                        logger.warning("Could not build feature vector -- sample skipped.")
                        continue
                    append_sample(vector, label)
                    captured += 1
                    logger.info("Captured sample %d/%d for '%s'", captured, target_samples, label)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


def main():
    parser = argparse.ArgumentParser(description="Collect labelled hand-gesture samples.")
    parser.add_argument("--label", required=True, help="Gesture label, e.g. A, HELLO, THANKS")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to capture")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    args = parser.parse_args()

    try:
        n = run_collection(args.label, args.samples, args.camera)
        print(f"Done. Captured {n} samples for label '{args.label.upper()}'.")
    except Exception as exc:  # surfaced as a clean CLI error, full trace in log file
        logger.exception("Data collection failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
