"""
landmark_extractor.py
----------------------
Wraps MediaPipe Hands so the rest of the codebase never talks to the
MediaPipe API directly. This is the module that applies the core
Computer Vision concept of the project: locating a hand in an image
and reducing it to a compact geometric representation (21 3D
landmarks) that downstream modules can classify.

Isolating this in one class means MediaPipe could be swapped for a
different hand-tracking backend (e.g. a custom CNN) without touching
preprocessing, training, or the CLI.
"""

from dataclasses import dataclass
from typing import List, Optional

import mediapipe as mp
import numpy as np

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HandLandmarks:
    """Container for a single detected hand's landmarks."""
    raw_landmarks: List        # list of landmark points (.x, .y, .z each)
    handedness: str
    mp_landmark_list: object   # original mediapipe object, kept for drawing


class LandmarkExtractor:
    """Detects a hand in a BGR frame and returns its landmarks."""

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "LandmarkExtractor initialised (max_hands=%s, det_conf=%s, track_conf=%s)",
            max_num_hands, min_detection_confidence, min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """
        Run hand detection on a single BGR frame.
        Returns the first detected hand's landmarks, or None if no
        hand was found in the frame.
        """
        import cv2  # local import keeps module import-order flexible

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = "Unknown"
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        return HandLandmarks(
            raw_landmarks=hand_landmarks.landmark,
            handedness=handedness_label,
            mp_landmark_list=hand_landmarks,
        )

    def draw(self, frame_bgr: np.ndarray, hand_landmarks: Optional[HandLandmarks]) -> np.ndarray:
        """Overlay the hand skeleton on a frame for visual feedback."""
        if hand_landmarks is None:
            return frame_bgr
        self._mp_drawing.draw_landmarks(
            frame_bgr, hand_landmarks.mp_landmark_list, self._mp_hands.HAND_CONNECTIONS
        )
        return frame_bgr

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
