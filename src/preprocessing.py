"""
preprocessing.py
-----------------
Converts raw MediaPipe hand landmarks into a normalized, fixed-length
feature vector suitable for a classical ML classifier.

Why normalization matters (this is the core design decision of the
project): raw landmark coordinates are in image space, so the same
gesture looks like a completely different vector depending on how far
the hand is from the camera or where it is in the frame. We remove
that variance by:

  1. Translating all landmarks so the wrist (landmark 0) is the origin.
  2. Scaling by the distance between the wrist and the middle-finger
     MCP joint (landmark 9), which stays roughly proportional to hand
     size regardless of distance from the camera.

This makes the feature vector translation- and scale-invariant, which
is what allows a small classical model (Random Forest / SVM) to
generalize well without needing a large deep-learning model or GPU.
"""

from typing import List, Optional

import numpy as np

from src.config import NUM_LANDMARKS, NUM_COORDS


WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def landmarks_to_array(raw_landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list into an (21, 3) numpy array."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32
    )
    if coords.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(
            f"Expected {NUM_LANDMARKS}x{NUM_COORDS} landmarks, got {coords.shape}"
        )
    return coords


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Apply translation and scale normalization described in the module
    docstring. Returns a flattened (63,) feature vector.
    """
    origin = coords[WRIST_IDX].copy()
    translated = coords - origin

    scale_ref = np.linalg.norm(translated[MIDDLE_MCP_IDX])
    if scale_ref < 1e-6:
        # Degenerate case (all points collapsed); avoid divide-by-zero.
        scale_ref = 1e-6

    normalized = translated / scale_ref
    return normalized.flatten()


def extract_feature_vector(raw_landmarks) -> Optional[np.ndarray]:
    """
    Full pipeline: raw MediaPipe landmarks -> normalized feature vector.
    Returns None (instead of raising) on malformed input so that calling
    code (real-time loops) can skip a bad frame gracefully.
    """
    try:
        coords = landmarks_to_array(raw_landmarks)
        return normalize_landmarks(coords)
    except (ValueError, AttributeError):
        return None


def batch_normalize(list_of_raw_landmarks: List) -> np.ndarray:
    """Vectorized helper used by the training pipeline on stored CSV rows."""
    vectors = [extract_feature_vector(lm) for lm in list_of_raw_landmarks]
    vectors = [v for v in vectors if v is not None]
    return np.array(vectors, dtype=np.float32)
