"""
Unit tests for src/preprocessing.py

These tests use plain Python objects (not a live webcam or MediaPipe)
to check the normalization math in isolation, satisfying the
"testing wherever applicable" requirement without needing camera
hardware in a CI environment.
"""

import numpy as np
import pytest

from src.preprocessing import (
    landmarks_to_array,
    normalize_landmarks,
    extract_feature_vector,
)


class FakeLandmark:
    """Minimal stand-in for a mediapipe landmark point."""
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _make_fake_hand(offset=(0.0, 0.0, 0.0), scale=1.0):
    """Build a deterministic 21-point fake hand for testing."""
    base_points = [(i * 0.01, i * 0.02, 0.0) for i in range(21)]
    return [
        FakeLandmark(
            offset[0] + x * scale,
            offset[1] + y * scale,
            offset[2] + z * scale,
        )
        for (x, y, z) in base_points
    ]


def test_landmarks_to_array_shape():
    hand = _make_fake_hand()
    arr = landmarks_to_array(hand)
    assert arr.shape == (21, 3)


def test_landmarks_to_array_wrong_length_raises():
    hand = _make_fake_hand()[:10]  # too few points
    with pytest.raises(ValueError):
        landmarks_to_array(hand)


def test_normalize_is_translation_invariant():
    hand_a = _make_fake_hand(offset=(0.0, 0.0, 0.0))
    hand_b = _make_fake_hand(offset=(0.3, 0.4, 0.0))  # same hand, shifted

    vec_a = normalize_landmarks(landmarks_to_array(hand_a))
    vec_b = normalize_landmarks(landmarks_to_array(hand_b))

    np.testing.assert_allclose(vec_a, vec_b, atol=1e-5)


def test_normalize_is_scale_invariant():
    hand_a = _make_fake_hand(scale=1.0)
    hand_b = _make_fake_hand(scale=2.5)  # same hand shape, different size

    vec_a = normalize_landmarks(landmarks_to_array(hand_a))
    vec_b = normalize_landmarks(landmarks_to_array(hand_b))

    np.testing.assert_allclose(vec_a, vec_b, atol=1e-5)


def test_extract_feature_vector_returns_none_on_bad_input():
    assert extract_feature_vector([FakeLandmark(0, 0, 0)]) is None
