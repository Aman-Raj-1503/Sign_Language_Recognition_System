"""
utils.py
--------
Cross-cutting helper utilities shared across the project:
- centralized logging configuration (addresses the "logging/monitoring"
  non-functional requirement)
- an FPS counter for on-screen performance feedback
- small drawing helpers used by the recognition/collection scripts

Keeping these in one module avoids duplicating boilerplate in every
script and keeps each script focused on a single responsibility.
"""

import logging
import time
from logging.handlers import RotatingFileHandler

import cv2

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger that writes to both the console and a
    rotating log file (data/../logs/app.log). Using a rotating handler
    keeps log files from growing without bound, addressing the
    "resource efficiency" non-functional requirement.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured (avoids duplicate handlers on reimport)
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            config.LOG_FILE, maxBytes=1_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Non-fatal: fall back to console-only logging if the file
        # system is read-only or the path is unavailable.
        logger.warning("Could not attach file log handler; using console only.")

    return logger


class FPSCounter:
    """Simple exponential-moving-average FPS counter for on-screen display."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last_time = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._last_time is not None:
            instant_fps = 1.0 / max(now - self._last_time, 1e-6)
            self.fps = (self.smoothing * self.fps) + (1 - self.smoothing) * instant_fps
        self._last_time = now
        return self.fps


def draw_hud(frame, lines, origin=(10, 25), line_height=25, color=(0, 255, 0)):
    """Draw a small heads-up-display text block on a video frame (in place)."""
    x, y = origin
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def open_camera(camera_index: int, width: int, height: int):
    """
    Open a webcam with basic validation. Raises a clear RuntimeError
    instead of letting OpenCV fail silently -- part of the project's
    error-handling strategy.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "Check that a webcam is connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
