"""
config.py
---------
Central configuration for the Sign Language Recognition System.
Keeping all tunable parameters in one place makes the system easier
to maintain and to adapt to new gesture sets or hardware.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "landmarks_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_classifier.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_report.json")

for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera / capture settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# MediaPipe Hands settings
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
NUM_LANDMARKS = 21          # MediaPipe hand landmark count
NUM_COORDS = 3               # x, y, z per landmark
FEATURE_VECTOR_LENGTH = NUM_LANDMARKS * NUM_COORDS  # 63

# ---------------------------------------------------------------------------
# Gesture classes (default: static ASL alphabet subset, easily extendable)
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "L",
    "O", "U", "V", "W", "Y", "HELLO", "THANKS", "YES", "NO", "SPACE",
]

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
MODEL_TYPE = "random_forest"   # options: "random_forest", "svm", "mlp"
CONFIDENCE_THRESHOLD = 0.65    # below this, prediction is shown as "Uncertain"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
