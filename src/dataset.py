"""
dataset.py
----------
Handles persistence of the labelled landmark dataset used to train the
classifier. The dataset is stored as a single CSV file where each row
is one normalized 63-value feature vector plus its gesture label.

Using a flat CSV (rather than a database) keeps the project runnable
anywhere with zero setup, which matters for a CLI tool that must be
easy for an evaluator to execute end-to-end.
"""

import csv
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [f"f{i}" for i in range(config.FEATURE_VECTOR_LENGTH)]
CSV_HEADER = FEATURE_COLUMNS + ["label"]


def append_sample(feature_vector: np.ndarray, label: str, csv_path: str = config.DATASET_CSV) -> None:
    """Append one labelled sample to the dataset CSV, creating it (with a
    header) on first write."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(list(feature_vector) + [label])


def load_dataset(csv_path: str = config.DATASET_CSV) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset CSV into (X, y) numpy arrays.
    Raises FileNotFoundError with a clear message if no data has been
    collected yet -- this is surfaced to the CLI user rather than a
    raw pandas traceback.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No dataset found at {csv_path}. "
            "Run `python -m src.main collect --label <GESTURE>` first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at {csv_path} is empty.")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    logger.info("Loaded dataset: %d samples across %d classes", len(y), len(set(y)))
    return X, y


def class_distribution(csv_path: str = config.DATASET_CSV) -> "pd.Series":
    """Return a count of samples per class -- used to warn about class imbalance."""
    df = pd.read_csv(csv_path)
    return df["label"].value_counts()
