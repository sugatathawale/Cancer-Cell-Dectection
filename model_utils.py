"""Prediction helpers for cancer cell detection (no TensorFlow required at runtime)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

IMG_SIZE = (96, 96)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "cancer_model.keras"
# PCam-style: label is driven by tissue in the center region
CENTER = (32, 64)


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)


def nuclear_density_score(image: Image.Image) -> float:
    """
    Estimate metastatic-like nuclear density from H&E color cues.

    Uses the center 32x32 (same idea as PatchCamelyon labeling): dark /
    purple-blue hematoxylin-stained nuclei raise the score.
    """
    arr = _to_rgb_array(image)
    y0, y1 = CENTER
    patch = arr[y0:y1, y0:y1]
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]

    dark = ((r + g + b) / 3.0) < 110
    hematoxylin = (b >= r * 0.85) & (b > g) & (r < 170)
    nuclei = dark | hematoxylin

    density = float(nuclei.mean())
    gray = (r + g + b) / 3.0
    texture = float(np.clip(np.std(gray) / 80.0, 0.0, 1.0))

    raw = 0.75 * density + 0.25 * texture
    score = 1.0 / (1.0 + np.exp(-12.0 * (raw - 0.28)))
    return float(np.clip(score, 0.0, 1.0))


def create_advanced_model():
    """Same architecture as the advanced model in main.ipynb (local training only)."""
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from tensorflow.keras.models import Sequential

    return Sequential(
        [
            Input(shape=(*IMG_SIZE, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )


def load_cancer_model(path: Optional[Path] = None):
    """Load a saved Keras model if TensorFlow + weights are available."""
    path = path or MODEL_PATH
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model

        return load_model(path)
    except Exception:
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_cancer(image: Image.Image, model: Any = None, threshold: float = 0.5) -> dict:
    """
    Return Yes/No cancer result.

    Uses nuclear-density scoring (works on Streamlit Cloud without TensorFlow).
    If a CNN model is passed, blends both scores.
    """
    feature_score = nuclear_density_score(image)
    cnn_score = None

    if model is not None:
        try:
            batch = preprocess_image(image)
            cnn_score = float(model.predict(batch, verbose=0)[0][0])
        except Exception:
            cnn_score = None

    if cnn_score is None:
        score = feature_score
    else:
        score = 0.65 * feature_score + 0.35 * cnn_score

    has_cancer = score >= threshold
    return {
        "has_cancer": has_cancer,
        "label": "Yes" if has_cancer else "No",
        "confidence": score if has_cancer else 1.0 - score,
        "raw_score": score,
        "feature_score": feature_score,
        "cnn_score": cnn_score,
    }
