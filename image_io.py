"""Image loading, saving, and white balance."""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load BGR image from disk."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Save BGR image to disk."""
    cv2.imwrite(path, img)


def white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance.
    Scales each channel so that the global mean becomes neutral gray.
    """
    img = img_bgr.astype(np.float32)
    means = img.mean(axis=(0, 1))          # [B, G, R]
    gray = means.mean()
    scale = gray / (means + 1e-6)
    return np.clip(img * scale, 0, 255).astype(np.uint8)
