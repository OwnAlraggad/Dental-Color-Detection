"""LAB colour sampling from tooth masks."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from utils import OVEREXP_PCT


def sample_lab(img_bgr: np.ndarray, tooth_masks: List[np.ndarray]) -> List[Optional[Tuple[float, float, float]]]:
    """
    For each tooth mask, compute mean LAB (excluding overexposed pixels).
    Returns list of (L, a, b) or None if insufficient pixels.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    results = []

    for mask in tooth_masks:
        pixels = img_lab[mask == 255]
        if len(pixels) < 30:
            results.append(None)
            continue

        L_pct = pixels[:, 0].astype(np.float32) * 100.0 / 255.0
        valid = pixels[L_pct < OVEREXP_PCT]
        if len(valid) < 10:
            valid = pixels   # fallback – keep all

        mean = valid.astype(np.float32).mean(axis=0)
        results.append((
            round(float(mean[0]) * 100.0 / 255.0, 1),
            round(float(mean[1]) - 128.0, 1),
            round(float(mean[2]) - 128.0, 1)
        ))
    return results