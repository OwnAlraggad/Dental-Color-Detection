"""Enamel mask extraction and upper/lower separation."""

from typing import Optional
import cv2
import numpy as np
from scipy.signal import find_peaks
from utils import ENAMEL_L_MIN, ENAMEL_S_MAX, ENAMEL_A_MAX, ENAMEL_B_MAX, CLOSE_KERNEL_SIZE, OPEN_KERNEL_SIZE


def extract_enamel_mask(img_bgr: np.ndarray,
                        y0: int, y1: int, x0: int, x1: int,
                        mouth_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create binary mask of teeth using LAB+HSV thresholds.
    If mouth_mask is provided, it is used to restrict the result.
    """
    h_img, w_img = img_bgr.shape[:2]
    roi = img_bgr[y0:y1, x0:x1]

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    L = roi_lab[:, :, 0]
    A = roi_lab[:, :, 1]
    B_lab = roi_lab[:, :, 2]
    S = roi_hsv[:, :, 1]

    teeth_mask = (L > ENAMEL_L_MIN) & (S < ENAMEL_S_MAX) & (A < ENAMEL_A_MAX) & (B_lab < ENAMEL_B_MAX)
    enamel_roi = teeth_mask.astype(np.uint8) * 255

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    enamel_roi = cv2.morphologyEx(enamel_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    enamel_roi = cv2.morphologyEx(enamel_roi, cv2.MORPH_OPEN, kernel, iterations=1)

    full_enamel = np.zeros((h_img, w_img), dtype=np.uint8)
    full_enamel[y0:y1, x0:x1] = enamel_roi

    if mouth_mask is not None:
        full_enamel = cv2.bitwise_and(full_enamel, mouth_mask)

    return full_enamel


def find_occlusion_line(enamel_mask: np.ndarray) -> Optional[int]:
    """
    Find horizontal occlusion line between upper and lower teeth using row-sum profile.
    Returns y-coordinate of the line, or None if not found.
    """
    h, w = enamel_mask.shape
    row_sum = enamel_mask.sum(axis=1)

    active_rows = np.where(row_sum > 0)[0]
    if len(active_rows) == 0:
        return None

    r0, r1 = active_rows[0], active_rows[-1]
    profile = row_sum[r0:r1 + 1]

    peaks, _ = find_peaks(profile, distance=15, prominence=profile.max() * 0.15)

    if len(peaks) >= 2:
        peaks = sorted(peaks)
        p1, p2 = peaks[0], peaks[1]
        valley_idx = p1 + np.argmin(profile[p1:p2])
        return r0 + valley_idx
    return None


def separate_upper_teeth(enamel_mask: np.ndarray) -> np.ndarray:
    """
    Keep only upper teeth by zeroing out everything below the occlusion line.
    If no clear line, assumes all teeth are upper.
    """
    occlusion_y = find_occlusion_line(enamel_mask)
    upper_mask = enamel_mask.copy()
    if occlusion_y is not None:
        upper_mask[occlusion_y:, :] = 0
    return upper_mask