"""Vertical boundary detection for splitting upper teeth into four incisors."""

from typing import List

import cv2
import numpy as np
from scipy.signal import find_peaks



def find_tooth_boundaries(upper_teeth_mask: np.ndarray,
                          midline_ref_x: int,
                          mouth_width: int) -> List[int]:
    """
    Locate vertical dividers between the four upper incisors using column-sum profile.
    Returns list of 5 x-coordinates: [left_edge, v_L1, midline, v_R1, right_edge].
    """
    col_sum = upper_teeth_mask.sum(axis=0)
    w_img = upper_teeth_mask.shape[1]

    # Anatomical proportions
    W_c = mouth_width * 0.14   # central incisor half-width approx
    W_l = mouth_width * 0.11   # lateral incisor width approx

    # Smooth profile
    if len(col_sum) > 5:
        profile = np.convolve(col_sum.astype(float), np.ones(5) / 5.0, mode='same')
    else:
        profile = col_sum.astype(float)

    # Find valleys (local minima)
    valleys, _ = find_peaks(-profile, prominence=1.0, distance=5)
    valleys_list = list(valleys)

    # Midline: valley closest to reference
    if valleys_list:
        midline_x = min(valleys_list, key=lambda v: abs(v - midline_ref_x))
    else:
        midline_x = midline_ref_x

    def best_valley_in_range(lo, hi, fallback):
        candidates = [v for v in valleys_list if lo <= v <= hi]
        if candidates:
            expected = (lo + hi) / 2
            return int(min(candidates, key=lambda v: abs(v - expected)))
        return int(fallback)

    v_L1 = best_valley_in_range(midline_x - 1.4 * W_c, midline_x - 0.7 * W_c, midline_x - W_c)
    v_L2 = best_valley_in_range(v_L1 - 1.25 * W_l, v_L1 - 0.75 * W_l, v_L1 - W_l)
    v_R1 = best_valley_in_range(midline_x + 0.7 * W_c, midline_x + 1.4 * W_c, midline_x + W_c)
    v_R2 = best_valley_in_range(v_R1 + 0.7 * W_l, v_R1 + 1.4 * W_l, v_R1 + W_l)

    boundaries = sorted([v_L2, v_L1, midline_x, v_R1, v_R2])
    # Clamp to image width
    boundaries = [max(0, min(x, w_img - 1)) for x in boundaries]

    return boundaries


def build_tooth_masks(upper_teeth_mask: np.ndarray, boundaries: List[int]) -> List[np.ndarray]:
    """
    Split the upper teeth mask into four vertical strips and keep the largest connected component in each.
    Returns list of four binary masks.
    """
    h, w = upper_teeth_mask.shape
    masks = []
    for i in range(4):
        xl = boundaries[i]
        xr = boundaries[i + 1]
        strip = np.zeros((h, w), dtype=np.uint8)
        strip[:, xl:xr] = 255
        tooth_i = cv2.bitwise_and(upper_teeth_mask, strip)

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(tooth_i, connectivity=8)
        if n < 2:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        masks.append((lbl == largest).astype(np.uint8) * 255)
    return masks