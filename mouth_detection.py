"""Mouth region detection: MediaPipe (primary) and vision-only fallback."""

from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from utils import MOUTH_Y_START, MOUTH_Y_END, MOUTH_WIDTH_CENTRAL_FRACTION, MOUTH_WIDTH_SCAN_FRACTION


def detect_mouth_with_mediapipe(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                              Optional[Tuple[int, int, int, int]],
                                                              Optional[int],
                                                              Optional[int]]:
    """
    Use MediaPipe Face Mesh to extract inner lip polygon.
    Returns:
        mouth_mask: binary mask of the inner lip region
        bbox: (y0, y1, x0, x1) of padded mouth region
        midline_ref_x: x coordinate of the facial midline
        mouth_width: horizontal distance between mouth corners
    If detection fails, returns (None, None, None, None).
    """
    h, w = img_bgr.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4
    ) as face_mesh:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None, None, None, None

        landmarks = results.multi_face_landmarks[0]

        # Inner lip indices (closed polygon)
        inner_lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        pts = np.array([[int(landmarks.landmark[i].x * w),
                         int(landmarks.landmark[i].y * h)] for i in inner_lip_indices], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Bounding box with padding
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad_y = max(int(bh * 0.1), 5)
        pad_x = max(int(bw * 0.1), 5)
        y0 = max(by - pad_y, 0)
        y1 = min(by + bh + pad_y, h - 1)
        x0 = max(bx - pad_x, 0)
        x1 = min(bx + bw + pad_x, w - 1)

        midline_ref_x = int((landmarks.landmark[13].x + landmarks.landmark[14].x) * w / 2)
        mouth_width = int(abs(landmarks.landmark[61].x - landmarks.landmark[291].x) * w)

        return mask, (y0, y1, x0, x1), midline_ref_x, mouth_width


def find_face_bbox(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect face bounding box using YCrCb skin segmentation."""
    h, w = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(float)
    cb = ycrcb[:, :, 2].astype(float)

    skin = ((cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=3)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if n < 2:
        return (0, 0, w, h)

    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]))


def compute_row_teeth_score(img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[int, float]:
    """
    For a given face bounding box, compute a score for each row representing likelihood of teeth.
    Returns (best_row_index_within_cropped_band, best_score).
    """
    fx, fy, fw, fh = face_bbox
    h_img, w_img = img_bgr.shape[:2]

    y0 = fy + int(fh * MOUTH_Y_START)
    y1 = min(fy + int(fh * MOUTH_Y_END), h_img - 1)

    # Central portion of face to avoid cheek bias
    x0_cent = max(fx + int(fw * (0.5 - MOUTH_WIDTH_CENTRAL_FRACTION/2)), 0)
    x1_cent = min(fx + int(fw * (0.5 + MOUTH_WIDTH_CENTRAL_FRACTION/2)), w_img - 1)

    if y1 - y0 < 10 or x1_cent - x0_cent < 10:
        return (h_img // 2, 0.0)

    roi_lab = cv2.cvtColor(img_bgr[y0:y1, x0_cent:x1_cent], cv2.COLOR_BGR2LAB)
    roi_hsv = cv2.cvtColor(img_bgr[y0:y1, x0_cent:x1_cent], cv2.COLOR_BGR2HSV)
    L = roi_lab[:, :, 0].astype(float)
    S = roi_hsv[:, :, 1].astype(float)

    row_score = L.mean(axis=1) / (S.mean(axis=1) + 1.0)
    best_row = int(np.argmax(row_score))
    return best_row, row_score[best_row]


def mouth_band_from_teeth_peak(img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Given face bounding box, locate mouth band around the row with highest teeth score.
    Returns (y_top, y_bot, x_left, x_right) in global coordinates.
    """
    fx, fy, fw, fh = face_bbox
    h_img, w_img = img_bgr.shape[:2]

    best_row, _ = compute_row_teeth_score(img_bgr, face_bbox)
    y0 = fy + int(fh * MOUTH_Y_START)
    half = max(int(fh * 0.08), 25)
    y_top = max(y0 + best_row - half, y0)
    y_bot = min(y0 + best_row + half, fy + int(fh * MOUTH_Y_END))

    # Expanded horizontal scan
    x0_scan = max(fx + int(fw * MOUTH_WIDTH_SCAN_FRACTION[0]), 0)
    x1_scan = min(fx + int(fw * MOUTH_WIDTH_SCAN_FRACTION[1]), w_img - 1)

    return y_top, y_bot, x0_scan, x1_scan


def get_mouth_region_fallback(img_bgr: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Complete fallback detection: returns (y_top, y_bot, x_left, x_right, midline_ref_x, mouth_width).
    """
    face_bbox = find_face_bbox(img_bgr)
    y_top, y_bot, x_left, x_right = mouth_band_from_teeth_peak(img_bgr, face_bbox)
    fx, fy, fw, fh = face_bbox
    midline_ref_x = fx + fw // 2
    mouth_width = int(fw * 0.38)
    return y_top, y_bot, x_left, x_right, midline_ref_x, mouth_width