"""Main orchestration class for dental color analysis."""

from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from image_io import load_image, white_balance
from mouth_detection import detect_mouth_with_mediapipe, get_mouth_region_fallback
from segmentation import extract_enamel_mask, separate_upper_teeth
from boundary import find_tooth_boundaries, build_tooth_masks
from sampling import sample_lab
from visualization import draw_annotated_image
from utils import TOOTH_LABELS


class DentalColorAnalyzer:
    """Analyzer that detects upper incisors and extracts LAB colours."""

    def __init__(self, apply_white_balance: bool = True):
        self.apply_white_balance = apply_white_balance

    def _locate_mouth_region(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], int, int]:
        """Locate mouth region, return (mouth_mask, (y0,y1,x0,x1), midline_ref_x, mouth_width)."""
        mouth_mask, bbox, midline_ref_x, mouth_width = detect_mouth_with_mediapipe(img)
        if mouth_mask is not None:
            return mouth_mask, bbox, midline_ref_x, mouth_width

        # Fallback
        y_top, y_bot, x_left, x_right, midline_ref_x, mouth_width = get_mouth_region_fallback(img)
        # Create a dummy mouth_mask (None) for fallback
        return None, (y_top, y_bot, x_left, x_right), midline_ref_x, mouth_width

    def analyze(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full analysis on a single image.
        Returns a dictionary with results (tooth LAB values, metadata).
        """
        # Load and optionally white balance
        img = load_image(image_path)
        if self.apply_white_balance:
            img_processed = white_balance(img)
        else:
            img_processed = img.copy()

        # Locate mouth
        mouth_mask, (y0, y1, x0, x1), midline_ref_x, mouth_width = self._locate_mouth_region(img_processed)

        # Segment enamel
        enamel_mask = extract_enamel_mask(img_processed, y0, y1, x0, x1, mouth_mask)
        if cv2.countNonZero(enamel_mask) < 100:
            raise RuntimeError("No teeth detected – is the mouth open and well-lit?")

        upper_teeth_mask = separate_upper_teeth(enamel_mask)
        if cv2.countNonZero(upper_teeth_mask) < 50:
            raise RuntimeError("No upper teeth detected after separation.")

        # Find boundaries and split
        boundaries = find_tooth_boundaries(upper_teeth_mask, midline_ref_x, mouth_width)
        tooth_masks = build_tooth_masks(upper_teeth_mask, boundaries)

        # Sample LAB values
        lab_values = sample_lab(img_processed, tooth_masks)

        # Prepare result dict
        results = {
            "image_path": image_path,
            "white_balance_applied": self.apply_white_balance,
            "tooth_lab_values": {},
            "tooth_pixel_counts": {},
            "average_lab": None,
        }
        valid_labs = []
        for label, lab, mask in zip(TOOTH_LABELS, lab_values, tooth_masks):
            area = int(cv2.countNonZero(mask))
            results["tooth_pixel_counts"][label] = area
            if lab is not None:
                results["tooth_lab_values"][label] = {"L": lab[0], "a": lab[1], "b": lab[2]}
                valid_labs.append(lab)
            else:
                results["tooth_lab_values"][label] = None

        if valid_labs:
            avg = tuple(round(sum(v[k] for v in valid_labs) / len(valid_labs), 1) for k in range(3))
            results["average_lab"] = {"L": avg[0], "a": avg[1], "b": avg[2]}

        # Generate visualization if output path provided
        if output_path:
            draw_annotated_image(img_processed, tooth_masks, lab_values, boundaries, output_path)

        return results