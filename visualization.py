"""Drawing functions for annotated output image."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from utils import TOOTH_LABELS, SEGMENT_COLOURS


def draw_segmentation_overlay(vis: np.ndarray, tooth_masks: List[np.ndarray]) -> np.ndarray:
    """Draw semi-transparent fills and hatch lines on a copy, return blended image."""
    overlay = vis.copy()
    h, w = vis.shape[:2]

    for i, mask in enumerate(tooth_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        colour = SEGMENT_COLOURS[i]

        # Fill
        cv2.drawContours(overlay, [cnt], -1, colour, -1)

        # Hatch lines (45°)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        step = max(4, bw // 8)
        for hx in range(bx - bh, bx + bw + bh, step):
            p1 = (hx, by)
            p2 = (hx + bh, by + bh)
            p1 = (max(0, min(p1[0], w - 1)), max(0, min(p1[1], h - 1)))
            p2 = (max(0, min(p2[0], w - 1)), max(0, min(p2[1], h - 1)))
            cv2.line(overlay, p1, p2, colour, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)


def draw_tooth_contours(vis: np.ndarray, tooth_masks: List[np.ndarray]) -> None:
    """Draw crisp white and coloured contours around each tooth."""
    for i, mask in enumerate(tooth_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        colour = SEGMENT_COLOURS[i]
        cv2.drawContours(vis, [cnt], -1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.drawContours(vis, [cnt], -1, colour, 1, cv2.LINE_AA)


def get_label_texts(lab_values: List[Optional[Tuple[float, float, float]]]) -> List[Optional[List[str]]]:
    """Generate label lines for each tooth."""
    tlabels = []
    for i, lab in enumerate(lab_values):
        if lab:
            tlabels.append([TOOTH_LABELS[i], f"L*={lab[0]:.1f}  a*={lab[1]:.1f}  b*={lab[2]:.1f}"])
        else:
            tlabels.append([TOOTH_LABELS[i], "no data"])
    return tlabels


def compute_bounding_boxes(tooth_masks: List[np.ndarray]) -> List[Optional[Tuple[int, int, int, int]]]:
    """Return bounding rect (x, y, w, h) for each tooth mask's largest contour, or None."""
    rects = []
    for mask in tooth_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            rects.append(None)
            continue
        cnt = max(contours, key=cv2.contourArea)
        rects.append(cv2.boundingRect(cnt))
    return rects


def draw_boxes_and_labels(vis: np.ndarray,
                          rects: List[Optional[Tuple[int, int, int, int]]],
                          tlabels: List[Optional[List[str]]]) -> None:
    """
    Draw bounding boxes and text labels with connector lines.
    Assumes vis is the image to draw on.
    """
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.35, min(0.55, w / 2000.0))
    fthick = 1
    line_height = int(cv2.getTextSize("Hg", font, fscale, fthick)[0][1] * 1.8) + 4

    # First draw bounding boxes
    for i, rect in enumerate(rects):
        if rect is None:
            continue
        bx, by, bw, bh = rect
        colour = SEGMENT_COLOURS[i]
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), colour, 2)

    # Compute label box widths
    widths = [0] * 4
    for i in range(4):
        if rects[i] is None or tlabels[i] is None:
            continue
        max_w = 0
        for txt in tlabels[i]:
            ts, _ = cv2.getTextSize(txt, font, fscale, fthick)
            tw = ts[1]
            max_w = max(max_w, tw)
        widths[i] = max_w + 12

    # Horizontal placement (two rows: even indices above, odd below)
    lx = [0] * 4
    gap = 8

    # Row A (indices 0 and 2)
    if rects[0] and rects[2]:
        cx0 = rects[0][0] + rects[0][2] // 2
        cx2 = rects[2][0] + rects[2][2] // 2
        lx0 = cx0 - widths[0] // 2
        lx2 = cx2 - widths[2] // 2
        if lx0 + widths[0] + gap > lx2:
            overlap = (lx0 + widths[0] + gap) - lx2
            lx0 -= overlap // 2
            lx2 += (overlap + 1) // 2
        lx[0] = max(5, min(lx0, w - 5 - widths[0]))
        lx[2] = max(5, min(lx2, w - 5 - widths[2]))
    else:
        for idx in (0, 2):
            if rects[idx]:
                cx = rects[idx][0] + rects[idx][2] // 2
                lx[idx] = max(5, min(cx - widths[idx] // 2, w - 5 - widths[idx]))

    # Row B (indices 1 and 3)
    if rects[1] and rects[3]:
        cx1 = rects[1][0] + rects[1][2] // 2
        cx3 = rects[3][0] + rects[3][2] // 2
        lx1 = cx1 - widths[1] // 2
        lx3 = cx3 - widths[3] // 2
        if lx1 + widths[1] + gap > lx3:
            overlap = (lx1 + widths[1] + gap) - lx3
            lx1 -= overlap // 2
            lx3 += (overlap + 1) // 2
        lx[1] = max(5, min(lx1, w - 5 - widths[1]))
        lx[3] = max(5, min(lx3, w - 5 - widths[3]))
    else:
        for idx in (1, 3):
            if rects[idx]:
                cx = rects[idx][0] + rects[idx][2] // 2
                lx[idx] = max(5, min(cx - widths[idx] // 2, w - 5 - widths[idx]))

    # Vertical positions
    valid_rects = [r for r in rects if r is not None]
    if valid_rects:
        top_y = min(r[1] for r in valid_rects)
    else:
        top_y = int(h * 0.3)

    block_height = line_height * 2 + 4
    for i, rect in enumerate(rects):
        if rect is None or not tlabels[i]:
            continue
        bx, by, bw, bh = rect
        colour = SEGMENT_COLOURS[i]

        is_row_b = (i % 2 == 1)
        stagger = block_height + 4 if is_row_b else 0
        block_bottom = top_y - stagger
        is_below = False
        if block_bottom - block_height < 2:
            block_bottom = by + bh + (block_height + 4) * (2 if is_row_b else 1)
            is_below = True

        # Connector line
        cx_label = lx[i] + widths[i] // 2
        cx_tooth = bx + bw // 2
        if is_below:
            cv2.line(vis, (cx_label, block_bottom - block_height), (cx_tooth, by + bh), colour, 1, cv2.LINE_AA)
        else:
            cv2.line(vis, (cx_label, block_bottom + 2), (cx_tooth, by), colour, 1, cv2.LINE_AA)

        # Draw label texts
        for j, txt in enumerate(reversed(tlabels[i])):
            ty = block_bottom - j * line_height
            tw, th = cv2.getTextSize(txt, font, fscale, fthick)[0]
            cv2.rectangle(vis, (lx[i], ty - th - 3), (lx[i] + tw + 6, ty + 3), (15, 15, 15), -1)
            cv2.putText(vis, txt, (lx[i] + 3, ty), font, fscale, colour, fthick, cv2.LINE_AA)


def draw_divider_lines(vis: np.ndarray, boundaries: List[int], rects: List[Optional[Tuple[int, int, int, int]]]) -> None:
    """Draw dashed cyan vertical lines restricted to teeth vertical span."""
    valid_rects = [r for r in rects if r is not None]
    if not valid_rects:
        return
    y_min = min(r[1] for r in valid_rects)
    y_max = max(r[1] + r[3] for r in valid_rects)
    for gx in boundaries[1:-1]:
        for yy in range(y_min, y_max, 6):
            cv2.line(vis, (gx, yy), (gx, min(yy + 3, y_max)), (0, 220, 220), 1)


def draw_annotated_image(img_bgr: np.ndarray,
                         tooth_masks: List[np.ndarray],
                         lab_values: List[Optional[Tuple[float, float, float]]],
                         boundaries: List[int],
                         output_path: str) -> np.ndarray:
    """Orchestrate all drawing steps and save the result."""
    vis = img_bgr.copy()
    vis = draw_segmentation_overlay(vis, tooth_masks)
    draw_tooth_contours(vis, tooth_masks)

    rects = compute_bounding_boxes(tooth_masks)
    tlabels = get_label_texts(lab_values)
    draw_boxes_and_labels(vis, rects, tlabels)
    draw_divider_lines(vis, boundaries, rects)

    cv2.imwrite(output_path, vis)
    return vis
