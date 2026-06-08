"""Common utilities and constants."""

# Tooth labels (left to right)
TOOTH_LABELS = [
    "UL Lateral",   # upper-left lateral incisor
    "UL Central",   # upper-left central incisor
    "UR Central",   # upper-right central incisor
    "UR Lateral",   # upper-right lateral incisor
]

# Colour palette for visualisation (BGR)
SEGMENT_COLOURS = [(34, 197, 94), (52, 211, 153), (16, 185, 129), (5, 150, 105)]

# Enamel thresholding (LAB & HSV)
ENAMEL_L_MIN = 145
ENAMEL_S_MAX = 70
ENAMEL_A_MAX = 136
ENAMEL_B_MAX = 140

# Overexposure cut-off for LAB sampling (L* percentage)
OVEREXP_PCT = 92.0

# Morphology kernel sizes
CLOSE_KERNEL_SIZE = 3
OPEN_KERNEL_SIZE = 3

# Fallback mouth band proportions
MOUTH_Y_START = 0.25
MOUTH_Y_END = 0.80
MOUTH_WIDTH_CENTRAL_FRACTION = 0.30   # central 30% of face for row scoring
MOUTH_WIDTH_SCAN_FRACTION = (0.15, 0.85)  # expanded scan for final segmentation