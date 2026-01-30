# config.py

CLASSES = [
    "Normal",
    "Imbalance",
    "Misalignment",
    "Bearing"
]

TARGET_FS = 4000
WINDOW_SEC = 1.0

MAX_ORDER = 16.0
ORDER_BINS = 64

# Multi-band order ranges (industry-style)
ORDER_BANDS = [
    (0.5, 2.0),   # imbalance / misalignment
    (2.0, 4.0),   # looseness
    (4.0, 8.0),   # bearing fundamentals
    (8.0, 16.0),  # bearing harmonics
]

OOD_THRESHOLD = 50.0
