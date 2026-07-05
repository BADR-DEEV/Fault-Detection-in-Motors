import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_FREQ_RAW = 50000
TACH_COL = 0
VIBRATION_COLS = [1, 2, 3]
ORDERS_PER_REV = 64
REVOLUTIONS_PER_WINDOW = 8
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FILES_PER_CLASS = 150
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 7


torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DATA_SOURCES = {
    "Normal": {"root": "normal"},
    "Imbalance": {"root": "imbalance"},
    "Horiz_Misalign": {"root": "horizontal-misalignment"},
    "Vert_Misalign": {"root": "vertical-misalignment"},
    "Ball_Fault": {"root": "underhang", "subfolders": ["ball_fault"]},
    "Outer_Race": {"root": "underhang", "subfolders": ["outer_race"]},
}
