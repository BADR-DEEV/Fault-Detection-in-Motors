import sys
import scipy.io
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.training.train_explain_cnn import (
    ExplainableCNN,
    compute_order_spectrum,
    safe_zscore
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 6.0
        self.order_bins = 128
        self.target_fs = 4000
        self.window_sec = 1.0
        self.original_fs = 17850  # dataset fs

cfg = InferenceConfig()

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_PATH = BASE_DIR / "data" / "new_data" / "Data.mat"
TARGETS_PATH = BASE_DIR / "data" / "new_data" / "targets.mat"
MODEL_PATH = BASE_DIR / "src" / "models_physics_aware_2" / "best_model_maf_another.pth"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data_mat = scipy.io.loadmat(DATA_PATH)
targets_mat = scipy.io.loadmat(TARGETS_PATH)

X = data_mat["Data"]          # (178, 3571)
y = targets_mat["Targets"]    # (178, 1)

# --------------------------------------------------
# RPM (KNOWN FROM PAPER)
# --------------------------------------------------
RPM = 1900.0

# --------------------------------------------------
# PREPROCESS FUNCTION
# --------------------------------------------------
def preprocess_sample(sig_1ch, cfg):
    # Fake 3-axis (replication)
    sig = np.stack([sig_1ch]*3, axis=1)

    sig = safe_zscore(sig)

    spec = compute_order_spectrum(
        sig,
        fs=cfg.original_fs,
        rpm=RPM,
        cfg=cfg
    )

    return spec

# --------------------------------------------------
# MODEL
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExplainableCNN(input_bins=cfg.order_bins).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------
results = []

for i in tqdm(range(X.shape[0])):
    sig = X[i]

    x = preprocess_sample(sig, cfg)
    x = torch.tensor(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()

    results.append((int(y[i][0]), pred, conf))

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
for gt, pred, conf in results[:10]:
    print(f"GT={gt}, Pred={pred}, Confidence={conf:.3f}")
