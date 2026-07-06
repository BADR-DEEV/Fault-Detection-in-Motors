import os
import sys
from pathlib import Path
import scipy.io
import numpy as np
import torch
import csv
from scipy.signal import resample_poly, welch

# Adjust path to find your source modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from playground.legacy.training.train_explain_cnn import (
    ExplainableCNN,
    compute_order_spectrum,
    safe_zscore
)

# ---------------- CONFIG ----------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 4
        self.order_bins = 128
        self.target_fs = 4000
        self.window_sec = 1.0

cfg = InferenceConfig()
classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]

ORIGINAL_FS = 1000
VARIABLE_NAME = "H"

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data" / "paper_data"   # contains Normal / Faulty
MODEL_PATH = BASE_DIR /  "src" / "models_physics_aware_2" / "best_model_maf_another.pth"
OUTPUT_CSV = BASE_DIR / "inference_results.csv"
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD MODEL --------
model = ExplainableCNN(input_bins=cfg.order_bins).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded")

# -------- HELPER FUNCTION --------
def infer_single_file(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        if VARIABLE_NAME not in mat:
            return None, None, None, "Error"

        sig_raw = mat[VARIABLE_NAME]

        # Shape fixes
        if sig_raw.ndim == 1:
            sig_raw = np.stack([sig_raw]*3, axis=1)
        elif sig_raw.ndim == 2 and sig_raw.shape[1] == 1:
            sig_raw = np.concatenate([sig_raw]*3, axis=1)
        elif sig_raw.shape[0] == 3:
            sig_raw = sig_raw.T

        # Resample
        if ORIGINAL_FS != cfg.target_fs:
            gcd = np.gcd(ORIGINAL_FS, cfg.target_fs)
            sig = resample_poly(
                sig_raw,
                cfg.target_fs // gcd,
                ORIGINAL_FS // gcd,
                axis=0
            )
        else:
            sig = sig_raw

        # RPM estimation
        f, Pxx = welch(sig[:, 0], fs=cfg.target_fs, nperseg=4096)
        mask = (f > 10) & (f < 65)
        rpm = f[mask][np.argmax(Pxx[mask])] * 60 if mask.any() else 0.0

        # Windowing
        win_pts = int(cfg.window_sec * cfg.target_fs)
        if len(sig) < win_pts:
            sig = np.pad(sig, ((0, win_pts-len(sig)), (0,0)), mode="edge")

        start = (len(sig) - win_pts) // 2
        sig = sig[start:start+win_pts]

        # Features
        sig = safe_zscore(sig)
        x = compute_order_spectrum(sig, cfg.target_fs, rpm, cfg)

        x = torch.from_numpy(x).float().unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        return classes[pred_idx], probs[pred_idx].item(), rpm, "OK"

    except Exception as e:
        return None, None, None, "Error"

# -------- MAIN LOOP --------
rows = []

for mat_file in DATA_ROOT.rglob("*.mat"):
    gt = "Healthy" if  mat_file.name[0] =="H" else "Faulty"
    print(mat_file.name[0])

    pred, conf, rpm, status = infer_single_file(mat_file)

    if status != "OK":
        rows.append([
            mat_file.name, gt, "Error", 0.0, 0.0
        ])
        continue

    rows.append([
        mat_file.name,
        gt,
        pred,
        f"{conf:.4f}",
        f"{rpm:.1f}"
    ])

# -------- SAVE CSV --------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["File", "GT", "Pred", "Confidence", "RPM"])
    writer.writerows(rows)

print(f"\n📁 Inference complete. Results saved to:\n{OUTPUT_CSV}")

















# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ---------------- CONFIG ----------------
# BASE_DIR = Path(__file__).resolve().parents[2]  # Adjust if needed
# CSV_FILE = BASE_DIR / "inference_results.csv"
# # ----------------------------------------

# # Load results CSV
# df = pd.read_csv(CSV_FILE)

# # Check data
# print(df.head())

# # Create a summary: counts of predicted classes per ground truth
# summary = df.groupby(["GT", "Pred"]).size().unstack(fill_value=0)

# # Plot
# ax = summary.plot(kind="bar", figsize=(10,6), rot=0)
# plt.title("Prediction Distribution for Healthy vs Faulty Signals")
# plt.xlabel("Ground Truth")
# plt.ylabel("Number of Files")
# plt.legend(title="Predicted Class")
# plt.tight_layout()
# plt.show()
