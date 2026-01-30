import sys
import csv
import scipy.io
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly, welch
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
        self.max_order = 5.0
        self.order_bins = 128
        self.target_fs = 4000
        self.window_sec = 1.0
        self.original_fs = 48000  # CWRU standard

cfg = InferenceConfig()

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_DIR = BASE_DIR / "data" / "CWRU"
MODEL_PATH = BASE_DIR / "src"/ "models_physics_aware_2" / "best_model_maf_another.pth"
OUTPUT_LOG = BASE_DIR / "cwru_results.csv"

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def find_signal(mat):
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v.squeeze()
    return None


def estimate_rpm(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=4096)
    mask = (f > 10) & (f < 65)
    return f[mask][np.argmax(Pxx[mask])] * 60


def preprocess(sig_1ch, cfg):
    # Replicate → 3 channels
    sig = np.stack([sig_1ch]*3, axis=1)

    # Resample
    gcd = np.gcd(cfg.original_fs, cfg.target_fs)
    sig = resample_poly(sig,
                        cfg.target_fs // gcd,
                        cfg.original_fs // gcd,
                        axis=0)

    rpm = estimate_rpm(sig[:, 0], cfg.target_fs)

    # Window
    win_pts = int(cfg.window_sec * cfg.target_fs)
    if sig.shape[0] < win_pts:
        sig = np.pad(sig, ((0, win_pts - sig.shape[0]), (0, 0)), mode="edge")

    start = (sig.shape[0] - win_pts) // 2
    sig = sig[start:start + win_pts]

    sig = safe_zscore(sig)
    spec = compute_order_spectrum(sig, cfg.target_fs, rpm, cfg)

    return spec, rpm


def label_from_filename(name):
    if "Normal" in name:
        print(name)
        return "Normal"
    else:
        return "Bearing"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExplainableCNN(input_bins=cfg.order_bins).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    files = sorted(DATA_DIR.glob("*.mat"))
    results = []

    with open(OUTPUT_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "GT", "Pred", "Confidence", "RPM"])

        for file in tqdm(files):
       
            try:
                mat = scipy.io.loadmat(file)
                sig = find_signal(mat)
                if sig is None:
                    raise ValueError("Signal not found")

                x, rpm = preprocess(sig, cfg)
                x = torch.tensor(x).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = probs.argmax().item()
                    conf = probs[0, pred_idx].item()

                classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
                pred = classes[pred_idx]

                gt = label_from_filename(file.name)

            except Exception as e:
                pred, conf, rpm, gt = "Error", 0.0, 0.0, "Unknown"

            writer.writerow([file.name, gt, pred, f"{conf:.4f}", f"{rpm:.1f}"])
            results.append((gt, pred))

    # Summary
    df = pd.DataFrame(results, columns=["GT", "Pred"])
    print("\n=== CWRU CONFUSION MATRIX (%) ===")
    print(pd.crosstab(df["GT"], df["Pred"], normalize="index") * 100)
    print("\nSaved to:", OUTPUT_LOG)


if __name__ == "__main__":
    main()
