import sys
import csv
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly, welch
from tqdm import tqdm

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f"Base Dir: {BASE_DIR}")


# Import the new GeneralizingCNN and utils
from train_explain_cnn import GeneralizingCNN, compute_robust_spectrum, get_rpm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 15.0
        self.order_bins = 256
        self.target_fs = 5000
        self.window_sec = 1.0
        self.original_fs = 50000

cfg = InferenceConfig()

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_DIR = BASE_DIR / "data" / "cwru"
MODEL_PATH = BASE_DIR / "src" / "models_gen_robust" / "best_model.pth"
OUTPUT_LOG = BASE_DIR / "cwru_results.csv"


print(DATA_DIR)
print(MODEL_PATH)
print(OUTPUT_LOG)
# --------------------------------------------------
# UTILS
# --------------------------------------------------
def find_signal(mat):
    """Find first 2D ndarray in .mat file"""
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v.squeeze()
    return None

def safe_zscore(sig):
    """Standardize signal along axis 0"""
    mean = sig.mean(0, keepdims=True)
    std = sig.std(0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (sig - mean) / std

def sliding_window_inference(sig_1ch, cfg, model, device, step_sec=0.5):
    """Perform sliding-window inference"""
    win_pts = int(cfg.window_sec * cfg.target_fs)
    step_pts = int(step_sec * cfg.target_fs)
    outputs = []

    # replicate to 3 channels
    sig = np.stack([sig_1ch]*3, axis=1)

    # Resample
    gcd = np.gcd(cfg.original_fs, cfg.target_fs)
    sig = resample_poly(sig, cfg.target_fs//gcd, cfg.original_fs//gcd, axis=0)

    rpm = get_rpm(sig[:, 0], cfg.target_fs)

    for start in range(0, sig.shape[0]-win_pts+1, step_pts):
        segment = sig[start:start+win_pts]
        segment = safe_zscore(segment)
        spec = compute_robust_spectrum(segment, cfg.target_fs, rpm, cfg)
        x_t = torch.tensor(spec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs.append(torch.softmax(model(x_t), dim=1).cpu().numpy())

    if len(outputs) == 0:
        segment = sig[:win_pts]
        segment = safe_zscore(segment)
        spec = compute_robust_spectrum(segment, cfg.target_fs, rpm, cfg)
        x_t = torch.tensor(spec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs.append(torch.softmax(model(x_t), dim=1).cpu().numpy())

    mean_probs = np.mean(outputs, axis=0)
    pred_idx = mean_probs.argmax()
    conf = mean_probs[0, pred_idx]
    return pred_idx, conf, rpm

def label_from_filename(name):
    name = name.lower()
    if "normal" in name: return "Normal"
    elif "imbalance" in name: return "Imbalance"
    elif "horizontal" in name or "vertical" in name: return "Misalignment"
    elif any(k in name for k in ["ball", "outer", "inner", "cage", "overhang"]):
        return "Bearing"
    return "Unknown"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GeneralizingCNN(input_bins=cfg.order_bins).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    files = sorted(DATA_DIR.glob("*.mat"))
    results = []

    # CSV log
    with open(OUTPUT_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "GT", "Pred", "Confidence", "RPM"])

        for file in tqdm(files):
            try:
                mat = loadmat(file)
                sig = find_signal(mat)
                if sig is None:
                    raise ValueError("Signal not found")

                pred_idx, conf, rpm = sliding_window_inference(sig, cfg, model, device)
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