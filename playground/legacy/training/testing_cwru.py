import sys
import csv
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly, welch
from tqdm import tqdm

# IMPORTANT: Match EXACTLY the architecture used during training
from playground.legacy.training.RES_CNN import ResNet1D  # Ensure this matches your training script path

# --------------------------------------------------
# CORRECTED CONFIG (MUST MATCH TRAINING PARAMETERS)
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        # CRITICAL: Must match training config EXACTLY
        self.max_order = 10.0   # Was 6.0 - must be 10.0 to match training
        self.order_bins = 512   # Was 128 - must be 512 to match training input size
        self.original_fs = 12000
        self.model_fs = 4000
        self.window_sec = 1.0
        self.classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]

cfg = InferenceConfig()

# --------------------------------------------------
# FIXED UTILS (Critical shape handling)
# --------------------------------------------------
def safe_zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def find_signal(mat):
    """Robustly extract 1D vibration signal from CWRU .mat files"""
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.size > 0:
            # Handle all common CWRU storage formats:
            # - Column vector (N, 1) → squeeze to (N,)
            # - Row vector (1, N) → squeeze to (N,)
            # - True 1D (N,) → keep as-is
            v = np.squeeze(v)
            if v.ndim == 1:
                return v
            elif v.ndim == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
                return v.flatten()
            # Fallback: take first channel if multi-channel
            elif v.ndim == 2:
                return v[:, 0].flatten()
    return None

def compute_order_spectrum(sig, fs, rpm, cfg):
    """Fixed version matching training pipeline EXACTLY"""
    sig = np.asarray(sig)
    
    # Normalize to (N, C) format - CRITICAL FIX
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]  # (N,) → (N, 1)
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1]:
        sig = sig.T  # (C, N) → (N, C)
    
    # Handle invalid RPM (matches training behavior)
    if rpm <= 100:
        rpm = 1500.0
    shaft_hz = rpm / 60.0

    # Welch transform - MATCH TRAINING EXACTLY (no return_onesided override)
    f, Pxx = welch(
        sig,
        fs=fs,
        nperseg=min(len(sig), 1024),
        axis=0  # Compute spectrum along time axis
    )
    
    # CRITICAL FIX: Ensure Pxx is always (n_freq, n_channels)
    if Pxx.ndim == 1:
        Pxx = Pxx[:, np.newaxis]  # (n_freq,) → (n_freq, 1)
    
    # Convert Hz → Orders
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    # Interpolate each channel to target orders
    specs = []
    for ch in range(Pxx.shape[1]):
        # FIXED: Pxx[:, ch] is now guaranteed 1D
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        
        # Log scaling + normalization (matches training)
        s_log = np.log(s + 1e-12)
        s_norm = (s_log - s_log.min()) / (s_log.max() - s_log.min() + 1e-6)
        specs.append(s_norm)
    
    return np.stack(specs).astype(np.float32)  # Returns (C, order_bins)

def label_from_filename(name):
    name = name.lower()
    if "normal" in name:
        return "Normal"
    if any(k in name for k in ["ball", "inner", "outer", "cage", "bearing"]):
        return "Bearing"
    # CWRU doesn't have pure imbalance/misalignment - map to closest
    if "28" in name or "32" in name:  # Common bearing fault codes
        return "Bearing"
    return "Normal"  # Conservative fallback

# --------------------------------------------------
# INFERENCE WITH CORRECTED SIGNAL HANDLING
# --------------------------------------------------
def process_mat_file(filepath, model, device, cfg, writer):
    print(f"\nProcessing {filepath.name}")

    mat = loadmat(filepath)
    sig_1ch = find_signal(mat)

    if sig_1ch is None or len(sig_1ch) < 100:
        print("  ❌ No valid signal found")
        return None

    # CRITICAL FIX: Ensure true 1D before stacking
    if sig_1ch.ndim != 1:
        sig_1ch = sig_1ch.flatten()
    
    # Replicate to 3 channels (model expects 3-channel input)
    sig = np.stack([sig_1ch, sig_1ch, sig_1ch], axis=1)  # (N, 3)

    # Resample to model's expected sampling rate
    gcd = np.gcd(int(cfg.original_fs), int(cfg.model_fs))
    sig = resample_poly(sig, cfg.model_fs // gcd, cfg.original_fs // gcd, axis=0)
    
    win_pts = int(cfg.window_sec * cfg.model_fs)
    if len(sig) < win_pts:
        # Pad short signals (matches training behavior)
        pad_len = win_pts - len(sig)
        sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')
    
    # Process non-overlapping windows
    preds, confs = [], []
    for start in range(0, len(sig) - win_pts + 1, win_pts):
        segment = sig[start:start + win_pts]
        segment = safe_zscore(segment)
        
        # FIXED: Now handles all signal shapes correctly
        spec = compute_order_spectrum(segment, cfg.model_fs, rpm=1797.0, cfg=cfg)
        x = torch.from_numpy(spec).unsqueeze(0).to(device)  # (1, C, order_bins)
        
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            preds.append(torch.argmax(probs, dim=1).item())
            confs.append(torch.max(probs).item())
    
    if not preds:
        return None
    
    # Majority vote across windows
    final_pred = max(set(preds), key=preds.count)
    final_conf = float(np.mean(confs))
    pred_label = cfg.classes[final_pred]
    gt = label_from_filename(filepath.name)
    
    writer.writerow([
        filepath.name,
        gt,
        pred_label,
        f"{final_conf:.4f}",
        "1797"
    ])
    
    print(f"  → GT: {gt:12s} | Pred: {pred_label:12s} | Conf: {final_conf:.2%}")
    return gt, pred_label

# --------------------------------------------------
# MAIN WITH SAFETY CHECKS
# --------------------------------------------------
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "cwru"
    MODEL_PATH = BASE_DIR / "src" / "models_sota_resnet" / "best_model_resnet.pth"
    OUTPUT_CSV = BASE_DIR / "cwru_results_resnet.csv"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"CWRU data directory not found at {DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with EXACT architecture used during training
    model = ResNet1D(input_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")

    files = sorted(DATA_DIR.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {DATA_DIR}")
    
    print(f"Found {len(files)} CWRU files to process")
    results = []

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "GT", "Pred", "Confidence", "RPM"])
        
        for file in tqdm(files, desc="Processing CWRU files"):
            try:
                out = process_mat_file(file, model, device, cfg, writer)
                if out:
                    results.append(out)
            except Exception as e:
                print(f"\n⚠️ Error processing {file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    if not results:
        print("\n❌ No valid results generated - check signal extraction")
        return
    
    # Generate report
    df = pd.DataFrame(results, columns=["GT", "Pred"])
    print("\n" + "="*60)
    print("CWRU CLASSIFICATION RESULTS (vs MAFAULDA-trained model)")
    print("="*60)
    print("\nConfusion Matrix (% of GT class):")
    cm = pd.crosstab(df.GT, df.Pred, normalize='index') * 100
    print(cm.round(1))
    
    print("\nClass Distribution:")
    print(df['GT'].value_counts().sort_index())
    
    print(f"\n✓ Results saved to: {OUTPUT_CSV.absolute()}")

if __name__ == "__main__":
    main()