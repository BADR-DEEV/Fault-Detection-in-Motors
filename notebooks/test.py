import os
import sys
from pathlib import Path
import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, welch

# Adjust path to find your source modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import your specific model functions
# Ensure train_explain_cnn contains ExplainableCNN and compute_order_spectrum
from playground.legacy.training.train_explain_cnn import ExplainableCNN, compute_order_spectrum, safe_zscore

# --- 1. DEFINE LOCAL CONFIG FOR INFERENCE ---
# We define this here so we don't depend on the training Config class
class InferenceConfig:
    def __init__(self):
        self.max_order = 4.0      # MUST match training (Physics Aware)
        self.order_bins = 128     # MUST match training
        self.target_fs = 4000     # The model expects 4k
        self.window_sec = 1.0     # The model expects 1 sec windows

# --- 2. SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parents[2] 
# Update this path to exactly where your .mat file is
MAT_FILE_PATH = BASE_DIR / "data" / "paper_data" / "Faulty" / "F56.mat"
VARIABLE_NAME = "H"
ORIGINAL_FS = 1000  # The sampling rate of the MAT file
# ---------------------

def load_and_predict():
    # 0. Setup Config
    cfg = InferenceConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    # Note: Removed input_bins arg, as the class usually doesn't take it in __init__
    model = ExplainableCNN(input_bins = 128).to(device)
    
    # Update path to your best model
    model_path = "models_physics_aware_2/best_model_maf_another.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model from {model_path}")
        print(f"Error: {e}")
        return
        
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Load Data
    print(f"Loading {MAT_FILE_PATH}...")
    if not MAT_FILE_PATH.exists():
        print("File not found!")
        return

    try:
        mat = scipy.io.loadmat(MAT_FILE_PATH)
    except Exception as e:
        print(f"Could not read .mat file: {e}")
        return
    
    if VARIABLE_NAME not in mat:
        print(f"Error: Variable '{VARIABLE_NAME}' not found. Keys found: {mat.keys()}")
        return

    sig_raw = mat[VARIABLE_NAME]
    
    # Shape correction: Ensure (Time, 3)
    # If (3, Time), transpose
    if sig_raw.shape[0] == 3 and sig_raw.shape[1] > 3:
        sig_raw = sig_raw.T
    
    # If 1D, duplicate to 3 channels
    if sig_raw.ndim == 1:
        sig_raw = np.stack([sig_raw, sig_raw, sig_raw], axis=1)
        
    # If (Time, 1), duplicate
    if sig_raw.ndim == 2 and sig_raw.shape[1] == 1:
        sig_raw = np.concatenate([sig_raw, sig_raw, sig_raw], axis=1)

    print(f"Original Signal Shape: {sig_raw.shape} @ {ORIGINAL_FS}Hz")

    # 3. Resample to 4000 Hz
    # The model was trained on 4000Hz patterns. Even if data is 1000Hz,
    # we must upsample to match the input layer size.
    if int(ORIGINAL_FS) != int(cfg.target_fs):
        print(f"Resampling from {ORIGINAL_FS} to {cfg.target_fs}...")
        gcd = np.gcd(int(ORIGINAL_FS), int(cfg.target_fs))
        sig_4k = resample_poly(sig_raw, int(cfg.target_fs//gcd), int(ORIGINAL_FS//gcd), axis=0)
    else:
        sig_4k = sig_raw.astype(np.float32)

    # 4. RPM ESTIMATION
    # Look for peak freq between 10Hz (600RPM) and 65Hz (3900RPM)
    f, Pxx = welch(sig_4k[:, 0], fs=cfg.target_fs, nperseg=4096)
    valid_mask = (f > 10) & (f < 65)
    
    if valid_mask.sum() > 0:
        dominant_freq = f[valid_mask][np.argmax(Pxx[valid_mask])]
        rpm_est = dominant_freq * 60.0
    else:
        rpm_est = 1500.0 # Fallback
        
    print(f"Estimated RPM: {rpm_est:.1f}")

    # 5. Windowing (Center Crop)
    win_pts = int(cfg.window_sec * cfg.target_fs) # 4000 pts
    
    if len(sig_4k) < win_pts:
        # Pad if too short
        pad_amt = win_pts - len(sig_4k)
        sig_4k = np.pad(sig_4k, ((0, pad_amt), (0,0)), mode='edge')
        
    # Take center window
    start = (len(sig_4k) - win_pts) // 2
    sig_window = sig_4k[start:start+win_pts]
    
    # 6. Feature Extraction
    # Z-Score
    sig_window = safe_zscore(sig_window)
    
    # Order Spectrum
    # Note: passing our local 'cfg' which has max_order and order_bins
    x = compute_order_spectrum(sig_window, cfg.target_fs, rpm_est, cfg)
    
    # To Tensor
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

    # 7. Prediction
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
    
    classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
    
    print("\n" + "="*30)
    print(f"PREDICTION: {classes[pred_idx].upper()}")
    print(f"Confidence: {probs[0][pred_idx]*100:.1f}%")
    print("="*30)
    print("Detailed Probabilities:")
    for i, c in enumerate(classes):
        bar = "|" * int(probs[0][i] * 20)
        print(f"{c:<15}: {probs[0][i]*100:5.1f}%  {bar}")

if __name__ == "__main__":
    load_and_predict()