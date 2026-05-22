import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, butter, filtfilt, resample_poly, hilbert
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------
# 1. Configuration
# -----------------------
class Config:
    def __init__(self, mafaulda_dir, etfa_dir, output_dir):
        self.mafaulda_dir = Path(mafaulda_dir) if mafaulda_dir else None
        self.etfa_dir = Path(etfa_dir) if etfa_dir else None
        self.output_dir = Path(output_dir)
        
        # Sensor Settings (ADXL357z Simulation)
        self.target_fs = 4000     # Resample everything to 4kHz
        self.noise_level = 0.05   # Add 5% random noise to simulate MEMS floor
        
        # Signal Processing
        self.low_hz = 2.0
        self.high_hz = 1900.0
        self.window_seconds = 1.0
        self.hop_seconds = 0.5
        
        # Order Analysis
        self.max_order = 10.0
        self.order_bins = 128
        
        # Training
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 40
        
        # Labels: 0=Normal, 1=Imbalance, 2=Misalignment, 3=Bearing
        self.class_names = ["Normal", "Imbalance", "Misalignment", "Bearing"]

# -----------------------
# 2. Signal Processing Utils
# -----------------------
def bandpass(x, fs, low, high):
    nyq = fs / 2.0
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x, axis=0)

def safe_zscore(x):
    # Critical for Domain Generalization:
    # Removes absolute amplitude differences between rigs.
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def resample_signal(sig, orig_fs, target_fs):
    if len(sig) == 0: return sig
    if int(orig_fs) == int(target_fs): return sig
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(sig, int(target_fs//gcd), int(orig_fs//gcd), axis=0).astype(np.float32)

def compute_features(sig, fs, rpm, cfg):
    """Generates Order Map + Scalar Stats."""
    # 1. Scalar Features (RMS, Crest Factor)
    feats = []
    for ch in range(sig.shape[1]):
        x = sig[:, ch]
        rms = np.sqrt(np.mean(x**2))
        peak = np.max(np.abs(x))
        feats.extend([np.log(rms + 1e-9), np.log(peak/(rms+1e-9) + 1e-9)])
    
    # 2. Order Map
    shaft_hz = max(rpm / 60.0, 0.1)
    order_axis = np.linspace(0, cfg.max_order, cfg.order_bins)
    maps = []
    
    for ch in range(sig.shape[1]):
        f, Pxx = welch(sig[:, ch], fs=fs, nperseg=min(1024, len(sig)))
        orders = f / shaft_hz
        # Interpolate spectrum to fixed order bins
        spec = np.interp(order_axis, orders, Pxx, left=0, right=0)
        # Log scale and normalize
        spec = np.log(spec + 1e-12)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        maps.append(spec.astype(np.float32))
        
    return np.stack(maps), np.array(feats, dtype=np.float32)

# -----------------------
# 3. Data Parsers
# -----------------------
def load_mafaulda(cfg):
    """
    Loads MAFAULDA, merging Horizontal/Vertical Misalignment.
    Returns: X (Data), y (Labels), F (Feats)
    """
    print("Loading MAFAULDA (Teacher Dataset)...")
    samples_x, samples_f, samples_y = [], [], []
    
    # Mapping Logic
    # We consolidate specific faults into 4 broad classes
    label_map = {
        "normal": 0,
        "imbalance": 1,
        "horizontal": 2, "vertical": 2, # Merged Misalignment
        "underhang": 3, "overhang": 3   # Merged Bearing (Outer/Inner/Ball/Cage)
    }
    
    files = list(cfg.mafaulda_dir.rglob("*.csv"))
    win_pts = int(cfg.window_seconds * cfg.target_fs)
    hop_pts = int(cfg.hop_seconds * cfg.target_fs)
    
    for p in files:
        # 1. Determine Label
        fname = str(p).lower()
        lbl = -1
        if "normal" in fname: lbl = label_map["normal"]
        elif "imbalance" in fname: lbl = label_map["imbalance"]
        elif "horizontal" in fname: lbl = label_map["horizontal"]
        elif "vertical" in fname: lbl = label_map["vertical"]
        elif "bearing" in fname: lbl = label_map["underhang"] # Catch all bearings
        
        if lbl == -1: continue

        # 2. Load & Process
        try:
            df = pd.read_csv(p, header=None)
            # MAFAULDA: Col 0=Tach, 1-3=Underhang Accel
            raw_acc = df.iloc[:, 1:4].values 
            
            # Get RPM (Simple Tach Estimator)
            tach = df.iloc[:, 0].values
            # Thresholding tach to count pulses
            tach_bin = (tach > 2.0).astype(int) 
            diff = np.diff(tach_bin)
            pulses = np.where(diff > 0)[0]
            if len(pulses) > 5:
                avg_diff = np.mean(np.diff(pulses))
                rpm = (50000.0 / avg_diff) * 60.0
            else:
                rpm = 1500.0 # Fallback default
            
            # Resample 50k -> 4k
            acc = resample_signal(raw_acc, 50000, cfg.target_fs)
            
            # Slice into windows
            for start in range(0, len(acc) - win_pts, hop_pts):
                w = acc[start:start+win_pts]
                w = bandpass(w, cfg.target_fs, cfg.low_hz, cfg.high_hz)
                w = safe_zscore(w) # Normalize amplitude
                
                omap, feats = compute_features(w, cfg.target_fs, rpm, cfg)
                samples_x.append(omap)
                samples_f.append(feats)
                samples_y.append(lbl)
                
        except Exception as e:
            print(f"Error parsing {p.name}: {e}")
            continue

    return np.stack(samples_x), np.stack(samples_f), np.array(samples_y)

def load_etfa(cfg):
    """
    Loads ETFA (External Validation).
    Only contains Normal (0) and Imbalance (1).
    """
    print("Loading ETFA (External Test)...")
    samples_x, samples_f, samples_y = [], [], []
    
    files = list(cfg.etfa_dir.rglob("*.csv"))
    win_pts = int(cfg.window_seconds * cfg.target_fs)
    
    for p in files:
        fname = p.name
        if fname.startswith("0"): lbl = 0 # Normal
        elif fname[0] in ['1','2','3','4']: lbl = 1 # Imbalance
        else: continue
            
        try:
            df = pd.read_csv(p)
            # ETFA: Cols 2,3,4 are vibration
            acc = df.iloc[:, 2:5].values
            rpm = df.iloc[:, 1].mean()
            
            # Resample 4096 -> 4000
            acc = resample_signal(acc, 4096, cfg.target_fs)
            
            for start in range(0, len(acc) - win_pts, win_pts): # No overlap for test
                w = acc[start:start+win_pts]
                w = bandpass(w, cfg.target_fs, cfg.low_hz, cfg.high_hz)
                w = safe_zscore(w)
                
                omap, feats = compute_features(w, cfg.target_fs, rpm, cfg)
                samples_x.append(omap)
                samples_f.append(feats)
                samples_y.append(lbl)
        except:
            continue
            
    return np.stack(samples_x), np.stack(samples_f), np.array(samples_y)

# -----------------------
# 4. Dataset with MEMS Simulation
# -----------------------
class VibrationDataset(Dataset):
    def __init__(self, X, F, y, augment=False, noise_level=0.0):
        self.X = torch.from_numpy(X)
        self.F = torch.from_numpy(F)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        f = self.F[i]
        
        # MEMS Simulation: Inject noise during TRAINING only
        if self.augment:
            # Noise added to the spectral map directly (simplified)
            # or ideally to the time signal, but here we add to features
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
            f_noise = torch.randn_like(f) * (self.noise_level * 0.1)
            f = f + f_noise

        return x, f, self.y[i]

# -----------------------
# 5. Model
# -----------------------
class HybridCNN(nn.Module):
    def __init__(self, in_ch, n_feats, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(64 + n_feats, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, f):
        z = self.conv(x).flatten(1)
        combined = torch.cat([z, f], dim=1)
        return self.head(combined)

# -----------------------
# 6. Main Execution
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mafaulda", type=str, required=True, help="Path to MAFAULDA")
    parser.add_argument("--etfa", type=str, help="Path to ETFA (optional)")
    parser.add_argument("--out", type=str, default="./models_v3")
    args = parser.parse_args()
    
    cfg = Config(args.mafaulda, args.etfa, args.out)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    X_mafa, F_mafa, y_mafa = load_mafaulda(cfg)
    print(f"MAFAULDA loaded: {len(y_mafa)} samples")
    
    # Normalize Scalar Features
    f_mean, f_std = F_mafa.mean(0), F_mafa.std(0) + 1e-6
    F_mafa = (F_mafa - f_mean) / f_std
    
    # 2. Split MAFAULDA (Train / Validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(sss.split(X_mafa, y_mafa))
    
    train_ds = VibrationDataset(X_mafa[tr_idx], F_mafa[tr_idx], y_mafa[tr_idx], augment=True, noise_level=cfg.noise_level)
    val_ds = VibrationDataset(X_mafa[val_idx], F_mafa[val_idx], y_mafa[val_idx], augment=False)
    
    # Balance Classes
    counts = np.bincount(y_mafa[tr_idx])
    weights = 1. / (counts + 1e-6)
    samp_weights = weights[y_mafa[tr_idx]]
    sampler = WeightedRandomSampler(samp_weights, len(samp_weights))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 3. Train Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNN(X_mafa.shape[1], F_mafa.shape[1], 4).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    
    print("\n--- Starting Training (Teacher: MAFAULDA) ---")
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for bx, bf, by in train_loader:
            bx, bf, by = bx.to(device), bf.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx, bf), by)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
        # Validation on MAFAULDA
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, bf, by in val_loader:
                bx, bf, by = bx.to(device), bf.to(device), by.to(device)
                preds.extend(model(bx, bf).argmax(1).cpu().numpy())
                trues.extend(by.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        print(f"Epoch {ep+1} | Loss: {np.mean(losses):.3f} | Mafa-Val Acc: {acc:.3f}")

    # 4. EXTERNAL TEST (ETFA)
    if cfg.etfa_dir:
        print("\n--- Running External Reality Check (Student: ETFA) ---")
        X_etfa, F_etfa, y_etfa = load_etfa(cfg)
        
        # Must apply same normalization as training
        F_etfa = (F_etfa - f_mean) / f_std
        
        etfa_ds = VibrationDataset(X_etfa, F_etfa, y_etfa, augment=False)
        etfa_loader = DataLoader(etfa_ds, batch_size=cfg.batch_size)
        
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, bf, by in etfa_loader:
                bx, bf, by = bx.to(device), bf.to(device), by.to(device)
                preds.extend(model(bx, bf).argmax(1).cpu().numpy())
                trues.extend(by.cpu().numpy())
                
        preds = np.array(preds)
        trues = np.array(trues)
        
        # ANALYSIS
        # ETFA only has Normal (0) and Imbalance (1).
        # If model predicts 2 (Misalign) or 3 (Bearing), it's a "Hallucination".
        
        valid_indices = np.isin(preds, [0, 1])
        hallucinations = ~valid_indices
        
        print(f"Total ETFA Samples: {len(trues)}")
        print(f"Hallucinations (Model saw Bearing/Misalignment where none exists): {hallucinations.sum()} samples")
        
        # Accuracy only on valid predictions (Normal vs Imbalance)
        # Or raw accuracy (considering hallucinations as errors)
        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
        print(f"Classes: {cfg.class_names}")
        print(confusion_matrix(trues, preds, labels=[0,1,2,3]))
        
        print("\nClassification Report:")
        print(classification_report(trues, preds, labels=[0,1], target_names=["Normal", "Imbalance"]))
        
    # Save Model
    torch.save(model.state_dict(), cfg.output_dir / "final_model.pth")
    print("Done.")