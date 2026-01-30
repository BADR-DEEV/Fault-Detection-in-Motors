import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, resample_poly, hilbert
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------------
# 1. INDUSTRIAL CONFIGURATION (SKF/SIEMENS STANDARD ALIGNMENT)
# -----------------------------------------------------------------------------
class Config:
    def __init__(self, raw_dir):
        script_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_data_dir = script_dir / "src" / "4k_mafulda_structured"
        self.processed_dir = script_dir / "src" / "4k_mafulda_structured"
        self.output_dir = Path("models_industrial_v1")
        
        self.orig_fs = 50000
        # Increased to 6kHz to capture higher harmonics of bearing faults
        self.target_fs = 4000 

        
        # --- PHYSICS CONSTRAINTS (GENERALIZATION) ---
        # SKF Enveloping usually looks at 0-50 Orders. 
        # We use 32 to capture 3x-8x bearing funds + 3 harmonics.
        self.max_order = 32  
        # High resolution needed to distinguish 3.0x (Misalign) from 3.05x (BPFO)
        self.order_bins = 256   
        self.window_sec = 1.0  
        
        # Training
        self.batch_size = 64
        self.epochs = 40 # Increased slightly for convergence
        self.lr = 0.0003 # Lower LR for stability
        
        # OOD Detection Threshold (Entropy based)
        # If entropy > threshold, classify as Unknown
        self.ood_entropy_threshold = 0.9 
        
        # Augmentation
        self.noise_std = 0.1 

def setup_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train_log.txt", mode="w")]
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING UTILS (ROBUST ORDER ANALYSIS)
# -----------------------------------------------------------------------------
def resample_sig(x, orig_fs, target_fs):
    if int(orig_fs) == int(target_fs): return x
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(x, int(target_fs//gcd), int(orig_fs//gcd), axis=0).astype(np.float32)

def get_rpm(tach, fs):
    # MAFAULDA tachs can be noisy. Thresholding must be robust.
    tach = tach - np.mean(tach)
    # Simple Schmitt Trigger simulation
    high = tach > 2.0 
    pulses = np.where(np.diff(high.astype(int)) > 0)[0]
    
    if len(pulses) < 2: return 0.0
    avg_diff = np.mean(np.diff(pulses))
    if avg_diff == 0: return 0.0
    
    rpm = (fs / avg_diff) * 60.0
    return rpm

def compute_industrial_spectrum(sig, fs, rpm, cfg):
    """
    Computes a Log-Magnitude Order Spectrum.
    This creates invariance to RPM changes (x-axis) and Power levels (normalization).
    """
    # Fallback for constant speed if tach fails (Standard Industrial Assumption)
    if rpm <= 100: rpm = 1500.0 
    shaft_hz = rpm / 60.0
    
    # 1. Welch Spectrum (Hz domain)
    # nperseg is critical: determines Hz resolution.
    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 2048), axis=0)
    
    # 2. Convert Hz -> Order
    # Order = Frequency / Shaft_Freq
    orders = f / shaft_hz
    
    # 3. Interpolate to Standardized Grid (0 to max_order)
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    specs = []
    for ch in range(sig.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        
        # --- PHYSICS FILTERING ---
        # 1. Zero out DC (0th order)
        s[:3] = 0.0 
        
        # 2. Log Scaling (dB)
        # Crucial for distinguishing Bearings (low energy) from Unbalance (high energy)
        s_log = 10 * np.log10(s + 1e-12)
        
        # 3. Instance Normalization (Power Invariance)
        # Makes the shape matter, not the absolute amplitude
        s_norm = (s_log - np.median(s_log)) / (np.std(s_log) + 1e-6)
        
        # Clip outliers to prevent exploding gradients
        s_norm = np.clip(s_norm, -3.0, 3.0)
        
        specs.append(s_norm)
        
    return np.stack(specs).astype(np.float32)

# -----------------------------------------------------------------------------
# 3. DATA PROCESSING (MAFAULDA STRUCTURE)
# -----------------------------------------------------------------------------
def process_data_mafaulda(cfg, log):
    meta_path = cfg.processed_dir / "metadata.csv"
    if meta_path.exists():
        log.info(f"Found existing metadata.csv. Using cached structure.")
        return pd.read_csv(meta_path)

    log.info(f"Scanning MAFAULDA structure at {cfg.raw_data_dir}...")
    files = sorted(list(cfg.raw_data_dir.rglob("*.csv")))
    metadata_rows = []
    
    # MAFAULDA Specific Mapping
    # 0: Normal, 1: Imbalance, 2: Misalignment, 3: Bearing
    
    for i, p in enumerate(files):
        fname = str(p).lower().replace("\\", "/")
        path_parts = fname.split("/")
        
        label = -1
        label_name = "Unknown"
        
        # Heuristic Logic for MAFAULDA Folder Structure
        if "normal" in fname:
            label, label_name = 0, "Normal"
        elif "imbalance" in fname:
            label, label_name = 1, "Imbalance"
        elif "horizontal" in fname or "vertical" in fname:
            # Misalignment is usually labeled by alignment direction in MAFAULDA
            label, label_name = 2, "Misalignment"
        elif any(x in fname for x in ["ball", "outer", "inner", "cage", "bearing"]):
            label, label_name = 3, "Bearing"
            
        # Filter: Only keep Overhang or Underhang (Simulates 3-axis industrial accel)
        if label != -1:
            # We assume columns are Accel X, Y, Z (or similar)
            # MAFAULDA: Col 0=Tach, Col 1-3=Accel
            try:
                # Read only first few rows to check format
                df_test = pd.read_csv(p, nrows=5, header=None)
                if df_test.shape[1] < 4: continue # Skip if not Tach + 3 Accel

                # Valid file, process fully
                df = pd.read_csv(p, header=None)
                tach = df.iloc[:, 0].values
                # Take all 3 axes (Simulating Triaxial Sensor)
                sig = df.iloc[:, 1:4].values 
                
                rpm = get_rpm(tach, cfg.orig_fs)
                
                # Resample immediately to save space
                sig_res = resample_sig(sig, cfg.orig_fs, cfg.target_fs)
                
                rel_path = p.relative_to(cfg.raw_data_dir)
                save_path = cfg.processed_dir / rel_path.with_suffix('.npz')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                np.savez(save_path, sig=sig_res, label=label, rpm=rpm)
                metadata_rows.append({
                    "path": str(save_path), 
                    "original_path": str(p), 
                    "label": label, 
                    "label_name": label_name, 
                    "rpm": round(rpm, 1)
                })
            except Exception as e:
                log.warning(f"Error reading {p.name}: {e}")

        if i % 100 == 0: log.info(f"Scanned {i}/{len(files)} files...")

    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(meta_path, index=False)
    log.info(f"Processed {len(df_meta)} files.")
    return df_meta

# -----------------------------------------------------------------------------
# 4. DATASET & OOD HANDLING
# -----------------------------------------------------------------------------
class IndustrialDataset(Dataset):
    def __init__(self, metadata_df, cfg, augment=False):
        self.meta = metadata_df
        self.cfg = cfg
        self.augment = augment

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        try:
            with np.load(row['path']) as data:
                sig = data['sig']
                label = int(data['label'])
                rpm = float(data['rpm'])
        except:
            # Fallback for corrupted files
            return torch.zeros((3, self.cfg.order_bins)), torch.tensor(0)
            
        win_pts = int(self.cfg.window_sec * self.cfg.target_fs)
        
        # Random crop for data augmentation
        if len(sig) > win_pts:
            if self.augment:
                start = np.random.randint(0, len(sig) - win_pts)
            else:
                start = (len(sig)-win_pts)//2
            sig_crop = sig[start:start+win_pts]
        else:
            sig_crop = np.pad(sig, ((0, win_pts - len(sig)), (0,0)))

        # Add Sensor Noise (Augmentation)
        if self.augment:
            noise = np.random.normal(0, 0.05, size=sig_crop.shape)
            sig_crop = sig_crop + noise

        # Compute Invariant Spectrum
        x = compute_industrial_spectrum(sig_crop, self.cfg.target_fs, rpm, self.cfg)
        
        # Feature Masking (Augmentation for Robustness)
        if self.augment and np.random.rand() > 0.5:
            # Randomly zero out a band of orders (simulates sensor dropout)
            mask_start = np.random.randint(0, self.cfg.order_bins - 10)
            x[:, mask_start:mask_start+10] = 0.0

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

# -----------------------------------------------------------------------------
# 5. MODEL WITH OOD HEAD
# -----------------------------------------------------------------------------
class IndustrialCNN(nn.Module):
    def __init__(self, input_bins, num_classes=4):
        super().__init__()
        
        # Wide kernel first to capture broad order patterns
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=15, stride=2, padding=7), # /2
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), # /4
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # /8
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # /16
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling (Invariant to exact bin shift)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def predict_with_ood(self, x, entropy_thresh=0.8):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        
        # Calculate Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        
        preds = torch.argmax(probs, dim=1)
        
        # If entropy is high, the model is confused -> Unknown
        # We assign a special label -1 for Unknown
        is_ood = entropy > entropy_thresh
        final_preds = preds.clone()
        final_preds[is_ood] = -1 # -1 signifies OOD
        
        return final_preds, probs, entropy

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    
    cfg = Config(args.data_dir)
    log = setup_logger(cfg.output_dir)
    
    # 1. Process Data
    df_meta = process_data_mafaulda(cfg, log)
    
    # 2. Split (Grouped by file to prevent leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_meta, df_meta['label'], df_meta['original_path']))
    
    train_ds = IndustrialDataset(df_meta.iloc[train_idx], cfg, augment=True)
    val_ds = IndustrialDataset(df_meta.iloc[val_idx], cfg, augment=False)
    
    # 3. Class Weighting
    counts = df_meta.iloc[train_idx]['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1.0)
    sample_weights = [weights[y] for y in df_meta.iloc[train_idx]['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dl = DataLoader(train_ds, cfg.batch_size, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)
    
    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndustrialCNN(input_bins=cfg.order_bins).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4) # AdamW for better generalization
    crit = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing helps OOD detection later
    
    log.info("Starting Training (Industrial Standard)...")
    
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
        # Validation with OOD check
        model.eval()
        all_preds, all_trues, ood_counts = [], [], 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                
                # Use the OOD prediction logic
                preds, probs, ent = model.predict_with_ood(x, entropy_thresh=cfg.ood_entropy_threshold)
                
                # Count how many validation samples were rejected as OOD
                # (Ideally 0 for in-distribution validation data, but tracks confidence)
                ood_counts += (preds == -1).sum().item()
                
                # For accuracy calc, we only look at non-OOD or treat OOD as error
                # Here we take the raw argmax for accuracy to check basic performance
                raw_preds = probs.argmax(1)
                all_preds.extend(raw_preds.cpu().numpy())
                all_trues.extend(y.cpu().numpy())
        
        acc = accuracy_score(all_trues, all_preds)
        log.info(f"Epoch {ep+1} | Loss: {np.mean(losses):.4f} | Val Acc: {acc:.4f} | Low Conf Samples: {ood_counts}")
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), cfg.output_dir / "last_model.pth")

    # 5. Final Report
    log.info("\n" + classification_report(all_trues, all_preds, target_names=["Normal", "Imbalance", "Misalign", "Bearing"]))
    
    # Save Model
    torch.save(model.state_dict(), cfg.output_dir / "industrial_model_final.pth")
    log.info("Pipeline Complete.")