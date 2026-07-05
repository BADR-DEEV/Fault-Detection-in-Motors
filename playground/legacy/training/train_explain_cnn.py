import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, resample_poly
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------------
# 1. CONFIGURATION (PHYSICS FOCUSED)
# -----------------------------------------------------------------------------
class Config:
    def __init__(self, raw_dir):
        script_dir = Path(__file__).resolve().parent
        

        self.raw_data_dir = Path(raw_dir)
        self.processed_dir = Path("4k_mafulda_structured") 
        self.output_dir = Path("models_physics_aware_2")
        
        self.orig_fs = 50000
        self.target_fs = 4000
        
        # --- PHYSICS CONSTRAINTS ---
        # We limit the view to 5.0 Orders.
        # Imbalance is at 1.0, Misalignment at 1.0/2.0.
        # We cut off the high-frequency "rattle" that confused the model.
        self.max_order = 6  
        self.order_bins  = 128   
        self.window_sec = 1.0  
        
        # Training
        self.batch_size = 64
        self.epochs = 35
        self.lr = 0.0005        
        
        # High Noise to force model to ignore subtle artifacts
        self.noise_std = 0.15 

        

def setup_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train_log.txt", mode="w")]
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING UTILS
# -----------------------------------------------------------------------------
def resample_sig(x, orig_fs, target_fs):
    if int(orig_fs) == int(target_fs): return x
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(x, int(target_fs//gcd), int(orig_fs//gcd), axis=0).astype(np.float32)

def get_rpm(tach, fs):
    tach = tach - np.mean(tach)
    pulses = np.where(np.diff((tach > 0.5).astype(int)) > 0)[0]
    if len(pulses) < 2: return 0.0
    avg_diff = np.mean(np.diff(pulses))
    return 0.0 if avg_diff == 0 else (fs / avg_diff) * 60.0

def safe_zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def compute_order_spectrum(sig, fs, rpm, cfg):
    if rpm <= 100: rpm = 1500.0 
    shaft_hz = rpm / 60.0
    
    # Welch Spectrum
    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 1024), axis=0)
    
    # Hz -> Order
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    specs = []
    for ch in range(sig.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        
        # --- NEW: PHYSICS FIX ---
        # Kill DC offset and very low freq (below 0.1 order)
        # This removes the noise on the far left of your graph
        s[:int(cfg.order_bins * 0.05)] = 0.0 
        # ------------------------

        s_log = np.log(s + 1e-12)
        s_norm = (s_log - s_log.mean()) / (s_log.std() + 1e-6)
        specs.append(s_norm)
        
    return np.stack(specs).astype(np.float32)

# -----------------------------------------------------------------------------
# 3. DATA PROCESSING
# -----------------------------------------------------------------------------
def process_data_preserving_structure(cfg, log):
    meta_path = cfg.processed_dir / "metadata.csv"
    if meta_path.exists():
        log.info(f"Found existing metadata.csv. Using cached structure.")
        return pd.read_csv(meta_path)

    log.info(f"Processing data... Mirroring structure to {cfg.processed_dir}")
    files = sorted(list(cfg.raw_data_dir.rglob("*.csv")))
    metadata_rows = []
    
    for i, p in enumerate(files):
        fname = str(p).lower().replace("\\", "/")
        label = -1
        label_name = "Unknown"
        
        if "normal" in fname: label, label_name = 0, "Normal"
        elif "imbalance" in fname and "bearing" not in fname: label, label_name = 1, "Imbalance"
        elif "horizontal" in fname or "vertical" in fname: label, label_name = 2, "Misalignment"
        elif any(k in fname for k in ["overhang", "underhang", "bearing", "ball", "outer", "inner", "cage"]): label, label_name = 3, "Bearing"
            
        if label == -1: continue 

        try:
            df = pd.read_csv(p, header=None)
            tach = df.iloc[:, 0].values
            sig = df.iloc[:, 1:4].values
            rpm = get_rpm(tach, cfg.orig_fs)
            sig_4k = resample_sig(sig, cfg.orig_fs, cfg.target_fs)
            
            rel_path = p.relative_to(cfg.raw_data_dir)
            save_path = cfg.processed_dir / rel_path.with_suffix('.npz')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_path, sig=sig_4k, label=label, rpm=rpm)
            metadata_rows.append({"path": str(save_path), "original_path": str(p), "label": label, "label_name": label_name, "rpm": round(rpm, 1)})
            
            if i % 200 == 0: log.info(f"Scanned {i} files...")
        except Exception as e:
            log.warning(f"Error processing {p.name}: {e}")

    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(meta_path, index=False)
    return df_meta

# -----------------------------------------------------------------------------
# 4. DATASET
# -----------------------------------------------------------------------------
class StructuredDataset(Dataset):
    def __init__(self, metadata_df, cfg, augment=False):
        self.meta = metadata_df
        self.cfg = cfg
        self.augment = augment

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        with np.load(row['path']) as data:
            sig = data['sig']
            label = int(data['label'])
            rpm = float(data['rpm'])
            
        win_pts = int(self.cfg.window_sec * self.cfg.target_fs)
        if len(sig) > win_pts:
            start = np.random.randint(0, len(sig) - win_pts) if self.augment else (len(sig)-win_pts)//2
            sig_crop = sig[start:start+win_pts]
        else:
            sig_crop = np.pad(sig, ((0, win_pts - len(sig)), (0,0)))

        sig_crop = safe_zscore(sig_crop)
        x = compute_order_spectrum(sig_crop, self.cfg.target_fs, rpm, self.cfg)
        
        if self.augment:
            noise = np.random.normal(0, self.cfg.noise_std, size=x.shape)
            x = x + noise
            
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label, dtype=torch.long)

# -----------------------------------------------------------------------------
# 5. DYNAMIC MODEL ARCHITECTURE (Fixes the Error)
# -----------------------------------------------------------------------------
class ExplainableCNN(nn.Module):
    def __init__(self, input_bins, num_classes=4):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2) # Size / 2
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2) # Size / 4
        )
        
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2) # Size / 8
        
        # --- DYNAMIC CALCULATION ---
        # Calculate what the flat size will be after 3 pooling layers
        final_width = input_bins // 8 
        linear_input_size = 64 * final_width
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.gradients = None

    def activations_hook(self, grad): self.gradients = grad

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)
        if x.requires_grad: x.register_hook(self.activations_hook)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self): return self.gradients
    
    def get_activations(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------
def save_explanation(model, dataset, cfg, log):
    log.info("Generating Saliency Maps...")
    device = next(model.parameters()).device
    model.eval()
    classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
    found = set()
    indices = np.random.permutation(len(dataset))
    
    for idx in indices:
        if len(found) == 4: break
        x, y = dataset[idx]
        label = int(y)
        if label in found: continue
        
        x_in = x.unsqueeze(0).to(device).requires_grad_()
        out = model(x_in)
        pred = out.argmax(1).item()
        if pred != label: continue
        found.add(label)
        
        model.zero_grad()
        out[0, pred].backward()
        
        grads = model.get_activations_gradient()
        pooled_grads = torch.mean(grads, dim=[0, 2])
        acts = model.get_activations(x_in).detach()
        for i in range(64): acts[:, i, :] *= pooled_grads[i]
        
        heatmap = torch.mean(acts, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-9)
        heatmap = resample_poly(heatmap, cfg.order_bins, len(heatmap))
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        orders = np.linspace(0, cfg.max_order, cfg.order_bins)
        
        ax[0].plot(orders, x[1].numpy(), 'k', alpha=0.8)
        ax[0].set_title(f"Class: {classes[label]}")
        ax[0].set_xlim(0, cfg.max_order)
        
        ax[1].plot(orders, x[1].numpy(), 'k', alpha=0.3)
        ax[1].fill_between(orders, 0, heatmap * np.max(x[1].numpy()), color='r', alpha=0.6)
        ax[1].set_title("Grad-CAM Saliency (Physics Check)")
        ax[1].set_xlim(0, cfg.max_order)
        
        plt.tight_layout()
        plt.savefig(cfg.output_dir / f"explain_{classes[label]}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    
    cfg = Config(args.data_dir)
    log = setup_logger(cfg.output_dir)
    
    df_meta = process_data_preserving_structure(cfg, log)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_meta, df_meta['label'], df_meta['original_path']))
    train_ds = StructuredDataset(df_meta.iloc[train_idx], cfg, augment=True)
    val_ds = StructuredDataset(df_meta.iloc[val_idx], cfg, augment=False)
    
    counts = df_meta.iloc[train_idx]['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-6)
    sample_weights = [weights[y] for y in df_meta.iloc[train_idx]['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dl = DataLoader(train_ds, 64, sampler=sampler)
    val_dl = DataLoader(val_ds, 64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pass input_bins to model so it sizes correctly
    model = ExplainableCNN(input_bins=cfg.order_bins).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    
    log.info("Starting Training (Physics Aware)...")
    best_acc = 0.0
    
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        log.info(f"Epoch {ep+1} | Loss: {np.mean(losses):.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.output_dir / "best_model_maf_another.pth")

    log.info("\n" + classification_report(trues, preds, target_names=["Normal", "Imbalance", "Misalign", "Bearing"]))
    save_explanation(model, val_ds, cfg, log)
    log.info("Done.")