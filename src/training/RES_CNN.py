import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import welch, resample_poly, correlate
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    def __init__(self, raw_dir):
        script_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_data_dir = script_dir / "data" / "raw_mafulda"
        self.processed_dir = script_dir / "data" / "4k_mafulda_structured"
        self.output_dir = Path("models_sota_resnet")
        print(f"Base Dir: {self.raw_data_dir}")
        print(f"Processed Dir: {self.processed_dir}")

        self.orig_fs = 50000
        self.target_fs = 4000  # Downsampling is fine for mechanical faults (<2kHz)
        
        # --- PHYSICS CONSTRAINTS ---
        # We increase resolution to capture distinct harmonics
        self.max_order = 10.0   # Look up to 10X RPM
        self.order_bins = 512   # Higher resolution (512 points for 10 orders)
        self.window_sec = 1.0  
        
        # Training
        self.batch_size = 32
        self.epochs = 40
        self.lr = 0.001        
        
        # Augmentation
        self.noise_std = 0.05 
        self.amp_jitter = 0.1

def setup_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train_log.txt", mode="w")]
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING (Physics & Features)
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

def compute_order_spectrum(sig, fs, rpm, cfg):
    if rpm <= 100:
        rpm = 1500.0

    shaft_hz = rpm / 60.0

    f, Pxx = welch(
        sig,
        fs=fs,
        nperseg=min(len(sig), 1024),
        axis=0,
        return_onesided=True  # 🔑 force training behavior
    )

    # ---- SHAPE GUARD (NO MATH CHANGE) ----
    if Pxx.ndim == 1:
        Pxx = Pxx[:, None]

    if Pxx.shape[0] != len(f):
        Pxx = Pxx[:len(f), :]
    # -------------------------------------

    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)

    specs = []
    for ch in range(Pxx.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)

        s_log = np.log(s + 1e-12)
        s_norm = (s_log - s_log.min()) / (s_log.max() - s_log.min() + 1e-6)

        specs.append(s_norm)

    return np.stack(specs).astype(np.float32)

# -----------------------------------------------------------------------------
# 3. DATASET
# -----------------------------------------------------------------------------
def process_data(cfg, log):
    meta_path = cfg.processed_dir / "metadata.csv"
    if meta_path.exists():
        log.info("Found metadata.csv. Loading...")
        return pd.read_csv(meta_path)

    log.info("Processing Raw Data...")
    files = sorted(list(cfg.raw_data_dir.rglob("*.csv")))
    metadata_rows = []
    
    for i, p in enumerate(files):
        fname = str(p).lower().replace("\\", "/")
        label = -1
        label_name = "Unknown"
        
        # Logic specific to MAFAULDA folder structure
        if "normal" in fname: label, label_name = 0, "Normal"
        elif "imbalance" in fname: label, label_name = 1, "Imbalance"
        elif "horizontal" in fname or "vertical" in fname: label, label_name = 2, "Misalignment"
        elif any(k in fname for k in ["overhang", "underhang", "ball", "outer", "inner", "cage"]): label, label_name = 3, "Bearing"
            
        if label == -1: continue 

        try:
            df = pd.read_csv(p, header=None)
            tach = df.iloc[:, 0].values
            sig = df.iloc[:, 1:4].values # Accel X, Y, Z
            rpm = get_rpm(tach, cfg.orig_fs)
            sig_4k = resample_sig(sig, cfg.orig_fs, cfg.target_fs)
            
            rel_path = p.relative_to(cfg.raw_data_dir)
            save_path = cfg.processed_dir / rel_path.with_suffix('.npz')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_path, sig=sig_4k, label=label, rpm=rpm)
            metadata_rows.append({"path": str(save_path), "original_path": str(p), "label": label, "label_name": label_name, "rpm": rpm})
        except Exception as e:
            log.warning(f"Error: {e}")

    df = pd.DataFrame(metadata_rows)
    df.to_csv(meta_path, index=False)
    return df

class VibrationDataset(Dataset):
    def __init__(self, metadata_df, cfg, augment=False):
        self.meta = metadata_df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        with np.load(row['path']) as data:
            sig = data['sig']        # (N, C)
            label = int(data['label'])
            rpm = float(data['rpm'])

        win_pts = int(self.cfg.window_sec * self.cfg.target_fs)

        # Random / Center crop
        if len(sig) > win_pts:
            start = np.random.randint(0, len(sig) - win_pts) if self.augment else (len(sig) - win_pts) // 2
            sig = sig[start:start + win_pts]
        else:
            sig = np.pad(sig, ((0, win_pts - len(sig)), (0, 0)))

        sig = safe_zscore(sig)

        # Order spectrum
        x = compute_order_spectrum(sig, self.cfg.target_fs, rpm, self.cfg)

        # Physics-aware augmentation
        if self.augment:
            x += np.random.normal(0, self.cfg.noise_std, x.shape)
            x *= np.random.uniform(
                1.0 - self.cfg.amp_jitter,
                1.0 + self.cfg.amp_jitter
            )

        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label)
# -----------------------------------------------------------------------------
# 4. SOTA MODEL: 1D ResNet (Better than plain CNN)
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, input_channels=3, num_classes=4):
        super().__init__()
        # Initial Feature Extraction
        self.in_planes = 32
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        
        # Classification Head (GAP + Dense)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------------
# 5. VISUALIZATION & METRICS
# -----------------------------------------------------------------------------
def plot_roc_curves(model, val_loader, num_classes, output_dir, device):
    model.eval()
    y_test = []
    y_score = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = torch.softmax(model(x), dim=1)
            y_test.extend(y.numpy())
            y_score.extend(outputs.cpu().numpy())
    
    y_test = np.array(y_test)
    y_score = np.array(y_score)
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    names = ["Normal", "Imbalance", "Misalignment", "Bearing"]
    plt.figure(figsize=(8, 6))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Analysis')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "ROC_Curves.png")
    plt.close()

def analyze_harmonics(dataset, cfg, output_dir):
    """
    Plots the average Order Spectrum for each class to verify Physics.
    This shows if the model is actually seeing 1X, 2X, 3X peaks.
    """
    classes = {0: "Normal", 1: "Imbalance", 2: "Misalignment", 3: "Bearing"}
    
    # Collect data by class
    class_data = {k: [] for k in classes}
    indices = np.random.choice(len(dataset), size=min(len(dataset), 500), replace=False)
    
    for i in indices:
        x, y = dataset[i]
        class_data[int(y)].append(x.numpy())
        
    orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    for cls_idx, data_list in class_data.items():
        if not data_list: continue
        # Average over samples, look at Channel 1 (usually Horizontal/Vertical)
        avg_spec = np.mean(np.array(data_list), axis=0)[1] 
        
        ax = axes[cls_idx]
        ax.plot(orders, avg_spec, color='blue')
        ax.set_title(f"Class: {classes[cls_idx]} (Average Spectrum)")
        ax.set_ylabel("Norm. Amplitude")
        
        # Draw Physics lines
        for harmonic in [1.0, 2.0, 3.0, 4.0]:
            ax.axvline(x=harmonic, color='r', linestyle='--', alpha=0.5)
            if cls_idx == 0: ax.text(harmonic, np.max(avg_spec)*0.8, f"{int(harmonic)}X", color='r')

    plt.xlabel("Order (X RPM)")
    plt.tight_layout()
    plt.savefig(output_dir / "Harmonic_Physics_Check.png")
    plt.close()

def plot_autocorr_demo(dataset, output_dir):
    """
    Demonstrates the Autocorrelation feature mentioned in Paper 1.
    """
    x, y = dataset[0] # Take one sample
    sig = x[1].numpy() # Take one channel
    
    # Compute Autocorr (inverse FFT of Power Spectrum)
    # Since x is already spectrum (magnitude), we treat it as Pxx
    # But Paper 1 does Autocorr on Time Domain. 
    # Here we just show the visual difference.
    
    acorr = np.correlate(sig, sig, mode='full')
    acorr = acorr[len(acorr)//2:]
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sig)
    plt.title("Order Spectrum (Frequency Domain)")
    
    plt.subplot(1, 2, 2)
    plt.plot(acorr)
    plt.title("Autocorrelation (Periodicity)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "Feature_Analysis_Autocorr.png")
    plt.close()

# -----------------------------------------------------------------------------
# 6. TRAINING MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    
    cfg = Config(args.data_dir)
    log = setup_logger(cfg.output_dir)
    
    # 1. Data Prep
    df_meta = process_data(cfg, log)
    
    # 2. Split (GroupShuffleSplit is CRITICAL to prevent data leakage from same file)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_meta, df_meta['label'], df_meta['original_path']))
    
    train_ds = VibrationDataset(df_meta.iloc[train_idx], cfg, augment=True)
    val_ds = VibrationDataset(df_meta.iloc[val_idx], cfg, augment=False)
    
    # Weighted Sampler for Class Imbalance
    counts = df_meta.iloc[train_idx]['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-6)
    sample_weights = [weights[y] for y in df_meta.iloc[train_idx]['label']]
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, cfg.batch_size, shuffle=False)
    
    # 3. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(input_channels=3, num_classes=4).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4) # AdamW helps generalization
    
    # Label Smoothing solves the "High Confidence Wrong Prediction" issue
    train_labels = df_meta.iloc[train_idx]['label'].values
    class_counts = np.bincount(train_labels, minlength=4)

    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    ) 
    
    # 4. Training Loop
    log.info("Starting Training (ResNet-1D)...")
    best_acc = 0.0
    
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        log.info(f"Epoch {ep+1}/{cfg.epochs} | Loss: {np.mean(losses):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.output_dir / "best_model_resnet.pth")

    # 5. Final Evaluation & Reports
    log.info("Generating Final Reports...")
    model.load_state_dict(torch.load(cfg.output_dir / "best_model_resnet.pth"))
    
    print("\n" + classification_report(trues, preds, target_names=["Normal", "Imbalance", "Misalign", "Bearing"]))
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Normal", "Imbalance", "Misalign", "Bearing"], yticklabels=["Normal", "Imbalance", "Misalign", "Bearing"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(cfg.output_dir / "confusion_matrix.png")
    # Generate Physics & Analysis Plots
    analyze_harmonics(val_ds, cfg, cfg.output_dir)
    plot_roc_curves(model, val_dl, 4, cfg.output_dir, device)
    plot_autocorr_demo(val_ds, cfg.output_dir)
    
    log.info(f"Done. Check {cfg.output_dir} for ROC curves and Harmonic plots.")