import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly, welch
from tqdm import tqdm
import sys

# --- CONFIG ---
class TransferConfig:
    def __init__(self, data_dir, model_path):
        self.raw_data_dir = Path(data_dir)
        self.output_dir = Path("models_transfer_learning")
        self.base_model_path = Path(model_path)
        
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir = self.output_dir / "cache_npz"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.dataset_fs = 4096
        self.model_fs = 4000
        self.max_order = 6.0
        self.order_bins = 128
        self.window_sec = 1.0
        
        # Transfer Learning Params
        self.samples_per_file = 300  # Take 300 random windows per CSV file
        self.batch_size = 32
        self.epochs = 10             # Quick fine-tuning
        self.lr = 0.0001             # Low learning rate to not break weights

# --- MODEL (Same as before) ---
class ExplainableCNN(nn.Module):
    def __init__(self, input_bins=128, num_classes=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        
        final_width = input_bins // 8 
        linear_input_size = 64 * final_width
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.classifier(x)
        return x

# --- HELPER FUNCTIONS ---
def safe_zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def compute_order_spectrum(sig, fs, rpm, cfg):
    if rpm < 10: rpm = 1500.0 
    shaft_hz = rpm / 60.0
    nperseg = min(len(sig), 1024)
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, axis=0)
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    specs = []
    for ch in range(sig.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        s[:int(cfg.order_bins * 0.05)] = 0.0 
        s_log = np.log(s + 1e-12)
        s_norm = (s_log - s_log.mean()) / (s_log.std() + 1e-6)
        specs.append(s_norm)
    return np.stack(specs).astype(np.float32)

# --- 1. PREPARE DATASET FROM CSVs ---
def prepare_transfer_data(cfg):
    print("--- Step 1: Extracting Transfer Data from CSVs ---")
    files = sorted(list(cfg.raw_data_dir.glob("*.csv")))
    meta_rows = []

    for f in files:
        # Label Logic: 0D/E -> 0 (Normal), 1D-4E -> 1 (Imbalance)
        if f.name.startswith("0"): label = 0
        else: label = 1
        
        # Skip Evaluation files for Training? No, let's use 'D' for Train, 'E' for Val
        is_val = "E.csv" in f.name
        split = "val" if is_val else "train"
        
        print(f"Processing {f.name} -> Label: {label} ({split})")
        
        # Read file
        df = pd.read_csv(f)
        
        # Filter valid RPM
        df = df[(df.iloc[:, 1] > 0) & (df.iloc[:, 1] < 10000)]
        
        data_len = len(df)
        win_len = int(cfg.dataset_fs * cfg.window_sec)
        
        # Randomly sample windows
        num_samples = cfg.samples_per_file
        starts = np.random.randint(0, data_len - win_len, size=num_samples)
        
        for i, start in enumerate(starts):
            chunk = df.iloc[start : start + win_len]
            
            rpm_vals = chunk.iloc[:, 1].values
            avg_rpm = np.mean(rpm_vals)
            
            vib_raw = chunk.iloc[:, 2:5].values
            
            # Resample
            if cfg.dataset_fs != cfg.model_fs:
                gcd = np.gcd(cfg.dataset_fs, cfg.model_fs)
                vib_4k = resample_poly(vib_raw, cfg.model_fs//gcd, cfg.dataset_fs//gcd, axis=0)
            else:
                vib_4k = vib_raw
            
            # Preprocess
            vib_4k = safe_zscore(vib_4k)
            x_map = compute_order_spectrum(vib_4k, cfg.model_fs, avg_rpm, cfg)
            
            # Save small file
            save_name = f"{f.stem}_{i}.npz"
            save_path = cfg.cache_dir / save_name
            np.savez(save_path, x=x_map, y=label)
            
            meta_rows.append({"path": str(save_path), "label": label, "split": split})

    return pd.DataFrame(meta_rows)

# --- DATASET CLASS ---
class TransferDataset(Dataset):
    def __init__(self, meta_df):
        self.meta = meta_df
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        data = np.load(row['path'])
        return torch.from_numpy(data['x']), torch.tensor(data['y'], dtype=torch.long)

# --- 2. FINE TUNING LOOP ---
def run_transfer_learning():
    # SETUP PATHS
    # Assume script is in src/training, go up to root
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT / "data/TFDT_Data"
    BASE_MODEL = ROOT / "src/models_physics_aware_2/best_model_maf_another.pth"
    
    cfg = TransferConfig(DATA_DIR, BASE_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    if not any(cfg.cache_dir.iterdir()):
        df_meta = prepare_transfer_data(cfg)
    else:
        print("Using cached extracted data...")
        # Reconstruct dataframe from files
        files = list(cfg.cache_dir.glob("*.npz"))
        rows = []
        for p in files:
            # Filename format: 0D_123.npz -> 0 is label 0. 1D_... -> 1 is label 1
            lbl = 0 if p.name.startswith("0") else 1
            split = "val" if "E_" in p.name else "train"
            rows.append({"path": str(p), "label": lbl, "split": split})
        df_meta = pd.DataFrame(rows)

    train_df = df_meta[df_meta['split'] == 'train']
    val_df = df_meta[df_meta['split'] == 'val']
    
    print(f"Training Samples: {len(train_df)} | Validation Samples: {len(val_df)}")
    
    # Sampler for Imbalance (We have fewer Normal files than Faulty files)
    counts = train_df['label'].value_counts().sort_index().values
    weights = 1.0 / counts
    sample_weights = [weights[y] for y in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dl = DataLoader(TransferDataset(train_df), cfg.batch_size, sampler=sampler)
    val_dl = DataLoader(TransferDataset(val_df), cfg.batch_size, shuffle=False)
    
    # 2. Load Model
    print("--- Step 2: Loading & Freezing Model ---")
    model = ExplainableCNN(input_bins=cfg.order_bins).to(device)
    model.load_state_dict(torch.load(cfg.base_model_path, map_location=device))
    
    # --- FREEZE LAYERS ---
    # We freeze block1 and block2 (Low level features)
    for param in model.block1.parameters(): param.requires_grad = False
    for param in model.block2.parameters(): param.requires_grad = False
    
    # We let conv3 and classifier adapt
    print("Frozen Block1 & Block2. Fine-tuning Conv3 & Classifier.")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Train
    print("--- Step 3: Training ---")
    best_acc = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y) # Model has 4 outputs, y has values 0 or 1. This works fine.
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                # Map outputs: 0->0, 1,2,3 -> 1 (Imbalance)
                # But for now, let's just check raw accuracy on classes 0 and 1
                preds = out.argmax(dim=1)
                
                # Logic: If GT is 0, Pred should be 0. If GT is 1, Pred should be 1, 2, or 3?
                # Actually, simply checking equality is hard because model might predict 'Misalign' (2) for 'Imbalance' (1).
                # Simplified Check: Normal vs Faulty
                is_normal_pred = (preds == 0)
                is_normal_true = (y == 0)
                correct += (is_normal_pred == is_normal_true).sum().item()
                total += y.size(0)
                
        acc = correct / total
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {np.mean(losses):.4f} | Val Accuracy (Normal/Faulty): {acc*100:.2f}%")
        
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.output_dir / "best_model_transfer.pth")
            
    print(f"Done. New model saved to {cfg.output_dir / 'best_model_transfer.pth'}")
    print("Use THIS model in your inference script now!")

if __name__ == "__main__":
    run_transfer_learning()