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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------------------

def extract_condition_from_path(p):
    """
    Extract condition from processed MAFAULDA path.
    Example:
    data/.../imbalance/10g/file.npz -> 10g
    """
    p = str(p).replace("\\", "/")
    parts = p.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid path for condition extraction: {p}")
    return parts[-2]


class Config:
    def __init__(self, _):
        script_dir = Path(__file__).resolve().parent.parent
        self.processed_dir = script_dir / "data" / "4k_mafulda_structured"
        self.output_dir = Path("models_industrial_v1")

        self.orig_fs = 50000
        self.target_fs = 4000

        self.max_order = 32
        self.order_bins = 256
        self.window_sec = 1.0

        self.batch_size = 64
        self.epochs = 120
        self.lr = 3e-3

        self.ood_entropy_threshold = 0.9
        self.noise_std = 0.05


def setup_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train_log.txt", mode="w"),
        ],
    )
    return logging.getLogger("train")


# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING
# -----------------------------------------------------------------------------

def resample_sig(x, orig_fs, target_fs):
    if int(orig_fs) == int(target_fs):
        return x
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(
        x, int(target_fs // gcd), int(orig_fs // gcd), axis=0
    ).astype(np.float32)


def compute_industrial_spectrum(sig, fs, rpm, cfg):
    if rpm <= 100:
        rpm = 1500.0

    shaft_hz = rpm / 60.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 2048), axis=0)

    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)

    specs = []
    for ch in range(sig.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        s[:3] = 0.0
        s = 10 * np.log10(s + 1e-12)
        s = (s - np.median(s)) / (np.std(s) + 1e-6)
        s = np.clip(s, -3.0, 3.0)
        specs.append(s)

    return np.stack(specs).astype(np.float32)


# -----------------------------------------------------------------------------
# 3. METADATA LOADER (NO REPROCESSING)
# -----------------------------------------------------------------------------

def load_metadata(cfg, log):
    meta_path = cfg.processed_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.csv not found")

    df = pd.read_csv(meta_path)

    def resolve_path(p):
        p = Path(p)

        # Case 1: already absolute → leave it
        if p.is_absolute():
            return p

        # Case 2: path already includes processed_dir name
        if p.parts[0] == cfg.processed_dir.name:
            return (cfg.processed_dir.parent / p).resolve()

        # Case 3: truly relative
        return (cfg.processed_dir / p).resolve()

    df["path"] = df["path"].apply(resolve_path)

    if "condition" not in df.columns:
        log.warning("`condition` column missing - deriving from path")
        df["condition"] = df["path"].astype(str).apply(extract_condition_from_path)

    log.info(f"Loaded metadata: {df.shape}")
    log.info(f"Labels:\n{df['label'].value_counts()}")
    log.info(f"Conditions: {df['condition'].nunique()}")

    return df

# -----------------------------------------------------------------------------
# 4. DATASET
# -----------------------------------------------------------------------------

class IndustrialDataset(Dataset):
    def __init__(self, meta, cfg, augment=False):
        self.meta = meta.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        path = Path(row["path"])

        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        with np.load(path) as data:
            sig = data["sig"]
            label = int(data["label"])
            rpm = float(data["rpm"])

        win_pts = int(self.cfg.window_sec * self.cfg.target_fs)

        if len(sig) > win_pts:
            start = (
                np.random.randint(0, len(sig) - win_pts)
                if self.augment
                else (len(sig) - win_pts) // 2
            )
            sig = sig[start : start + win_pts]
        else:
            sig = np.pad(sig, ((0, win_pts - len(sig)), (0, 0)))

        if self.augment:
            sig += np.random.normal(0, self.cfg.noise_std, sig.shape)

        x = compute_industrial_spectrum(sig, self.cfg.target_fs, rpm, self.cfg)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# 5. MODEL
# -----------------------------------------------------------------------------

class IndustrialCNN(nn.Module):
    def __init__(self, bins, classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 16, 15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_with_ood(self, x, thresh):
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(1)
        preds = probs.argmax(1)
        preds[entropy > thresh] = -1
        return preds, probs, entropy


# -----------------------------------------------------------------------------
# 6. TRAINING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False)
    args = parser.parse_args()

    cfg = Config(args.data_dir)
    log = setup_logger(cfg.output_dir)

    df_meta = load_metadata(cfg, log)

    log.info(f"Loaded metadata: {df_meta.shape}")
    log.info(f"Labels:\n{df_meta['label'].value_counts()}")
    log.info(f"Conditions: {df_meta['condition'].nunique()}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(
        gss.split(df_meta, df_meta["label"], groups=df_meta["condition"])
    )

    train_df = df_meta.iloc[train_idx]
    val_df = df_meta.iloc[val_idx]

    train_ds = IndustrialDataset(train_df, cfg, augment=True)
    val_ds = IndustrialDataset(val_df, cfg, augment=False)

    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(4), fill_value=0)
    )
    weights = 1.0 / (counts + 1.0)

    sampler = WeightedRandomSampler(
        [weights.loc[y] for y in train_df["label"]],
        len(train_df),
    )

    train_dl = DataLoader(train_ds, cfg.batch_size, sampler=sampler)
    val_dl = DataLoader(val_ds, cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndustrialCNN(cfg.order_bins).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

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
                p, _, _ = model.predict_with_ood(x, cfg.ood_entropy_threshold)
                preds.extend(torch.clamp(p, min=0).cpu().numpy())
                trues.extend(y.cpu().numpy())

        acc = accuracy_score(trues, preds)
        log.info(f"Epoch {ep+1} | Loss {np.mean(losses):.4f} | Acc {acc:.4f}")

    torch.save(model.state_dict(), cfg.output_dir / "industrial_model_final.pth")
    log.info("Training complete.")