# train_mafaulda_order_cnn_v2.py
import sys
import json
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from scipy.signal import welch, butter, filtfilt, decimate, hilbert
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -----------------------
# Signal processing utils
# -----------------------
def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.1, low)
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)

def safe_zscore(x: np.ndarray, axis=None, eps: float = 1e-6) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)

def estimate_rpm_from_tach(tach: np.ndarray, fs: float, rpm_min=600, rpm_max=4000) -> float:
    """Robust RPM estimate:
    1) rising-edge timing
    2) fallback to Welch peak
    """
    t = tach.astype(np.float64)
    t = t - np.median(t)

    # clean it a bit to stabilize thresholding
    try:
        t_f = bandpass(t[:, None], fs, low=5, high=200, order=3).squeeze(1)
    except Exception:
        t_f = t

    thr = 0.5 * np.std(t_f)
    if not np.isfinite(thr) or thr <= 0:
        thr = np.percentile(np.abs(t_f), 75) * 0.3

    edges = np.where((t_f[:-1] < thr) & (t_f[1:] >= thr))[0]
    if len(edges) >= 5:
        period = np.median(np.diff(edges)) / fs
        if period > 0:
            rpm = 60.0 / period
            return float(np.clip(rpm, rpm_min, rpm_max))

    # fallback: spectral peak
    nper = min(8192, len(t_f))
    f, Pxx = welch(t_f, fs=fs, nperseg=nper)
    band = (f >= rpm_min / 60.0) & (f <= rpm_max / 60.0)
    if not np.any(band):
        return float((rpm_min + rpm_max) / 2.0)
    rpm = f[band][np.argmax(Pxx[band])] * 60.0
    return float(np.clip(rpm, rpm_min, rpm_max))

BEARING_CHAR_ORDERS = {
    "FTF": 0.3750,
    "BSF": 1.8710,
    "BPFO": 2.9980,
    "BPFI": 5.0020,
}

def harmonic_energy_ratio(order_axis: np.ndarray, spectrum_lin: np.ndarray, order: float, width: float = 0.05) -> float:
    m = (order_axis >= order - width) & (order_axis <= order + width)
    e = float(spectrum_lin[m].sum())
    tot = float(spectrum_lin.sum()) + 1e-12
    return e / tot

def make_envelope(sig_raw: np.ndarray, fs_raw: float, env_low: float, env_high: float) -> np.ndarray:
    """Bearing-friendly: bandpass -> Hilbert envelope."""
    x = bandpass(sig_raw, fs_raw, env_low, env_high, order=4)
    env = np.abs(hilbert(x, axis=0))
    return env.astype(np.float32)

def make_order_representation(
    sig_ds: np.ndarray,
    fs: float,
    rpm: float,
    max_order: float,
    order_bins: int,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (order_map: CxB, feats: D)."""
    shaft_hz = max(rpm / 60.0, 1e-3)
    order_axis = np.linspace(0.0, max_order, order_bins, endpoint=True)

    maps: List[np.ndarray] = []
    feats: List[float] = []

    # time stats per channel (robust-ish)
    for ch in range(sig_ds.shape[1]):
        x = sig_ds[:, ch]
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        crest = float(np.max(np.abs(x)) / (rms + 1e-12))
        z = (x - x.mean()) / (x.std() + 1e-6)
        kurt = float(np.mean(z ** 4))
        feats.extend([np.log(rms + 1e-12), np.log(crest + 1e-12), np.log(kurt + 1e-12)])

    for ch in range(sig_ds.shape[1]):
        f, Pxx = welch(sig_ds[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
        orders = f / shaft_hz
        mask = (orders > 0) & (orders <= max_order)



        spectrum_lin = np.interp(order_axis, orders[mask], Pxx[mask]) if mask.sum() >= 8 else np.zeros_like(order_axis)
        spectrum_log = np.log(spectrum_lin + 1e-12).astype(np.float32)

        # normalize across order bins (gain/mounting robustness)
        spectrum_log = safe_zscore(spectrum_log, axis=0).astype(np.float32)
        maps.append(spectrum_log)

        # energies: 1x/2x/3x + bearing characteristic orders
        for o in (1.0, 2.0, 3.0, BEARING_CHAR_ORDERS["FTF"], BEARING_CHAR_ORDERS["BSF"], BEARING_CHAR_ORDERS["BPFO"], BEARING_CHAR_ORDERS["BPFI"]):
            r = harmonic_energy_ratio(order_axis, spectrum_lin, o, width=0.05)
            feats.append(float(np.log(r + 1e-12)))

        # generic shape stats
        tot = float(spectrum_lin.sum()) + 1e-12
        centroid = float((order_axis * spectrum_lin).sum() / tot)
        flatness = float(np.exp(np.mean(np.log(spectrum_lin + 1e-12))) / (np.mean(spectrum_lin) + 1e-12))
        feats.extend([centroid, np.log(flatness + 1e-12)])

    return np.stack(maps, axis=0).astype(np.float32), np.array(feats, dtype=np.float32)

# -----------------------
# Dataset indexing/labels
# -----------------------
def parse_label_from_path(path: Path, granularity: str = "coarse") -> Optional[str]:
    """Adapt this to your folder conventions."""
    s = str(path).lower().replace("\\", "/")

    if "imbalance" in s:
        base = "imbalance"
    elif "horizontal-misalignment" in s or ("horizontal" in s and "misalignment" in s):
        base = "misalignment_horizontal"
    elif "vertical-misalignment" in s or ("vertical" in s and "misalignment" in s):
        base = "misalignment_vertical"
    elif "underhang" in s or "overhang" in s or "bearing" in s:
        # try to parse element too
        pos = "underhang" if "underhang" in s else ("overhang" if "overhang" in s else "bearing")
        if any(k in s for k in ["outer", "outer-race", "outer_track"]):
            elem = "outer"
        elif any(k in s for k in ["inner", "inner-race", "inner_track"]):
            elem = "inner"
        elif any(k in s for k in ["ball", "rolling", "element"]):
            elem = "ball"
        elif "cage" in s:
            elem = "cage"
        else:
            elem = "unknown"
        base = f"bearing_{pos}_{elem}"
    elif "/normal" in s or s.endswith("normal") or "normal" in s:
        base = "normal"
    else:
        return None

    if granularity == "coarse":
        if base.startswith("misalignment"):
            return "misalignment"
        if base.startswith("bearing"):
            return "bearing"
        return base

    return base

def build_index(data_dir: Path, granularity: str) -> pd.DataFrame:
    rows = []
    for p in sorted(data_dir.rglob("*.csv")):
        label = parse_label_from_path(p, granularity=granularity)
        if label is None:
            continue
        rows.append({"path": str(p), "label": label})
    if not rows:
        raise RuntimeError(f"No labeled CSV files found under {data_dir}")
    df = pd.DataFrame(rows)
    df["file_id"] = np.arange(len(df), dtype=np.int32)
    return df

def read_mafaulda_csv(csv_path: Path, sensors: str) -> Tuple[np.ndarray, np.ndarray]:
    """MaFaulDa CSVs: 8 columns: tach + 6 accel + mic. :contentReference[oaicite:3]{index=3}"""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 8:
        raise ValueError(f"{csv_path} has {df.shape[1]} columns, expected 8")

    tach = df.iloc[:, 0].values.astype(np.float32)
    under = df.iloc[:, 1:4].values.astype(np.float32)
    over  = df.iloc[:, 4:7].values.astype(np.float32)

    if sensors == "underhang":
        acc = under
    elif sensors == "overhang":
        acc = over
    elif sensors == "both":
        acc = np.concatenate([under, over], axis=1)
    else:
        raise ValueError(f"Unknown sensors={sensors}")

    return tach, acc

# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    data_dir: Path
    output_dir: Path



    raw_fs: int = 50000
    downsample_factor: int = 10      # => 5 kHz
    low_hz: float = 10.0
    high_hz: float = 2000.0

    use_envelope: bool = True
    env_low_hz: float = 2000.0
    env_high_hz: float = 10000.0

    window_seconds: float = 1.0
    hop_seconds: float = 0.5

    max_order: float = 10.0
    order_bins: int = 256
    nperseg: int = 1024
    noverlap: int = 512

    label_granularity: str = "coarse"
    sensors: str = "underhang"

    seed: int = 42
    batch_size: int = 128
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 12
    label_smoothing: float = 0.05
    num_workers: int = 2

    # order-domain augmentation
    aug_prob: float = 0.8
    aug_noise_std: float = 0.08
    aug_freq_mask_max_width: int = 24
    aug_channel_dropout_prob: float = 0.15
    aug_roll_max: int = 6

    def __post_init__(self):
        self.fs = float(self.raw_fs) / float(self.downsample_factor)

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# -----------------------
# Feature extraction cache
# -----------------------
def extract_features(cfg: Config, cache_path: Path):
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["F"], d["y"], d["groups"], list(d["class_names"])

    idx = build_index(cfg.data_dir, cfg.label_granularity)
    class_names = sorted(idx["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}

    X_list, F_list, y_list, g_list = [], [], [], []

    win_len = int(cfg.raw_fs * cfg.window_seconds)
    hop = int(cfg.raw_fs * cfg.hop_seconds)

    for row in idx.itertuples(index=False):
        csv_path = Path(row.path)
        file_id = int(row.file_id)
        label_id = class_to_id[row.label]

        tach, acc = read_mafaulda_csv(csv_path, cfg.sensors)
        rpm = estimate_rpm_from_tach(tach, fs=cfg.raw_fs, rpm_min=600, rpm_max=4000)

        n = len(acc)
        if n < win_len:
            continue

        for start in range(0, n - win_len + 1, hop):
            acc_w = acc[start:start + win_len]

            # envelope (bearing-friendly) before decimation
            if cfg.use_envelope:
                env = make_envelope(acc_w, cfg.raw_fs, cfg.env_low_hz, cfg.env_high_hz)
                env_ds = decimate(env, cfg.downsample_factor, axis=0, zero_phase=True)
                env_ds = safe_zscore(env_ds, axis=0).astype(np.float32)

            # base path: decimate + bandpass + normalize
            acc_ds = decimate(acc_w, cfg.downsample_factor, axis=0, zero_phase=True)
            acc_ds = bandpass(acc_ds, cfg.fs, cfg.low_hz, cfg.high_hz)
            acc_ds = safe_zscore(acc_ds, axis=0).astype(np.float32)

            omap, feats = make_order_representation(acc_ds, cfg.fs, rpm, cfg.max_order, cfg.order_bins, cfg.nperseg, cfg.noverlap)

            if cfg.use_envelope:
                emap, efeats = make_order_representation(env_ds, cfg.fs, rpm, cfg.max_order, cfg.order_bins, cfg.nperseg, cfg.noverlap)
                omap = np.concatenate([omap, emap], axis=0)
                feats = np.concatenate([feats, efeats], axis=0)

            X_list.append(omap)
            F_list.append(feats)
            y_list.append(label_id)
            g_list.append(file_id)

    X = np.stack(X_list).astype(np.float32)  # (N, C, B)
    F = np.stack(F_list).astype(np.float32)  # (N, D)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, F=F, y=y, groups=groups, class_names=np.array(class_names, dtype=object))
    return X, F, y, groups, class_names



# -----------------------
# Torch dataset + augmentation
# -----------------------
class OrderDataset(Dataset):
    def __init__(self, X, F, y, feat_mean, feat_std, cfg: Config, augment: bool):
        self.X, self.F, self.y = X, F, y
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.cfg = cfg
        self.augment = augment

    def __len__(self): return len(self.y)

    def _augment_x(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.cfg.aug_prob:
            return x
        x = x.copy()

        r = np.random.randint(-self.cfg.aug_roll_max, self.cfg.aug_roll_max + 1)
        if r != 0:
            x = np.roll(x, shift=r, axis=1)

        width = np.random.randint(0, self.cfg.aug_freq_mask_max_width + 1)
        if width > 0:
            start = np.random.randint(0, max(1, x.shape[1] - width))
            x[:, start:start + width] = x.min()

        x += np.random.normal(0.0, self.cfg.aug_noise_std, size=x.shape).astype(np.float32)

        if np.random.rand() < self.cfg.aug_channel_dropout_prob:
            ch = np.random.randint(0, x.shape[0])
            x[ch, :] = 0.0

        return x.astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i]
        f = (self.F[i] - self.feat_mean) / (self.feat_std + 1e-6)
        if self.augment:
            x = self._augment_x(x)
            f = f + np.random.normal(0.0, 0.02, size=f.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(f.astype(np.float32)), torch.tensor(self.y[i], dtype=torch.long)

# -----------------------
# Model
# -----------------------
class OrderNet(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int, num_classes: int):
        super().__init__()
        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=k//2, bias=False),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.SiLU(),
            )

        self.backbone = nn.Sequential(
            block(in_ch, 64, 7),
            nn.MaxPool1d(2),
            block(64, 128, 5),
            nn.MaxPool1d(2),
            block(128, 256, 3),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, feats):
        z = self.backbone(x).squeeze(-1)
        z = torch.cat([z, feats], dim=1)
        return self.head(z)

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, fb, yb in loader:
        xb, fb = xb.to(device), fb.to(device)
        pred = model(xb, fb).argmax(1).cpu().numpy()
        ys.append(yb.numpy()); ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="models_out")
    ap.add_argument("--label_granularity", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--sensors", choices=["underhang", "overhang", "both"], default="underhang")
    ap.add_argument("--no_envelope", action="store_true")
    args = ap.parse_args()



    cfg = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        label_granularity=args.label_granularity,
        sensors=args.sensors,
        use_envelope=not args.no_envelope,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(cfg.output_dir / "training.log", mode="w")],
    )
    log = logging.getLogger(__name__)
    log.info("Config:\n%s", json.dumps(asdict(cfg), indent=2, default=str))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    cache_path = cfg.output_dir / f"features_{cfg.label_granularity}_{cfg.sensors}_{'env' if cfg.use_envelope else 'noenv'}.npz"
    X, F, y, groups, class_names = extract_features(cfg, cache_path)

    # train/val/test with group split (no leakage)
    gss1 = GroupShuffleSplit(test_size=0.2, random_state=cfg.seed)
    trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))

    gss2 = GroupShuffleSplit(test_size=0.2, random_state=cfg.seed + 1)
    tr_rel, va_rel = next(gss2.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_rel]
    val_idx   = trainval_idx[va_rel]

    Xtr, Ftr, ytr = X[train_idx], F[train_idx], y[train_idx]
    Xva, Fva, yva = X[val_idx],   F[val_idx],   y[val_idx]
    Xte, Fte, yte = X[test_idx],  F[test_idx],  y[test_idx]

    feat_mean = Ftr.mean(0).astype(np.float32)
    feat_std  = Ftr.std(0).astype(np.float32) + 1e-6

    train_ds = OrderDataset(Xtr, Ftr, ytr, feat_mean, feat_std, cfg, augment=True)
    val_ds   = OrderDataset(Xva, Fva, yva, feat_mean, feat_std, cfg, augment=False)
    test_ds  = OrderDataset(Xte, Fte, yte, feat_mean, feat_std, cfg, augment=False)

    # balanced sampling
    counts = np.bincount(ytr, minlength=len(class_names))
    sample_w = 1.0 / (counts[ytr] + 1e-6)
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.float32), num_samples=len(ytr), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = OrderNet(in_ch=X.shape[1], feat_dim=F.shape[1], num_classes=len(class_names)).to(device)

    class_w = compute_class_weights(ytr, len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    patience_ctr = 0
    ckpt_path = cfg.output_dir / "best_model.pth"

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for xb, fb, yb in train_dl:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb, fb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_m = evaluate(model, val_dl, device)
        sched.step(val_m["macro_f1"])
        log.info("Epoch %03d | loss %.4f | val acc %.3f | val macroF1 %.3f", ep+1, np.mean(losses), val_m["acc"], val_m["macro_f1"])



        if val_m["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val_m["macro_f1"]
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "cfg": asdict(cfg),
                    "feat_mean": feat_mean,
                    "feat_std": feat_std,
                  
                },
                ckpt_path,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                log.warning("Early stopping.")
                break

    # final test (ONLY ONCE)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te = evaluate(model, test_dl, device)
    log.info("TEST acc %.3f | TEST macroF1 %.3f", te["acc"], te["macro_f1"])
    log.info("Confusion matrix:\n%s", te["cm"])
    log.info("Report:\n%s", classification_report(te["y_true"], te["y_pred"], target_names=class_names))

if __name__ == "__main__":
    main()




    # python training/train_cnn_new_data.py --data_dir data/raw_mafulda --output_dir ./models --label_granularity coarse --sensors underhang
