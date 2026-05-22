# ==============================================================================
# PHYSICS-AWARE FAULT DIAGNOSIS SYSTEM WITH DOMAIN ADAPTATION
# Fixed Architecture + Cross-Dataset Generalization
# ==============================================================================
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import welch, resample_poly, hilbert
from scipy.stats import entropy
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pywt  # pip install PyWavelets

# -----------------------------------------------------------------------------
# 1. CRITICAL FIX: Safe Weight Initialization (Handles None bias)
# -----------------------------------------------------------------------------
def safe_init_weights(m):
    """Safe weight initialization that handles layers without bias"""
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# -----------------------------------------------------------------------------
# 2. PHYSICS-AWARE CONFIGURATION
# -----------------------------------------------------------------------------
class BearingGeometry:
    """Bearing fault frequency calculator based on geometry"""
    def __init__(self, bore_diameter_mm=20, pitch_diameter_mm=38.5, 
                 ball_count=9, contact_angle_deg=0, rpm=1797):
        self.bore_diameter = bore_diameter_mm / 1000.0
        self.pitch_diameter = pitch_diameter_mm / 1000.0
        self.ball_count = ball_count
        self.contact_angle = np.radians(contact_angle_deg)
        self.rpm = rpm
        
    def bpfo(self):
        """Ball Pass Frequency Outer Race (Hz)"""
        return (self.ball_count / 2) * (self.rpm / 60) * (
            1 - (self.bore_diameter / self.pitch_diameter) * np.cos(self.contact_angle)
        )
    
    def bpfi(self):
        """Ball Pass Frequency Inner Race (Hz)"""
        return (self.ball_count / 2) * (self.rpm / 60) * (
            1 + (self.bore_diameter / self.pitch_diameter) * np.cos(self.contact_angle)
        )

class Config:
    def __init__(self, raw_dir=None):
        script_dir = Path(raw_dir).resolve().parent.parent.parent if raw_dir else Path(__file__).resolve().parent.parent.parent
        self.raw_data_dir = script_dir / "data" / "raw_mafulda"
        self.processed_dir = script_dir / "data" / "4k_mafulda_structured"
        self.output_dir = Path("models_physics_aware_v2")
        
        # Sampling rates
        self.orig_fs = 50000
        self.target_fs = 4000
        
        # Physics constraints
        self.max_order = 12.0   # Extended for bearing harmonics
        self.order_bins = 512
        self.window_sec = 1.0
        self.min_signal_sec = 0.5
        
        # Bearing geometry (MAFAULDA typical: NU 204 ECP)
        self.bearing = BearingGeometry(
            bore_diameter_mm=20,
            pitch_diameter_mm=38.5,
            ball_count=9,
            contact_angle_deg=0
        )
        
        # Training
        self.batch_size = 32
        self.epochs = 50
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Physics-aware augmentation
        self.rpm_jitter = 0.15
        self.amp_jitter = 0.2
        self.noise_std = 0.03

def setup_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train_log.txt", mode="w")]
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 3. PHYSICS-CORRECT SIGNAL PROCESSING (Critical Fixes)
# -----------------------------------------------------------------------------
def safe_zscore(x, axis=0):
    """Normalize BEFORE padding (critical fix for short signals)"""
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + 1e-6
    return (x - mean) / std

def resample_sig(x, orig_fs, target_fs):
    if int(orig_fs) == int(target_fs): 
        return x.astype(np.float32)
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(x, int(target_fs//gcd), int(orig_fs//gcd), axis=0).astype(np.float32)

def get_rpm(tach, fs):
    """Robust RPM estimation with fallback"""
    tach = tach - np.mean(tach)
    pulses = np.where(np.diff((tach > 0.5).astype(int)) > 0)[0]
    if len(pulses) < 2: 
        return 1500.0
    avg_diff = np.mean(np.diff(pulses))
    return (fs / avg_diff) * 60.0 if avg_diff > 0 else 1500.0

def decompose_signal(sig, fs, method='wavelet'):
    """
    Physics-aware component separation for single-axis data
    Creates physically meaningful 3-channel representation
    """
    if method == 'hilbert':
        analytic = hilbert(sig)
        return np.column_stack([
            sig,                    # Original vibration
            np.imag(analytic),      # Quadrature component (90° phase shift)
            np.abs(analytic)        # Envelope (bearing impacts)
        ])
    
    elif method == 'wavelet':
        # Extract high-frequency fault components using wavelet decomposition
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        # Zero out low-frequency approximation coefficients
        coeffs[0] = np.zeros_like(coeffs[0])
        hf_signal = pywt.waverec(coeffs, 'db4')
        # Ensure same length as original
        hf_signal = hf_signal[:len(sig)]
        return np.column_stack([
            sig,                    # Original signal
            hf_signal,              # High-frequency fault components
            np.abs(hilbert(sig))    # Envelope for impact detection
        ])
    
    else:  # 'none' - for true 3-axis data
        return np.column_stack([
            sig,
            np.roll(sig, shift=5),   # Small phase shift to avoid perfect correlation
            np.roll(sig, shift=-5)
        ])

def compute_order_spectrum(sig, fs, rpm, cfg, bearing_geom=None):
    """Physics-enhanced order spectrum computation"""
    if rpm <= 100 or rpm > 5000:
        rpm = 1500.0
    
    shaft_hz = rpm / 60.0
    nperseg = min(len(sig), 2048)
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2, axis=0)
    
    # Handle 1D Pxx (single channel)
    if Pxx.ndim == 1:
        Pxx = Pxx[:, np.newaxis]
    
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    specs = []
    for ch in range(Pxx.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        s_log = np.log(s + 1e-12)
        # Robust normalization using percentiles (handles outliers)
        s_norm = (s_log - np.percentile(s_log, 5)) / (np.percentile(s_log, 95) - np.percentile(s_log, 5) + 1e-6)
        s_norm = np.clip(s_norm, 0, 1)
        specs.append(s_norm)
    
    return np.stack(specs).astype(np.float32)

# -----------------------------------------------------------------------------
# 4. DATASET WITH PHYSICS AUGMENTATION
# -----------------------------------------------------------------------------
class VibrationDataset(Dataset):
    def __init__(self, metadata_df, cfg, augment=False, decompose_method='wavelet'):
        self.meta = metadata_df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.decompose_method = decompose_method
        
    def __len__(self):
        return len(self.meta)
    
    def _physics_augment(self, sig, rpm):
        """Physics-aware augmentation preserving fault signatures"""
        if not self.augment:
            return sig, rpm
        
        # RPM jittering (simulate MAFAULDA's variable RPM)
        rpm_jittered = rpm * np.random.uniform(1 - self.cfg.rpm_jitter, 1 + self.cfg.rpm_jitter)
        
        # Amplitude scaling (sensor sensitivity mismatch)
        sig_aug = sig * np.random.uniform(1 - self.cfg.amp_jitter, 1 + self.cfg.amp_jitter)
        
        # Add minimal noise
        sig_aug += np.random.normal(0, self.cfg.noise_std * np.std(sig_aug), sig_aug.shape)
        
        return sig_aug, rpm_jittered
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        
        with np.load(row['path']) as data:
            sig = data['sig']
            label = int(data['label'])
            rpm = float(data['rpm'])
        
        # Ensure 2D format (N, C)
        if sig.ndim == 1:
            sig = sig[:, np.newaxis]
        
        # Physics augmentation
        sig, rpm = self._physics_augment(sig, rpm)
        
        # Window selection
        win_pts = int(self.cfg.window_sec * self.cfg.target_fs)
        if len(sig) > win_pts:
            start = np.random.randint(0, len(sig) - win_pts) if self.augment else (len(sig) - win_pts) // 2
            sig_win = sig[start:start + win_pts]
        else:
            # CRITICAL FIX: Normalize BEFORE padding
            sig_norm = safe_zscore(sig, axis=0)
            pad_len = win_pts - len(sig_norm)
            sig_win = np.pad(sig_norm, ((0, pad_len), (0, 0)), mode='constant')
        
        # Component decomposition for single-axis data
        if sig_win.shape[1] == 1:
            sig_win = decompose_signal(sig_win[:, 0], self.cfg.target_fs, self.decompose_method)
        
        # Final per-channel normalization
        sig_win = safe_zscore(sig_win, axis=0)
        
        # Order spectrum
        x = compute_order_spectrum(sig_win, self.cfg.target_fs, rpm, self.cfg)
        
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label)

# -----------------------------------------------------------------------------
# 5. ENHANCED ARCHITECTURE WITH SE BLOCKS (Fixed Initialization)
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel-wise attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PhysicsAwareResNet(nn.Module):
    """
    Enhanced ResNet with SE Blocks and physics-aware design
    FIXED: Safe weight initialization handles None bias
    """
    def __init__(self, input_channels=3, num_classes=4):
        super().__init__()
        self.in_planes = 32
        
        # Initial layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with SE blocks
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
        # CRITICAL FIX: Use safe initialization that handles None bias
        self.apply(safe_init_weights)
    
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
        x = self.dropout(x)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------------
# 6. TEST-TIME ADAPTATION (Critical for Cross-Dataset Generalization)
# -----------------------------------------------------------------------------
class TestTimeAdapter:
    """
    Physics-aware test-time adaptation to handle domain shift
    """
    def __init__(self, model, cfg, n_augmentations=5):
        self.model = model
        self.cfg = cfg
        self.n_augmentations = n_augmentations
        self.original_bn_stats = self._save_bn_stats()
    
    def _save_bn_stats(self):
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'momentum': module.momentum
                }
        return stats
    
    def _restore_bn_stats(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d) and name in self.original_bn_stats:
                stats = self.original_bn_stats[name]
                module.running_mean.copy_(stats['running_mean'])
                module.running_var.copy_(stats['running_var'])
                module.momentum = stats['momentum']
    
    def predict_with_adaptation(self, x_tensor, rpm=1797.0, device='cpu'):
        """
        Ensemble prediction with physics-aware augmentations
        Returns: (prediction, confidence, uncertainty)
        """
        self.model.train()  # Enable BN update temporarily
        
        predictions = []
        for _ in range(self.n_augmentations):
            # Apply physics-aware augmentation to spectrum
            x_np = x_tensor.cpu().numpy().copy()
            
            # Small harmonic shift to simulate RPM variation
            shift = int(np.random.uniform(-3, 3))
            if shift != 0:
                x_np = np.roll(x_np, shift, axis=2)
            
            # Amplitude variation
            x_np *= np.random.uniform(0.95, 1.05, size=x_np.shape)
            
            x_aug = torch.from_numpy(x_np).float().to(device)
            
            with torch.no_grad():
                logits = self.model(x_aug)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        self.model.eval()
        self._restore_bn_stats()
        
        # Ensemble results
        predictions = np.stack(predictions, axis=0)
        mean_probs = predictions.mean(axis=0)
        pred = np.argmax(mean_probs, axis=1)
        confidence = np.max(mean_probs, axis=1)
        uncertainty = entropy(predictions, axis=2).mean(axis=0)
        
        return pred, confidence, uncertainty

# -----------------------------------------------------------------------------
# 7. DATA PROCESSING (Minimal version - use your existing process_data)
# -----------------------------------------------------------------------------
def process_data(cfg, log):
    """Minimal version - replace with your existing process_data function"""
    meta_path = cfg.processed_dir / "metadata.csv"
    if meta_path.exists():
        log.info("Found metadata.csv. Loading...")
        return pd.read_csv(meta_path)
    
    log.error("metadata.csv not found! Please run data preprocessing first.")
    log.error("Use your existing process_data function from the training script.")
    raise FileNotFoundError(f"metadata.csv not found at {meta_path}")

# -----------------------------------------------------------------------------
# 8. TRAINING PIPELINE
# -----------------------------------------------------------------------------
def train_model(cfg, df_meta, log):
    # Split with group awareness
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_meta, df_meta['label'], df_meta['original_path']))
    
    # Create datasets with wavelet decomposition
    train_ds = VibrationDataset(df_meta.iloc[train_idx], cfg, augment=True, decompose_method='wavelet')
    val_ds = VibrationDataset(df_meta.iloc[val_idx], cfg, augment=False, decompose_method='wavelet')
    
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    model = PhysicsAwareResNet(input_channels=3, num_classes=4).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Class weights for imbalance
    train_labels = df_meta.iloc[train_idx]['label'].values
    class_counts = np.bincount(train_labels, minlength=4)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    log.info("Starting Training (Physics-Aware ResNet)...")
    best_acc = 0.0
    
    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        log.info(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {train_loss/len(train_dl):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            cfg.output_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), cfg.output_dir / "best_model_physics_aware.pth")
            log.info(f"  → New best model saved (acc={acc:.4f})")
    
    return model, best_acc

# -----------------------------------------------------------------------------
# 9. CROSS-DATASET INFERENCE
# -----------------------------------------------------------------------------
def preprocess_cross_dataset_sample(sig_1ch, cfg, rpm=1797.0, decompose_method='wavelet'):
    """
    Physics-correct preprocessing for cross-dataset inference
    CRITICAL FIXES:
      1. Normalize BEFORE padding
      2. Wavelet decomposition for single-axis data
      3. Robust spectral normalization
    """
    # 1. Resample
    gcd = np.gcd(int(cfg.original_fs), int(cfg.target_fs))
    sig_resampled = resample_poly(
        sig_1ch, 
        cfg.target_fs // gcd, 
        cfg.original_fs // gcd
    )
    
    # 2. CRITICAL: Normalize BEFORE padding
    sig_norm = safe_zscore(sig_resampled)
    
    # 3. Pad after normalization
    win_pts = int(cfg.window_sec * cfg.target_fs)
    if len(sig_norm) < win_pts:
        pad_len = win_pts - len(sig_norm)
        sig_norm = np.pad(sig_norm, (0, pad_len), mode='constant')
    else:
        start = (len(sig_norm) - win_pts) // 2
        sig_norm = sig_norm[start:start + win_pts]
    
    # 4. Component decomposition
    sig_3ch = decompose_signal(sig_norm, cfg.target_fs, decompose_method)
    
    # 5. Final normalization
    sig_3ch = safe_zscore(sig_3ch, axis=0)
    
    # 6. Order spectrum
    spec = compute_order_spectrum(sig_3ch, cfg.target_fs, rpm, cfg)
    
    return spec

def run_cross_dataset_inference(model_path, data_path, targets_path, 
                               dataset_name="unknown", rpm=1797.0,
                               decompose_method='wavelet'):
    """
    Cross-dataset inference with physics validation
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    OUTPUT_DIR = BASE_DIR / f"{dataset_name}_results_v2"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.original_fs = 17850  # Adjust per dataset
    
    # Load model
    model = PhysicsAwareResNet(input_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    import scipy.io
    data_mat = scipy.io.loadmat(data_path)
    targets_mat = scipy.io.loadmat(targets_path)
    
    X = data_mat["Data"]
    y_original = targets_mat["Targets"].flatten().astype(int)
    
    # Class mapping (9-class → 4-class)
    class_mapping = {1:0, 2:3, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:1}
    y_mapped = np.array([class_mapping.get(label, 3) for label in y_original])
    
    # Setup TTA
    tta = TestTimeAdapter(model, cfg, n_augmentations=7)
    
    # Inference
    predictions, confidences, uncertainties = [], [], []
    
    for i in tqdm(range(X.shape[0]), desc=f"Processing {dataset_name}"):
        sig_1ch = X[i].flatten()
        spec = preprocess_cross_dataset_sample(sig_1ch, cfg, rpm, decompose_method)
        x_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        
        pred, conf, unc = tta.predict_with_adaptation(x_tensor, rpm, device)
        predictions.append(pred[0])
        confidences.append(conf[0])
        uncertainties.append(unc[0])
    
    # Evaluation
    y_pred = np.array(predictions)
    y_conf = np.array(confidences)
    y_unc = np.array(uncertainties)
    
    uncertainty_threshold = np.percentile(y_unc, 75)
    confident_mask = y_unc < uncertainty_threshold
    
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET INFERENCE: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Total samples: {len(y_mapped)}")
    print(f"Confident predictions: {np.sum(confident_mask)}/{len(y_mapped)}")
    
    acc_overall = accuracy_score(y_mapped, y_pred)
    acc_confident = accuracy_score(y_mapped[confident_mask], y_pred[confident_mask]) if np.any(confident_mask) else 0.0
    
    print(f"\nOverall Accuracy: {acc_overall:.2%}")
    print(f"Confident Subset Accuracy: {acc_confident:.2%}")
    
    # Classification report
    present_classes = sorted(np.unique(y_mapped))
    present_names = ["Normal", "Imbalance", "Misalignment", "Bearing"]
    
    print("\nClassification Report:")
    print(classification_report(
        y_mapped, 
        y_pred, 
        labels=present_classes,
        target_names=[present_names[i] for i in present_classes],
        digits=4
    ))
    
    # Save results
    results_df = pd.DataFrame({
        'sample_id': range(len(y_mapped)),
        'original_class': y_original,
        'mapped_class': y_mapped,
        'prediction': y_pred,
        'confidence': y_conf,
        'uncertainty': y_unc,
        'is_confident': confident_mask
    })
    results_df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR.absolute()}")
    return results_df

# -----------------------------------------------------------------------------
# 10. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-Aware Fault Diagnosis")
    parser.add_argument("--mode", choices=["train", "test_ahar"], required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    
    cfg = Config(args.data_dir)
    log = setup_logger(cfg.output_dir)
    
    if args.mode == "train":
        log.info("Processing MAFAULDA data...")
        df_meta = process_data(cfg, log)  # Use your existing process_data function
        
        log.info("Training physics-aware model...")
        model, best_acc = train_model(cfg, df_meta, log)
        log.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")
        
    elif args.mode == "test_ahar":
        from tqdm import tqdm  # Import here to avoid dependency in training mode
        
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        MODEL_PATH = BASE_DIR / "src" / "models_physics_aware_v2" / "best_model_physics_aware.pth"
        DATA_PATH = BASE_DIR / "data" / "new_data" / "Data.mat"
        TARGETS_PATH = BASE_DIR / "data" / "new_data" / "targets.mat"
        
        if not MODEL_PATH.exists():
            log.error(f"Model not found at {MODEL_PATH}")
            log.error("Please train the model first using: --mode train")
            sys.exit(1)
        
        run_cross_dataset_inference(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            targets_path=TARGETS_PATH,
            dataset_name="ahar_institute",
            rpm=1900.0,
            decompose_method='wavelet'
        )