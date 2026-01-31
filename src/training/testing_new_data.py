import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import hilbert, resample_poly, welch
from tqdm import tqdm
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# --------------------------------------------------
# EXACT TRAINING ARCHITECTURE (ResNet1D)
# --------------------------------------------------
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
        self.in_planes = 32
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        
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

# --------------------------------------------------
# EXACT TRAINING SIGNAL PROCESSING
# --------------------------------------------------
def safe_zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def compute_order_spectrum(sig, fs, rpm, cfg):
    sig = np.asarray(sig)
    
    # Normalize to (N, C) format
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1]:
        sig = sig.T
    
    if rpm <= 100:
        rpm = 1500.0
    shaft_hz = rpm / 60.0

    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 1024), axis=0)
    
    # Ensure Pxx is (n_freq, n_channels)
    if Pxx.ndim == 1:
        Pxx = Pxx[:, np.newaxis]
    
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    specs = []
    for ch in range(Pxx.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        s_log = np.log(s + 1e-12)
        s_norm = (s_log - s_log.min()) / (s_log.max() - s_log.min() + 1e-6)
        specs.append(s_norm)
    
    return np.stack(specs).astype(np.float32)

# --------------------------------------------------
# CONFIG (MUST MATCH TRAINING EXACTLY)
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 10.0   # Critical: matches training
        self.order_bins = 512   # Critical: matches ResNet input shape
        self.original_fs = 17850
        self.target_fs = 4000
        self.window_sec = 1.0
        self.window_pts = int(self.window_sec * self.target_fs)
        self.classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
        # Physics-based mapping from 9-class → 4-class
        self.class_mapping = {
            1: 0,  # Healthy bearing → Normal
            2: 3,  # Inner ring fault → Bearing
            3: 3,  # Inner ring fault → Bearing
            4: 3,  # Inner ring fault → Bearing
            5: 3,  # Inner ring fault → Bearing
            6: 3,  # Outer ring fault → Bearing
            7: 3,  # Outer ring fault → Bearing
            8: 3,  # Combined fault → Bearing
            9: 1   # Unbalance → Imbalance
            # Note: Misalignment (class 2) has no samples in this dataset
        }

cfg = InferenceConfig()

# --------------------------------------------------
# PATHS & OUTPUT SETUP
# --------------------------------------------------
DATA_PATH = BASE_DIR / "data" / "new_data" / "Data.mat"
TARGETS_PATH = BASE_DIR / "data" / "new_data" / "targets.mat"
MODEL_PATH = BASE_DIR / "src" / "models_sota_resnet" / "best_model_resnet.pth"
OUTPUT_DIR = BASE_DIR / "new_data_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------
# LOAD & MAP TARGETS (9-class → 4-class)
# --------------------------------------------------
print(f"Loading data from {DATA_PATH}")
data_mat = scipy.io.loadmat(DATA_PATH)
targets_mat = scipy.io.loadmat(TARGETS_PATH)

X = data_mat["Data"]    # (178, 3571)
y_original = targets_mat["Targets"].flatten().astype(int)  # (178,) values 1-9

# Map 9-class → 4-class using physics-based grouping
y_mapped = np.array([cfg.class_mapping[label] for label in y_original])

print(f"Loaded {X.shape[0]} samples")
print("\nOriginal class distribution (9 classes):")
for cls in sorted(np.unique(y_original)):
    count = np.sum(y_original == cls)
    print(f"  Class {cls}: {count:3d} samples")

print("\nMapped class distribution (4 classes):")
for cls_idx, cls_name in enumerate(cfg.classes):
    count = np.sum(y_mapped == cls_idx)
    print(f"  {cls_idx} ({cls_name:15s}): {count:3d} samples")
print(f"  ⚠️  Note: Misalignment class (idx=2) has 0 samples in this dataset")

# --------------------------------------------------
# PREPROCESSING PIPELINE (EXACT TRAINING REPLICATION)
# --------------------------------------------------
def preprocess_sample_fixed(sig_1ch, cfg, rpm=1900.0):
    # 1. Resample FIRST
    gcd = np.gcd(int(cfg.original_fs), int(cfg.target_fs))
    sig_resampled = resample_poly(sig_1ch, cfg.target_fs//gcd, cfg.original_fs//gcd)
    
    # 2. Z-score BEFORE padding (critical fix!)
    sig_norm = (sig_resampled - sig_resampled.mean()) / (sig_resampled.std() + 1e-6)
    
    # 3. Pad ONLY if necessary (after normalization)
    if len(sig_norm) < cfg.window_pts:
        pad_len = cfg.window_pts - len(sig_norm)
        sig_norm = np.pad(sig_norm, (0, pad_len), mode='constant')
    
    # 4. For single-axis: DO NOT replicate channels artificially
    # Instead: use physics-based channel synthesis OR 1-channel model
    # Option A (recommended): Use Hilbert transform for quadrature component
    analytic = hilbert(sig_norm)
    sig_3ch = np.column_stack([
        sig_norm,                # Real component (original)
        np.imag(analytic),       # Quadrature component (90° phase shift)
        np.abs(analytic)         # Envelope (amplitude modulation)
    ])
    
    # 5. Compute order spectrum (now with physically meaningful channels)
    return compute_order_spectrum(sig_3ch, cfg.target_fs, rpm, cfg)

# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = ResNet1D(input_channels=3, num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✓ Model loaded from {MODEL_PATH}")

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
print("\nRunning inference...")
predictions = []
confidences = []

for i in tqdm(range(X.shape[0]), desc="Processing samples"):
    sig_1ch = X[i].flatten()  # Force 1D
    
    # Preprocess EXACTLY like training
    spec = preprocess_sample(sig_1ch, cfg, rpm=1900.0)
    x = torch.from_numpy(spec).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()
    
    predictions.append(pred)
    confidences.append(conf)

y_pred = np.array(predictions)
y_conf = np.array(confidences)

# --------------------------------------------------
# EVALUATION WITH PROPER CLASS HANDLING
# --------------------------------------------------
print("\n" + "="*70)
print("CLASSIFICATION RESULTS (9-class dataset → 4-class model mapping)")
print("="*70)

# Identify which classes actually appear in ground truth
present_classes = sorted(np.unique(y_mapped))
present_names = [cfg.classes[i] for i in present_classes]

# Filter predictions to only include present classes for metrics
valid_mask = np.isin(y_mapped, present_classes)
y_true_valid = y_mapped[valid_mask]
y_pred_valid = y_pred[valid_mask]

# Overall accuracy
acc = accuracy_score(y_true_valid, y_pred_valid)
print(f"\nOverall Accuracy: {acc:.2%} ({np.sum(y_true_valid == y_pred_valid)}/{len(y_true_valid)})")

# Classification report (only for present classes)
print("\nClassification Report (present classes only):")
print(classification_report(
    y_true_valid,
    y_pred_valid,
    labels=present_classes,
    target_names=present_names,
    digits=4
))

# Confusion matrix (4x4 but with row/col for missing class)
cm = confusion_matrix(y_mapped, y_pred, labels=[0, 1, 2, 3])
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cfg.classes, yticklabels=cfg.classes,
            linewidths=0.5, linecolor='gray')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('True', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix (4-Class Mapping)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_4class.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix_4class.png'}")

# Per-class analysis
print("\nPer-Class Performance:")
for cls_idx in range(4):
    mask = y_mapped == cls_idx
    if np.any(mask):
        cls_acc = np.mean(y_pred[mask] == cls_idx)
        count = np.sum(mask)
        print(f"  {cfg.classes[cls_idx]:15s} (idx={cls_idx}): {cls_acc:6.2%} ({count:3d} samples)")
    else:
        print(f"  {cfg.classes[cls_idx]:15s} (idx={cls_idx}): N/A (0 samples)")

# Save detailed results
results_df = pd.DataFrame({
    'sample_id': range(len(y_mapped)),
    'original_class': y_original,
    'mapped_class': y_mapped,
    'mapped_label': [cfg.classes[i] for i in y_mapped],
    'prediction': y_pred,
    'prediction_label': [cfg.classes[i] for i in y_pred],
    'confidence': y_conf,
    'correct': y_mapped == y_pred
})
results_df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)
print(f"\n✓ Detailed results saved to {OUTPUT_DIR / 'detailed_results.csv'}")

# Physics validation: Plot average spectra per mapped class
print("\nGenerating physics validation plots...")
class_spectra = {i: [] for i in range(4)}
orders = np.linspace(0, cfg.max_order, cfg.order_bins)

for i in range(len(y_mapped)):
    sig_1ch = X[i].flatten()
    spec = preprocess_sample(sig_1ch, cfg, rpm=1900.0)
    class_spectra[y_mapped[i]].append(spec[1])  # Channel 1 (horizontal)

plt.figure(figsize=(12, 8))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']  # Green, Blue, Red, Orange
for cls_idx in range(4):
    if class_spectra[cls_idx]:
        avg_spec = np.mean(class_spectra[cls_idx], axis=0)
        plt.plot(orders, avg_spec, label=cfg.classes[cls_idx], 
                linewidth=2.5, color=colors[cls_idx], alpha=0.9)

plt.xlabel('Order (X RPM)', fontsize=13, fontweight='bold')
plt.ylabel('Normalized Amplitude', fontsize=13, fontweight='bold')
plt.title('Average Order Spectra by Mapped Class (Physics Validation)', 
         fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3, linestyle='--')
for harmonic in [1.0, 2.0, 3.0, 4.0]:
    plt.axvline(x=harmonic, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    plt.text(harmonic + 0.15, plt.ylim()[1]*0.92, f'{int(harmonic)}X', 
            color='darkred', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "physics_validation_spectra.png", dpi=150, bbox_inches='tight')
print(f"✓ Physics validation plot saved to {OUTPUT_DIR / 'physics_validation_spectra.png'}")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Dataset: Ahrar Institute (178 samples, 17.85 kHz, 1900 RPM)")
print(f"Fault Types: 9 original classes → mapped to 4 diagnostic classes")
print(f"Model: ResNet1D trained on MAFAULDA (4-class)")
print(f"Accuracy: {acc:.2%}")
print(f"\n⚠️  Critical Note: This dataset contains NO misalignment faults.")
print(f"   Model predictions for 'Misalignment' class should be treated with caution.")
print(f"\n✓ All results saved to: {OUTPUT_DIR.absolute()}")
print("="*70)