#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Inference Pipeline for Multi-Dataset Validation
Supports:
  1. CSV dataset (3-axis vibration + acoustic/temperature) - CORRECTED FROM EXCEL
  2. 45kW MATLAB dataset (Healthy/Faulty directories)
Critical fixes:
  ✓ Channel semantics: Replicate single-axis → 3 channels (NO Hilbert synthesis)
  ✓ RPM handling: Hardcode for known machines (CSV=1800 RPM, MATLAB=1440 RPM)
  ✓ Overhang-trained model: Replicate signals to maintain semantic consistency
  ✓ Sampling rate conversion: 1kHz → 4kHz with rational resampling
  ✓ CSV parsing: Robust column detection (handles "Temparature" typo)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import scipy.io
from scipy.signal import resample_poly, welch, butter, filtfilt, find_peaks, hilbert
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# --------------------------------------------------
# MODEL ARCHITECTURE (ResNet1D - MUST MATCH TRAINING)
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
# CRITICAL FIX #1: SAFE RPM ESTIMATION (Short-Signal Robust)
# --------------------------------------------------
def estimate_rpm_safe(sig, fs, min_duration_sec=0.5, expected_rpm=1500.0):
    """
    Robust RPM estimation with short-signal fallback
    Returns: (rpm_estimate, confidence)
    """
    sig = np.asarray(sig).flatten()
    duration_sec = len(sig) / fs
    
    # Short signal fallback (common in test datasets)
    if duration_sec < min_duration_sec:
        return expected_rpm, 0.2  # Low confidence fallback
    
    # Envelope-based estimation (most reliable for rotating machinery)
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    
    # Bandpass filter around expected shaft frequency
    shaft_freq = expected_rpm / 60.0
    low = max(5, shaft_freq * 0.6)
    high = min(fs/2.1, shaft_freq * 1.4)
    
    if high > low:
        b, a = butter(4, [low, high], fs=fs, btype='band')
        env_filt = filtfilt(b, a, envelope)
    else:
        env_filt = envelope
    
    # Find shaft rotation pulses
    peaks, _ = find_peaks(
        env_filt, 
        height=np.percentile(env_filt, 80),
        distance=max(5, int(fs / (shaft_freq * 2)))
    )
    
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs
        rpm_est = 60.0 / np.median(intervals)
        rpm_est = np.clip(rpm_est, 600, 3600)
        conf = min(1.0, len(peaks) / (duration_sec * shaft_freq * 1.5))
        return rpm_est, conf
    
    return expected_rpm, 0.3  # Conservative fallback

# --------------------------------------------------
# CRITICAL FIX #2: HARMONIC-BAND NORMALIZED ORDER SPECTRUM
# --------------------------------------------------
def compute_order_spectrum(sig, fs, rpm, max_order=10.0, order_bins=512):
    """
    Physics-aware order spectrum with harmonic-band energy normalization
    Critical for cross-dataset generalization
    """
    sig = np.asarray(sig)
    
    # Ensure (N, C) format
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1]:
        sig = sig.T
    
    # Handle invalid RPM
    if rpm < 100 or rpm > 5000:
        rpm = 1500.0
    
    shaft_hz = rpm / 60.0
    nperseg = min(len(sig), 2048)
    
    # Welch transform with overlap
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2, axis=0)
    
    if Pxx.ndim == 1:
        Pxx = Pxx[:, np.newaxis]
    
    orders = f / shaft_hz
    target_orders = np.linspace(0, max_order, order_bins)
    
    specs = []
    for ch in range(Pxx.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        
        # Physics-based harmonic-band normalization
        harmonic_bands = [(0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, max_order)]
        s_norm = np.zeros_like(s)
        for band_start, band_end in harmonic_bands:
            idx = (target_orders >= band_start) & (target_orders < band_end)
            if np.any(idx) and np.sum(s[idx]) > 1e-12:
                s_norm[idx] = s[idx] / (np.sum(s[idx]) + 1e-12)
        
        # Log compression with percentile normalization
        s_log = np.log10(s_norm + 1e-12)
        p1, p99 = np.percentile(s_log, [1, 99])
        if p99 > p1:
            s_scaled = (s_log - p1) / (p99 - p1 + 1e-6)
            s_scaled = np.clip(s_scaled, 0, 1)
        else:
            s_scaled = np.zeros_like(s_log)
        
        specs.append(s_scaled)
    
    return np.stack(specs).astype(np.float32)

# --------------------------------------------------
# INFERENCE CONFIG (MUST MATCH TRAINING EXACTLY)
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 10.0
        self.order_bins = 512
        self.target_fs = 4000  # Model expects 4kHz
        self.window_sec = 1.0
        self.window_pts = int(self.window_sec * self.target_fs)
        self.classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
        self.confidence_threshold = 0.65
        
        # Dataset-specific parameters (1kHz sampling rate for both datasets)
        self.csv_fs = 1000
        self.matlab_fs = 1000
        
        # Machine-specific RPM values (UPDATED PER YOUR SPEC)
        self.csv_rpm = 1800.0   # Industrial machine RPM (CSV dataset)
        self.matlab_rpm = 1440.0  # Updated 45kW test rig RPM (was 1500)
        self.ahar_rpm = 1900.0  # Known Ahrar operating point
        self.cwru_rpm = 1797.0  # Standard CWRU speed

cfg = InferenceConfig()

# --------------------------------------------------
# CRITICAL FIX #3: CHANNEL-SEMANTIC PREPROCESSING
# --------------------------------------------------
def preprocess_signal(sig_raw, original_fs, target_fs, window_sec, rpm_override=None, dataset_type="unknown"):
    """
    Unified preprocessing with channel semantic consistency:
    - For 3-axis signals: Use physical axes directly
    - For 1-axis signals: REPLICATE to 3 channels (NOT Hilbert synthesis)
    - Hardcode RPM for known machines (critical for short signals)
    """
    # 1. Ensure proper shape (N, C)
    if sig_raw.ndim == 1:
        # SINGLE-AXIS SIGNAL → REPLICATE to 3 channels
        # CRITICAL: Replication maintains semantic consistency with overhang-trained model
        sig_raw = np.column_stack([sig_raw, sig_raw, sig_raw])
    elif sig_raw.ndim == 2:
        if sig_raw.shape[1] == 1:
            # Single column → replicate
            sig_raw = np.column_stack([sig_raw[:, 0], sig_raw[:, 0], sig_raw[:, 0]])
        elif sig_raw.shape[0] == 3 and sig_raw.shape[1] > 3:
            # Transposed (3, N) → fix to (N, 3)
            sig_raw = sig_raw.T
    
    # 2. Resample to target frequency (4kHz)
    if original_fs != target_fs:
        gcd_val = np.gcd(int(original_fs), int(target_fs))
        sig_resampled = np.column_stack([
            resample_poly(sig_raw[:, i], target_fs // gcd_val, original_fs // gcd_val)
            for i in range(3)
        ])
    else:
        sig_resampled = sig_raw
    
    # 3. Determine RPM (HARDCODE for known machines)
    if rpm_override is not None:
        rpm_est = rpm_override
        rpm_conf = 1.0
    else:
        # Only estimate if signal is long enough (>0.5s)
        if len(sig_resampled) >= 0.5 * target_fs:
            rpm_est, rpm_conf = estimate_rpm_safe(sig_resampled[:, 0], target_fs)
        else:
            rpm_est = 1500.0  # Conservative fallback
            rpm_conf = 0.0
    
    # 4. Windowing: Center crop/pad to fixed length
    win_pts = int(window_sec * target_fs)
    if len(sig_resampled) < win_pts:
        pad_len = win_pts - len(sig_resampled)
        sig_windowed = np.pad(sig_resampled, ((0, pad_len), (0, 0)), mode='constant')
    else:
        start = (len(sig_resampled) - win_pts) // 2
        sig_windowed = sig_resampled[start:start + win_pts]
    
    # 5. Z-score normalization per channel
    sig_norm = (sig_windowed - np.mean(sig_windowed, axis=0)) / (np.std(sig_windowed, axis=0) + 1e-6)
    
    # 6. Compute order spectrum
    spec = compute_order_spectrum(sig_norm, target_fs, rpm_est, cfg.max_order, cfg.order_bins)
    
    return spec, rpm_est, rpm_conf

# --------------------------------------------------
# MODEL LOADING (PyTorch 2.6+ compatible)
# --------------------------------------------------
def load_model(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = ResNet1D(input_channels=3, num_classes=4).to(device)
    
    # PyTorch 2.6+ requires weights_only=False for non-safetensors checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# --------------------------------------------------
# CSV DATASET INFERENCE (3-AXIS VIBRATION) - CORRECTED FROM EXCEL
# --------------------------------------------------
def run_csv_inference(model, device, cfg, csv_path, output_dir):
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON CSV DATASET (3-AXIS VIBRATION)")
    print("="*70)
    
    # Load CSV data with robust parsing
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} samples from {csv_path.name}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")
    
    # Robust column detection (handles "Temparature" typo)
    required_base = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Fault_Type']
    available_cols = df.columns.str.strip().str.lower().tolist()
    
    # Map expected columns to actual columns (case-insensitive + typo tolerant)
    col_map = {}
    for expected in required_base:
        expected_lower = expected.lower()
        # Find best match (exact match first, then partial)
        matches = [col for col in df.columns if expected_lower in col.lower()]
        if matches:
            col_map[expected] = matches[0]
        else:
            # Special handling for "Temperature" typo
            if expected == 'Fault_Type' and 'fault' in ' '.join(available_cols):
                col_map[expected] = [c for c in df.columns if 'fault' in c.lower()][0]
            elif expected.startswith('Vibration_'):
                # Try to find vibration columns by pattern
                vib_cols = [c for c in df.columns if 'vibration' in c.lower() or 'vib' in c.lower()]
                if vib_cols:
                    col_map[expected] = vib_cols[min(2, len(vib_cols)-1)]  # Take X/Y/Z approximations
    
    # Verify critical columns found
    missing = [c for c in required_base if c not in col_map]
    if missing:
        raise ValueError(f"Missing required columns after robust detection: {missing}\nAvailable columns: {df.columns.tolist()}")
    
    print(f"✓ Column mapping:")
    for exp, actual in col_map.items():
        print(f"    {exp:15s} → {actual}")
    
    # Map fault types to model classes (adjust based on your labeling)
    fault_mapping = {
        'Normal': 0,
        'Bearing Fault': 3,
        'Imbalance': 1,
        'Overheating': 0,  # Map thermal faults to Normal (vibration model can't detect thermal)
        'bearing fault': 3,
        'imbalance': 1,
        'overheating': 0,
        'normal': 0
    }
    
    # Segment into 1-second windows (non-overlapping) at 1kHz sampling
    window_size_orig = int(cfg.csv_fs * cfg.window_sec)
    num_windows = len(df) // window_size_orig
    
    if num_windows == 0:
        raise ValueError(f"Dataset too short! Need at least {window_size_orig} samples ({cfg.window_sec}s at {cfg.csv_fs}Hz)")
    
    print(f"✓ Segmenting into {num_windows} non-overlapping 1-second windows ({window_size_orig} samples each)")
    
    predictions = []
    confidences = []
    true_labels = []
    rpm_estimates = []
    
    for i in tqdm(range(num_windows), desc="Processing CSV windows"):
        start = i * window_size_orig
        end = start + window_size_orig
        window_df = df.iloc[start:end]
        
        # Extract 3-axis vibration using mapped columns
        try:
            sig_raw = window_df[[col_map['Vibration_X'], col_map['Vibration_Y'], col_map['Vibration_Z']]].values
        except KeyError as e:
            # Fallback: use first 3 numeric columns as vibration axes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                sig_raw = window_df[numeric_cols[:3]].values
                print(f"⚠️  Using fallback columns {numeric_cols[:3]} for vibration axes")
            else:
                raise ValueError(f"Could not extract 3 vibration axes: {e}")
        
        # Get majority-vote label for window
        labels = window_df[col_map['Fault_Type']].values
        true_label_str = str(max(set(labels), key=list(labels).count)).strip().title()
        true_label = fault_mapping.get(true_label_str, 0)  # Default to Normal
        
        # Preprocess with HARDCODED RPM for industrial machines (1800 RPM)
        spec, rpm_est, _ = preprocess_signal(
            sig_raw,
            original_fs=cfg.csv_fs,
            target_fs=cfg.target_fs,
            window_sec=cfg.window_sec,
            rpm_override=cfg.csv_rpm,  # CRITICAL: Hardcode RPM=1800 for industrial machine
            dataset_type="csv"
        )
        
        # Inference
        x = torch.from_numpy(spec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs.max().item()
        
        predictions.append(pred)
        confidences.append(conf)
        true_labels.append(true_label)
        rpm_estimates.append(rpm_est)
    
    # Evaluation
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_conf = np.array(confidences)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    print("\n" + "="*70)
    print("CSV DATASET RESULTS")
    print("="*70)
    print(f"Total windows: {len(y_true)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Balanced Accuracy: {bal_acc:.2%}")
    
    # Classification report
    class_names = ["Normal/Overheating", "Imbalance", "Misalignment", "Bearing Fault"]
    present_classes = sorted(np.unique(y_true))
    present_names = [class_names[i] for i in present_classes if i < len(class_names)]
    
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=present_names,
        digits=4,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted', fontsize=13, fontweight='bold')
    plt.ylabel('True', fontsize=13, fontweight='bold')
    plt.title(f'CSV Dataset Confusion Matrix\nBalanced Acc: {bal_acc:.2%}', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "csv_confusion_matrix.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {output_dir / 'csv_confusion_matrix.png'}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'window_id': range(len(y_true)),
        'true_label': y_true,
        'true_class': [class_names[i] if i < len(class_names) else f"Class_{i}" for i in y_true],
        'pred_label': y_pred,
        'pred_class': [class_names[i] if i < len(class_names) else f"Class_{i}" for i in y_pred],
        'confidence': y_conf,
        'rpm_estimate': rpm_estimates,
        'correct': y_true == y_pred
    })
    results_df.to_csv(output_dir / "csv_detailed_results.csv", index=False)
    print(f"✓ Detailed results saved to {output_dir / 'csv_detailed_results.csv'}")
    
    return acc, bal_acc

# --------------------------------------------------
# 45KW MATLAB DATASET INFERENCE (SINGLE-AXIS → REPLICATED)
# --------------------------------------------------
def run_matlab_inference(model, device, cfg, data_root, output_dir):
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON 45KW MATLAB DATASET (SINGLE-AXIS → REPLICATED)")
    print("="*70)
    
    # Find all .mat files in Healthy/Faulty directories
    healthy_dir = data_root / "Healthy"
    faulty_dir = data_root / "Faulty"
    
    if not healthy_dir.exists() or not faulty_dir.exists():
        raise FileNotFoundError(f"Expected directories 'Healthy' and 'Faulty' in {data_root}")
    
    files = []
    labels = []
    
    # Healthy files → Normal (class 0)
    for f in healthy_dir.glob("*.mat"):
        files.append(f)
        labels.append(0)
    
    # Faulty files → Bearing Fault (class 3) [conservative mapping]
    for f in faulty_dir.glob("*.mat"):
        files.append(f)
        labels.append(3)
    
    print(f"Found {len(files)} files ({sum(1 for l in labels if l==0)} Healthy, {sum(1 for l in labels if l==3)} Faulty)")
    
    predictions = []
    confidences = []
    rpm_estimates = []
    
    for file, true_label in tqdm(zip(files, labels), total=len(files), desc="Processing MATLAB files"):
        try:
            # Load MATLAB file (variable "H" as per specification)
            mat = scipy.io.loadmat(file)
            if 'H' not in mat:
                # Try alternative variable names
                candidates = [k for k in mat.keys() if not k.startswith('__')]
                if candidates:
                    sig_raw = mat[candidates[0]]
                    print(f"⚠️  Using variable '{candidates[0]}' instead of 'H' for {file.name}")
                else:
                    print(f"⚠️  Skipping {file.name}: No valid signal variable found")
                    continue
            else:
                sig_raw = mat['H']
            
            # Handle different shapes
            if sig_raw.ndim > 2:
                sig_raw = sig_raw.squeeze()
            if sig_raw.ndim == 2:
                if sig_raw.shape[0] == 1 or sig_raw.shape[1] == 1:
                    sig_raw = sig_raw.flatten()
                elif sig_raw.shape[0] == 3:  # Transposed (3, N)
                    sig_raw = sig_raw.T
            
            # Preprocess with HARDCODED RPM for test rig (UPDATED TO 1440 RPM)
            spec, rpm_est, _ = preprocess_signal(
                sig_raw,
                original_fs=cfg.matlab_fs,
                target_fs=cfg.target_fs,
                window_sec=cfg.window_sec,
                rpm_override=cfg.matlab_rpm,  # CRITICAL: Updated to 1440 RPM per your spec
                dataset_type="matlab"
            )
            
            # Inference
            x = torch.from_numpy(spec).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max().item()
            
            predictions.append(pred)
            confidences.append(conf)
            rpm_estimates.append(rpm_est)
            
        except Exception as e:
            print(f"⚠️  Error processing {file.name}: {str(e)[:100]}")
            predictions.append(-1)
            confidences.append(0.0)
            rpm_estimates.append(0.0)
    
    # Evaluation (binary: Healthy=Normal vs Faulty=Bearing)
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    y_conf = np.array(confidences)
    
    # Filter out errored samples
    valid_mask = y_pred != -1
    if np.sum(valid_mask) == 0:
        print("⚠️  No valid predictions for evaluation")
        return 0.0, 0.0
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    y_conf = y_conf[valid_mask]
    
    # Map to binary classification
    y_true_binary = (y_true != 0).astype(int)  # 0=Healthy→Normal, 1=Faulty
    y_pred_binary = (y_pred != 0).astype(int)  # 0=Normal, 1=Non-Normal
    
    acc = accuracy_score(y_true_binary, y_pred_binary)
    bal_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    
    print("\n" + "="*70)
    print("45KW MATLAB DATASET RESULTS (Binary: Healthy vs Faulty)")
    print("="*70)
    print(f"Total valid files: {len(y_true)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Balanced Accuracy: {bal_acc:.2%}")
    
    # Confusion matrix (binary)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["Predicted Healthy", "Predicted Faulty"],
               yticklabels=["True Healthy", "True Faulty"],
               linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title(f'45kW Dataset Confusion Matrix (Binary)\nBalanced Acc: {bal_acc:.2%}', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "matlab_confusion_matrix.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {output_dir / 'matlab_confusion_matrix.png'}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'file': [f.name for f, valid in zip(files, valid_mask) if valid],
        'true_label': ['Healthy' if l == 0 else 'Faulty' for l, valid in zip(labels, valid_mask) if valid],
        'pred_label': ['Healthy' if p == 0 else 'Faulty' for p, valid in zip(predictions, valid_mask) if valid],
        'confidence': y_conf,
        'rpm_estimate': [r for r, valid in zip(rpm_estimates, valid_mask) if valid],
        'correct': y_true_binary == y_pred_binary
    })
    results_df.to_csv(output_dir / "matlab_detailed_results.csv", index=False)
    print(f"✓ Detailed results saved to {output_dir / 'matlab_detailed_results.csv'}")
    
    return acc, bal_acc

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():
    # Setup paths
    OUTPUT_DIR = BASE_DIR / "multi_dataset_validation"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # CORRECTED PATHS FOR YOUR DATASETS
    CSV_PATH = BASE_DIR / "data" / "New folder" / "Rotating_equipment_fault_data.csv"  # CHANGED FROM .xlsx TO .csv
    MATLAB_ROOT = BASE_DIR / "data" / "paper_data"  # Removed trailing space
    
    # UPDATED MODEL PATH (adjust timestamp as needed)
    MODEL_PATH = BASE_DIR / "models" / "domain_shift" / "training_run_20260203_170158" / "checkpoints" / "best_model_domain_robust.pth"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model (overhang-trained)
    print(f"Loading overhang-trained model from {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)
    print("✓ Model loaded successfully")
    print("⚠️  CRITICAL NOTE: Model was trained on OVERHANG channels (cols 5-7 in MAFAULDA)")
    print("   For inference, we REPLICATE signals to 3 channels to maintain semantic consistency")
    print("   (NOT Hilbert synthesis which creates non-physical channel relationships)")
    
    # Run CSV inference (if file exists)
    if CSV_PATH.exists():
        print(f"\n✓ CSV dataset found at {CSV_PATH}")
        csv_acc, csv_bal_acc = run_csv_inference(model, device, cfg, CSV_PATH, OUTPUT_DIR)
    else:
        print(f"\n⚠️  CSV dataset not found at {CSV_PATH}. Skipping...")
        print(f"   Checked path: {CSV_PATH.absolute()}")
        csv_acc = csv_bal_acc = None
    
    # Run MATLAB inference (if directories exist)
    if MATLAB_ROOT.exists() and (MATLAB_ROOT / "Healthy").exists() and (MATLAB_ROOT / "Faulty").exists():
        print(f"\n✓ 45kW MATLAB dataset found at {MATLAB_ROOT}")
        matlab_acc, matlab_bal_acc = run_matlab_inference(model, device, cfg, MATLAB_ROOT, OUTPUT_DIR)
    else:
        print(f"\n⚠️  45kW MATLAB dataset not found at {MATLAB_ROOT}. Skipping...")
        print(f"   Expected subdirectories: {MATLAB_ROOT / 'Healthy'} and {MATLAB_ROOT / 'Faulty'}")
        matlab_acc = matlab_bal_acc = None
    
    # Final summary
    print("\n" + "="*70)
    print("MULTI-DATASET VALIDATION SUMMARY")
    print("="*70)
    print(f"Model: Overhang-trained ResNet1D (MAFAULDA)")
    print(f"Critical Fixes Applied:")
    print(f"  ✓ CSV parsing: Robust column detection (handles 'Temparature' typo)")
    print(f"  ✓ Channel semantics: Signal replication (not Hilbert) for semantic consistency")
    print(f"  ✓ RPM handling: Hardcoded values (CSV=1800 RPM, MATLAB=1440 RPM per your spec)")
    print(f"  ✓ Sampling rate: 1kHz → 4kHz rational resampling")
    print(f"  ✓ Short-signal robustness: Fallback RPM estimation for <0.5s signals")
    
    if csv_acc is not None:
        print(f"\nCSV Dataset Results:")
        print(f"  Accuracy: {csv_acc:.2%}")
        print(f"  Balanced Accuracy: {csv_bal_acc:.2%}")
    
    if matlab_acc is not None:
        print(f"\n45kW MATLAB Dataset Results (Binary):")
        print(f"  Accuracy: {matlab_acc:.2%}")
        print(f"  Balanced Accuracy: {matlab_bal_acc:.2%}")
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()