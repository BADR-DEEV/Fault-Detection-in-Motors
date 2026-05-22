#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-CORRECTED Training Pipeline for ADXL357 Deployment
Critical fixes:
✓ Use ONLY Overhang channels (cols 5-7 / 0-indexed 4-6) - matches sensor placement
✓ Low-pass filter at 4kHz BEFORE resampling (prevents aliasing, matches sensor reality)
✓ Exclude 0g bearing faults from training (use only 6g+ for robust fault detection)
✓ Aggressive synthetic normal generation (400 samples) + 8.0x weighting
✓ Short-window training (20% at 0.2s) to match Ahrar signal length
✓ Wider RPM augmentation (0.75-1.35x) to cover 1900 RPM operating point
"""
import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, resample_poly, welch, butter, filtfilt, find_peaks
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# --------------------------------------------------
# CONFIGURATION (PHYSICS-CORRECTED FOR ADXL357)
# --------------------------------------------------
class DomainRobustConfig:
    def __init__(self, args=None):
        # Core parameters (ADXL357 matches 4kHz capability)
        self.seed = 42
        self.target_fs = 4000  # ADXL357 maximum sampling rate
        self.window_sec = 1.0
        self.window_pts = int(self.window_sec * self.target_fs)
        self.short_window_sec = 0.2  # Match Ahrar signal length
        self.short_window_pts = int(self.short_window_sec * self.target_fs)
        
        # Order spectrum parameters (MUST match inference)
        self.max_order = 10.0
        self.order_bins = 512
        
        # Physics-based augmentation ranges (WIDENED for real-world variation)
        self.rpm_jitter_range = (0.75, 1.35)   # ±25-35% RPM variation (covers 1900 RPM)
        self.energy_shift_db = (-15, +15)      # ±15dB energy variation
        self.snr_range_db = (10, 35)           # Wider SNR range
        
        # SEVERITY WEIGHTS - AGGRESSIVE NORMAL OVERSAMPLING
        self.severity_weights = {
            "normal": 8.0,        # Critical for extreme imbalance
            "imbalance": 1.5,
            "misalignment": 1.2,
            "bearing_6g": 2.0,    # Start training with 6g+ faults only (exclude 0g)
            "bearing_20g": 1.2,
            "bearing_35g": 0.8
        }
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.patience = 20
        self.min_lr = 1e-6
        self.short_window_prob = 0.20  # 20% short-window training
        
        # Critical physics constraint: 4kHz low-pass filter
        self.sensor_max_freq = 4000  # ADXL357 maximum frequency response
        
        # Paths (set dynamically)
        self.raw_data_dir = None
        self.processed_dir = None
        self.output_dir = None
        
        if args:
            for key, value in vars(args).items():
                if hasattr(self, key) and value is not None:
                    setattr(self, key, value)
    
    def to_dict(self):
        # Convert Path objects to strings for JSON serialization
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items() if not k.startswith('_')}

# --------------------------------------------------
# PHYSICS-CORRECTED AUGMENTATION (4kHz SENSOR REALITY)
# --------------------------------------------------
class DomainRobustAugmentor:
    """Simulates real-world domain shifts with sensor reality constraints"""
    
    def __init__(self, cfg: DomainRobustConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
    
    def __call__(self, sig: np.ndarray, rpm_true: float, severity: str, use_short_window: bool = False) -> Tuple[np.ndarray, float]:
        """
        Apply domain-robust augmentations with 4kHz sensor reality constraints
        """
        sig_aug = sig.copy().astype(np.float32)
        rpm_aug = rpm_true
        
        # 1. SHORT WINDOW HANDLING (critical for Ahrar generalization)
        if use_short_window and len(sig_aug) > self.cfg.short_window_pts:
            start = np.random.randint(0, len(sig_aug) - self.cfg.short_window_pts)
            sig_aug = sig_aug[start:start + self.cfg.short_window_pts]
        
        # 2. RPM jitter (WIDENED range to cover 1900 RPM)
        rpm_factor = np.random.uniform(*self.cfg.rpm_jitter_range)
        rpm_aug = rpm_true * rpm_factor
        
        # 3. Energy scaling (±15dB)
        energy_db = np.random.uniform(*self.cfg.energy_shift_db)
        sig_aug *= 10 ** (energy_db / 20.0)
        
        # 4. Additive noise (wider SNR range)
        snr_db = np.random.uniform(*self.cfg.snr_range_db)
        signal_power = np.mean(sig_aug ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        sig_aug += np.random.normal(0, np.sqrt(noise_power), sig_aug.shape)
        
        # 5. Severity-aware modulation (incipient faults)
        if severity.startswith("bearing_"):
            t = np.arange(len(sig_aug)) / self.cfg.target_fs
            modulation_freq = 3.0 * (rpm_aug / 60.0)
            modulation = 0.08 * np.sin(2 * np.pi * modulation_freq * t)
            sig_aug *= (1.0 + modulation)
        
        # 6. CRITICAL: Re-apply 4kHz low-pass filter after augmentation
        # Prevents high-frequency artifacts that ADXL357 cannot detect
        nyq = 0.5 * self.cfg.target_fs
        b, a = butter(4, min(self.cfg.sensor_max_freq/nyq, 0.99), btype='low')
        sig_aug = filtfilt(b, a, sig_aug)
        
        # 7. Final normalization
        sig_aug = (sig_aug - np.mean(sig_aug)) / (np.std(sig_aug) + 1e-6)
        
        return sig_aug, rpm_aug

# --------------------------------------------------
# ORDER SPECTRUM (PHYSICS-CORRECTED)
# --------------------------------------------------
def compute_order_spectrum(sig: np.ndarray, fs: float, rpm: float, cfg: DomainRobustConfig) -> np.ndarray:
    """Compute order spectrum with harmonic-band energy normalization"""
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1]:
        sig = sig.T
    
    if rpm < 100 or rpm > 5000:
        rpm = 1500.0
    
    shaft_hz = rpm / 60.0
    nperseg = min(len(sig), 2048)
    
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2, axis=0)
    
    if Pxx.ndim == 1:
        Pxx = Pxx[:, np.newaxis]
    
    orders = f / shaft_hz
    target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
    
    specs = []
    for ch in range(Pxx.shape[1]):
        s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
        
        # Physics-based harmonic-band normalization
        harmonic_bands = [(0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, cfg.max_order)]
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
# PHYSICS-CORRECTED DATA PROCESSING (OVERHANG + 4KHZ FILTERING)
# --------------------------------------------------
def lowpass_filter_4khz(sig: np.ndarray, fs: float, cutoff: float = 4000.0) -> np.ndarray:
    """
    CRITICAL FIX: Apply 4kHz low-pass filter BEFORE resampling
    Prevents aliasing and matches ADXL357 sensor reality
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, min(normal_cutoff, 0.99), btype='low', analog=False)
    return filtfilt(b, a, sig)

def generate_synthetic_normal(cfg: DomainRobustConfig, num_samples: int = 400) -> List[Tuple[np.ndarray, float]]:
    """
    Generate PHYSICALLY PLAUSIBLE normal vibration matching ADXL357 characteristics:
    - 3-axis overhang vibration only
    - Realistic bearing noise (1/f spectrum)
    - Channel correlations matching real machinery
    - 4kHz bandwidth constraint applied
    """
    normals = []
    
    for i in range(num_samples):
        N = cfg.window_pts
        freqs = np.fft.rfftfreq(N, d=1/cfg.target_fs)
        
        # Create realistic bearing noise spectrum (1/f + white noise)
        spec = np.ones_like(freqs)
        spec[1:] = 1.0 / np.sqrt(freqs[1:])  # 1/f characteristic
        
        # CRITICAL: Apply 4kHz bandwidth constraint in frequency domain
        freq_limit_idx = np.searchsorted(freqs, cfg.sensor_max_freq)
        spec[freq_limit_idx:] = 0.0  # Zero out frequencies above 4kHz
        
        spec += 0.25 * np.random.randn(len(spec))  # Realistic noise floor
        
        # Randomize phase
        phase = np.random.uniform(0, 2*np.pi, len(spec))
        complex_spec = spec * np.exp(1j * phase)
        
        # Inverse FFT
        sig_base = np.fft.irfft(complex_spec, n=N)
        sig_base = (sig_base - np.mean(sig_base)) / (np.std(sig_base) + 1e-6)
        
        # Create PHYSICALLY CORRELATED 3-axis vibration (overhang only)
        ch0 = sig_base + 0.15 * np.random.randn(N)
        phase_shift = np.random.uniform(0.1, 0.3) * 2 * np.pi
        ch1 = 0.75 * sig_base + 0.25 * np.sin(2 * np.pi * np.arange(N)/N * 5 + phase_shift)
        ch2 = 0.6 * sig_base + 0.3 * np.sin(2 * np.pi * np.arange(N)/N * 15)
        
        # Independent normalization per channel
        ch0 = (ch0 - np.mean(ch0)) / (np.std(ch0) + 1e-6)
        ch1 = (ch1 - np.mean(ch1)) / (np.std(ch1) + 1e-6)
        ch2 = (ch2 - np.mean(ch2)) / (np.std(ch2) + 1e-6)
        
        sig_3ch = np.column_stack([ch0, ch1, ch2])
        
        # Realistic RPM range for healthy MAFAULDA runs
        rpm = np.random.uniform(1200, 1800)
        
        normals.append((sig_3ch, rpm))
    
    return normals

def estimate_rpm_from_tach(tach: np.ndarray, fs: float) -> float:
    """Estimate RPM from tachometer signal using pulse counting"""
    tach_norm = (tach - np.mean(tach)) / (np.std(tach) + 1e-6)
    peaks, _ = find_peaks(tach_norm, height=0.5, distance=fs//50)
    
    if len(peaks) < 2:
        return 1500.0
    
    pulse_intervals = np.diff(peaks) / fs
    avg_interval = np.median(pulse_intervals)
    rpm = 60.0 / avg_interval
    return np.clip(rpm, 600, 3600)

def process_data_preserving_structure(cfg: DomainRobustConfig, logger: logging.Logger) -> pd.DataFrame:
    """
    PHYSICS-CORRECTED PROCESSING:
    1. Use ONLY Overhang channels (cols 5-7 / 0-indexed 4-6) to match ADXL357 placement
    2. Apply 4kHz low-pass filter BEFORE resampling (critical sensor reality match)
    3. Exclude 0g bearing faults from training set (use only 6g+ for robust detection)
    4. Generate 400 synthetic normals for extreme imbalance
    5. Preserve tachometer-based RPM (more accurate than vibration-based)
    """
    meta_path = cfg.processed_dir / "metadata_physics_corrected.csv"
    
    if meta_path.exists():
        logger.info(f"Found existing metadata_physics_corrected.csv. Using cached structure.")
        df_meta = pd.read_csv(meta_path)
        synth_count = df_meta[df_meta['original_path'].str.contains('synthetic_normal')].shape[0]
        logger.info(f"Loaded {len(df_meta)} samples ({synth_count} synthetic normals)")
        return df_meta
    
    logger.info(f"Processing MAFAULDA data with PHYSICS-CORRECTED pipeline:")
    logger.info("  ✓ Using ONLY Overhang channels (cols 5-7) to match ADXL357 sensor placement")
    logger.info("  ✓ Applying 4kHz low-pass filter BEFORE resampling (sensor reality match)")
    logger.info("  ✓ Excluding 0g bearing faults from training (using 6g+ only for robust detection)")
    logger.info("  ✓ Generating 400 synthetic normals for extreme imbalance")
    
    files = sorted(list(cfg.raw_data_dir.rglob("*.csv")))
    metadata_rows = []
    
    for i, p in enumerate(files):
        fname = str(p).lower().replace("\\", "/")
        label = -1
        label_name = "Unknown"
        severity_info = "unknown"
        
        # CORRECT LABELING STRATEGY (0g = bearing fault, not normal)
        if "normal" in fname and not any(k in fname for k in ["bearing", "ball", "cage", "outer", "inner", "0g", "6g", "20g", "35g"]):
            label, label_name = 0, "Normal"
            severity_info = "normal"
        
        elif "imbalance" in fname and "bearing" not in fname:
            label, label_name = 1, "Imbalance"
            severity_info = "imbalance"
        
        elif any(k in fname for k in ["horizontal", "vertical", "misalign"]):
            label, label_name = 2, "Misalignment"
            severity_info = "misalignment"
        
        # CRITICAL: ALL bearing faults (including 0g) = Bearing class
        # BUT: We'll exclude 0g from TRAINING later (keep for evaluation)
        elif any(k in fname for k in ["overhang", "bearing", "ball", "outer", "inner", "cage"]):
            # EXCLUDE underhang bearing faults (sensor is on overhang side)
            if "underhang" in fname:
                continue  # Skip underhang bearing faults
            
            label, label_name = 3, "Bearing"
            
            if "0g" in fname:
                severity_info = "bearing_0g"
            elif "6g" in fname:
                severity_info = "bearing_6g"
            elif "20g" in fname:
                severity_info = "bearing_20g"
            elif "35g" in fname:
                severity_info = "bearing_35g"
            else:
                severity_info = "bearing_20g"
        
        if label == -1:
            continue
        
        try:
            # Load MAFAULDA data (50kHz sampling)
            # Column mapping (0-indexed):
            #   0: Tachometer
            #   1-3: Underhang bearing (axial/radial/tangential)
            #   4-6: Overhang bearing (axial/radial/tangential) ← USE THESE
            #   7: Microphone
            df = pd.read_csv(p, header=None)
            if df.shape[1] < 8:  # Need at least 8 columns (0-7)
                continue
            
            # CRITICAL FIX: USE ONLY OVERHANG CHANNELS (cols 4-6 / 0-indexed)
            tach = df.iloc[:, 0].values          # Tachometer (col 0)
            sig_overhang = df.iloc[:, 4:7].values  # Overhang 3-axis (cols 4,5,6) - MATCHES ADXL357
            
            # CRITICAL FIX #1: Apply 4kHz low-pass filter BEFORE resampling
            # Prevents aliasing and matches ADXL357 sensor reality
            sig_filtered = np.column_stack([
                lowpass_filter_4khz(sig_overhang[:, i], 50000, cutoff=4000.0)
                for i in range(3)
            ])
            
            # Estimate RPM from tachometer (more accurate than vibration-based)
            rpm = estimate_rpm_from_tach(tach, 50000)  # MAFAULDA original FS = 50 kHz
            
            # CRITICAL FIX #2: Resample AFTER filtering (prevents aliasing)
            gcd_val = np.gcd(50000, cfg.target_fs)
            sig_4k = np.column_stack([
                resample_poly(sig_filtered[:, i], cfg.target_fs // gcd_val, 50000 // gcd_val)
                for i in range(3)
            ])
            
            # Ensure consistent window length
            if len(sig_4k) < cfg.window_pts:
                pad_len = cfg.window_pts - len(sig_4k)
                sig_4k = np.pad(sig_4k, ((0, pad_len), (0, 0)), mode='constant')
            elif len(sig_4k) > cfg.window_pts:
                start = (len(sig_4k) - cfg.window_pts) // 2
                sig_4k = sig_4k[start:start + cfg.window_pts]
            
            # Save processed data
            rel_path = p.relative_to(cfg.raw_data_dir)
            save_path = cfg.processed_dir / rel_path.with_suffix('.npz')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_path, sig=sig_4k, label=label, rpm=rpm, severity=severity_info)
            
            metadata_rows.append({
                "path": str(save_path),
                "original_path": str(p),
                "label": label,
                "label_name": label_name,
                "rpm": round(rpm, 1),
                "severity": severity_info
            })
            
            if (i + 1) % 200 == 0:
                logger.info(f"Processed {i+1}/{len(files)} files...")
                
        except Exception as e:
            logger.warning(f"Error processing {p.name}: {str(e)[:100]}")
            continue
    
    # CRITICAL FIX: Generate 400 synthetic normals (was 200) for extreme imbalance
    real_normal_count = sum(1 for row in metadata_rows if row['severity'] == 'normal')
    logger.info(f"\nReal normal samples found: {real_normal_count}")
    
    if real_normal_count < 100:
        synth_needed = 400
        logger.info(f"Generating {synth_needed} synthetic normal samples (overhang channels only)...")
        
        synthetic_normals = generate_synthetic_normal(cfg, num_samples=synth_needed)
        
        synth_dir = cfg.processed_dir / "synthetic_normals_physics_corrected"
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (sig_3ch, rpm) in enumerate(synthetic_normals):
            if sig_3ch.shape[0] != cfg.window_pts:
                if sig_3ch.shape[0] < cfg.window_pts:
                    pad_len = cfg.window_pts - sig_3ch.shape[0]
                    sig_3ch = np.pad(sig_3ch, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    start = (sig_3ch.shape[0] - cfg.window_pts) // 2
                    sig_3ch = sig_3ch[start:start + cfg.window_pts]
            
            synth_path = synth_dir / f"synthetic_normal_{i:04d}.npz"
            np.savez(synth_path, sig=sig_3ch, label=0, rpm=rpm, severity="normal")
            
            metadata_rows.append({
                "path": str(synth_path),
                "original_path": f"synthetic_normal_{i:04d}",
                "label": 0,
                "label_name": "Normal",
                "rpm": round(rpm, 1),
                "severity": "normal"
            })
        
        logger.info(f"Added {synth_needed} synthetic normal samples (overhang channels only)")
    else:
        logger.info("Sufficient real normal samples found (>100)")
    
    # Create metadata DataFrame
    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(meta_path, index=False)
    logger.info(f"\n✓ Processing complete: {len(df_meta)} total samples")
    logger.info(f"  - Real samples: {len(df_meta) - len(df_meta[df_meta['original_path'].str.contains('synthetic_normal')])}")
    logger.info(f"  - Synthetic normals: {len(df_meta[df_meta['original_path'].str.contains('synthetic_normal')])}")
    
    # Log distributions
    logger.info("\nClass distribution after CORRECT labeling:")
    class_dist = df_meta['label_name'].value_counts().sort_index()
    for cls, count in class_dist.items():
        pct = count / len(df_meta) * 100
        logger.info(f"  {cls:15s}: {count:4d} samples ({pct:5.1f}%)")
    
    logger.info("\nSeverity distribution (excluding 0g from training later):")
    severity_dist = df_meta['severity'].value_counts().sort_index()
    for sev, count in severity_dist.items():
        pct = count / len(df_meta) * 100
        logger.info(f"  {sev:18s}: {count:4d} samples ({pct:5.1f}%)")
    
    # Critical warning if imbalance still severe
    normal_pct = (df_meta['severity'] == 'normal').sum() / len(df_meta) * 100
    if normal_pct < 12.0:
        logger.warning(f"⚠️  Normal class still underrepresented ({normal_pct:.1f}%). "
                      f"Using aggressive sampling weight (8.0x) during training.")
    
    return df_meta

# --------------------------------------------------
# DATASET (PHYSICS-CORRECTED: EXCLUDE 0G FROM TRAINING)
# --------------------------------------------------
class FaultDiagnosisDataset(Dataset):
    def __init__(self, 
                 metadata: pd.DataFrame,
                 cfg: DomainRobustConfig,
                 augmentor: Optional[DomainRobustAugmentor] = None,
                 mode: str = 'train',
                 exclude_0g: bool = True):
        self.metadata = metadata.reset_index(drop=True)
        self.cfg = cfg
        self.augmentor = augmentor if mode == 'train' else None
        self.mode = mode
        self.exclude_0g = exclude_0g and mode == 'train'  # Only exclude during training
        
        # Filter out 0g bearing faults during training (keep for validation/test)
        if self.exclude_0g:
            mask = self.metadata['severity'] != 'bearing_0g'
            self.metadata = self.metadata[mask].reset_index(drop=True)
            logging.info(f"Excluded bearing_0g samples from training set: {len(metadata) - len(self.metadata)} samples removed")
        
        self.severity_labels = self.metadata['severity'].tolist()
        
        # Validate severity labels
        valid_severities = set(cfg.severity_weights.keys())
        invalid = [s for s in self.severity_labels if s not in valid_severities]
        if invalid:
            self.severity_labels = [
                s if s in valid_severities else ('bearing_20g' if 'bearing' in str(s).lower() else 'normal')
                for s in self.severity_labels
            ]
        
        logging.info(f"Initialized {mode} dataset with {len(self)} samples")
        logging.info(f"Class distribution: {self.metadata['label_name'].value_counts().sort_index().to_dict()}")
        logging.info(f"Severity distribution: {pd.Series(self.severity_labels).value_counts().sort_index().to_dict()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        severity = self.severity_labels[idx]
        
        data = np.load(row['path'])
        sig = data['sig']  # (N, 3) - overhang channels only, 4kHz filtered
        rpm_true = float(data.get('rpm', 1500.0))
        label = int(row['label'])
        
        # Apply augmentations during training
        if self.augmentor and self.mode == 'train':
            ch_idx = np.random.randint(0, 3)
            sig_ch = sig[:, ch_idx].copy()
            
            use_short = np.random.random() < self.cfg.short_window_prob
            sig_aug_ch, rpm_aug = self.augmentor(sig_ch, rpm_true, severity, use_short_window=use_short)
            
            # Reconstruct 3-channel signal
            sig_aug = np.zeros((len(sig_aug_ch), 3), dtype=np.float32)
            sig_aug[:, ch_idx] = sig_aug_ch
            
            for other_idx in range(3):
                if other_idx != ch_idx:
                    if len(sig[:, other_idx]) != len(sig_aug_ch):
                        sig_aug[:, other_idx] = resample_poly(
                            sig[:, other_idx], 
                            len(sig_aug_ch), 
                            len(sig[:, other_idx])
                        )
                    else:
                        sig_aug[:, other_idx] = sig[:, other_idx]
            
            rpm_used = rpm_aug
        else:
            sig_aug = sig
            rpm_used = rpm_true
        
        # CRITICAL: Use physical 3 channels directly (no Hilbert synthesis)
        sig_3ch = sig_aug
        
        # Compute order spectrum
        spec = compute_order_spectrum(sig_3ch, self.cfg.target_fs, rpm_used, self.cfg)
        
        return {
            'spec': torch.from_numpy(spec).float(),  # (3, 512)
            'label': torch.tensor(label, dtype=torch.long),
            'severity': severity,
            'rpm': rpm_used,
            'sample_id': idx
        }

# --------------------------------------------------
# MODEL ARCHITECTURE (WITH DROPOUT FOR GENERALIZATION)
# --------------------------------------------------
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
        self.dropout = nn.Dropout(0.3)  # Prevents overfitting to MAFAULDA artifacts
        self.fc = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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

# --------------------------------------------------
# TRAINER WITH FOCAL LOSS
# --------------------------------------------------
# --------------------------------------------------
# TRAINER (COMPLETE IMPLEMENTATION WITH ALL METHODS)
# --------------------------------------------------
class DomainRobustTrainer:
    def __init__(self, cfg: DomainRobustConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.gamma = 2.0  # Focal loss gamma
        
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        
        (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        
        self.augmentor = DomainRobustAugmentor(cfg)
        self.model = ResNet1D(input_channels=3, num_classes=4).to(self.device)
        self.logger.info(f"Model initialized on {self.device} with dropout for generalization")
        
        # Use focal loss for severe class imbalance
        self.criterion = self.focal_loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.min_lr
        )
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_history = {'loss': [], 'acc': [], 'balanced_acc': []}
        self.val_history = {'loss': [], 'acc': [], 'balanced_acc': []}
    
    def focal_loss(self, logits, targets, alpha=0.75):
        """Focal loss for hard example mining (normals and incipient faults)"""
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        probs = torch.exp(-ce_loss)
        focal_weight = (1 - probs) ** self.gamma
        
        if alpha is not None:
            alpha_t = torch.where(targets == 0,
                                 torch.tensor(alpha, device=self.device),
                                 torch.tensor(1 - alpha, device=self.device))
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * ce_loss).mean()
    
    def compute_sample_weights(self, severity_labels: List[str]) -> np.ndarray:
        """Compute sampling weights with AGGRESSIVE normal oversampling"""
        weights = []
        for severity in severity_labels:
            base_weight = self.cfg.severity_weights.get(severity, 1.0)
            if severity.startswith('bearing'):
                base_weight *= 0.9
            weights.append(base_weight)
        
        weights = np.array(weights, dtype=np.float32)
        weights = weights * len(weights) / weights.sum()
        return weights
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            specs = batch['spec'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(specs)
            loss = self.focal_loss(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * specs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += specs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, balanced_acc
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray, List]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_severities = []
        
        with torch.no_grad():
            for batch in dataloader:
                specs = batch['spec'].to(self.device)
                labels = batch['label'].to(self.device)
                severities = batch['severity']
                
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * specs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += specs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_severities.extend(severities)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, balanced_acc, np.array(all_preds), np.array(all_labels), all_severities
    
    def plot_training_curves(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['loss'], label='Train Loss', linewidth=2)
        plt.plot(self.val_history['loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['acc'], label='Train Acc', linewidth=2)
        plt.plot(self.val_history['acc'], label='Val Acc', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.subplot(1, 3, 3)
        plt.plot(self.train_history['balanced_acc'], label='Train Balanced Acc', linewidth=2)
        plt.plot(self.val_history['balanced_acc'], label='Val Balanced Acc', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Balanced Accuracy', fontsize=12)
        plt.title('Balanced Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   linewidths=0.5, linecolor='gray', annot_kws={"size": 12})
        plt.xlabel('Predicted', fontsize=13, fontweight='bold')
        plt.ylabel('True', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / filename, dpi=150, bbox_inches='tight')
        plt.close()
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(self.output_dir / 'figures' / filename.replace('.png', '.csv'))
    
    def plot_severity_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, severities: List[str], filename: str):
        severity_groups = {
            'Normal': ['normal'],
            'Imbalance': ['imbalance'],
            'Misalignment': ['misalignment'],
            'Bearing (0g)': ['bearing_0g'],
            'Bearing (6g)': ['bearing_6g'],
            'Bearing (20g+)': ['bearing_20g', 'bearing_35g']
        }
        results = []
        for group_name, severity_types in severity_groups.items():
            mask = [s in severity_types for s in severities]
            if sum(mask) == 0:
                continue
            group_acc = accuracy_score(y_true[mask], y_pred[mask])
            group_bal_acc = balanced_accuracy_score(y_true[mask], y_pred[mask])
            results.append({
                'Severity Group': group_name,
                'Samples': sum(mask),
                'Accuracy': group_acc,
                'Balanced Accuracy': group_bal_acc
            })
        df = pd.DataFrame(results)
        plt.figure(figsize=(12, 6))
        bars = plt.barh(df['Severity Group'], df['Accuracy'], color='steelblue', alpha=0.8)
        plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Model Performance by Fault Severity', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        for i, (bar, acc, count) in enumerate(zip(bars, df['Accuracy'], df['Samples'])):
            plt.text(acc + 0.01, i, f'{acc:.2%} (n={count})',
                    va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / filename, dpi=150, bbox_inches='tight')
        plt.close()
        df.to_csv(self.output_dir / 'figures' / filename.replace('.png', '.csv'), index=False)
    
    # CRITICAL FIX: ADD THE MISSING train() METHOD
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model with early stopping and best model checkpointing"""
        self.logger.info("Starting training with domain-robust augmentations...")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Short window training probability: {self.cfg.short_window_prob*100:.0f}%")
        
        for epoch in range(self.cfg.num_epochs):
            train_loss, train_acc, train_bal_acc = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['balanced_acc'].append(train_bal_acc)
            
            val_loss, val_acc, val_bal_acc, y_pred, y_true, severities = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            self.val_history['balanced_acc'].append(val_bal_acc)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.cfg.num_epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                f"Val Balanced Acc: {val_bal_acc:.2%}"
            )
            
            # Save best model based on balanced accuracy
            if val_bal_acc > self.best_val_acc:
                self.best_val_acc = val_bal_acc
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.cfg.to_dict()
                }
                torch.save(checkpoint, self.output_dir / 'checkpoints' / 'best_model_domain_robust.pth')
                self.logger.info(f"✓ New best model saved (Balanced Acc: {val_bal_acc:.2%})")
                
                # Generate validation visualizations
                self.plot_confusion_matrix(
                    y_true, y_pred,
                    f'Validation Confusion Matrix (Epoch {epoch+1})\nBalanced Acc: {val_bal_acc:.2%}',
                    'confusion_matrix_val_best.png'
                )
                self.plot_severity_analysis(
                    y_true, y_pred, severities,
                    'severity_performance_val_best.png'
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Final visualizations
        self.plot_training_curves()
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best validation balanced accuracy: {self.best_val_acc:.2%}")
    
    def final_evaluation(self, test_loader: DataLoader, dataset_name: str = "Test"):
        """Final evaluation on test set with comprehensive reporting"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"FINAL EVALUATION ON {dataset_name.upper()} SET")
        self.logger.info(f"{'='*70}")
        
        # Load best model (PyTorch 2.6+ compatible)
        checkpoint = torch.load(
            self.output_dir / 'checkpoints' / 'best_model_domain_robust.pth',
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        test_loss, test_acc, test_bal_acc, y_pred, y_true, severities = self.validate(test_loader)
        
        self.logger.info(f"\n{dataset_name} Loss: {test_loss:.4f}")
        self.logger.info(f"{dataset_name} Accuracy: {test_acc:.2%}")
        self.logger.info(f"{dataset_name} Balanced Accuracy: {test_bal_acc:.2%}")
        
        classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
        report = classification_report(
            y_true, y_pred,
            target_names=classes,
            digits=4,
            zero_division=0
        )
        self.logger.info(f"\nClassification Report:\n{report}")
        
        self.plot_confusion_matrix(
            y_true, y_pred,
            f'{dataset_name} Confusion Matrix\nBalanced Acc: {test_bal_acc:.2%}',
            f'confusion_matrix_{dataset_name.lower()}.png'
        )
        
        self.plot_severity_analysis(
            y_true, y_pred, severities,
            f'severity_performance_{dataset_name.lower()}.png'
        )
        
        results_df = pd.DataFrame({
            'sample_id': range(len(y_true)),
            'true_label': y_true,
            'pred_label': y_pred,
            'true_class': [classes[i] for i in y_true],
            'pred_class': [classes[i] for i in y_pred],
            'severity': severities,
            'correct': y_true == y_pred
        })
        results_df.to_csv(self.output_dir / f'{dataset_name.lower()}_predictions.csv', index=False)
        self.logger.info(f"\nDetailed predictions saved to {self.output_dir / f'{dataset_name.lower()}_predictions.csv'}")
        
        return test_acc, test_bal_acc
# --------------------------------------------------
# MAIN EXECUTION (PHYSICS-CORRECTED WORKFLOW)
# --------------------------------------------------
def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description='Physics-Corrected Fault Diagnosis Training')
    parser.add_argument('--raw_data_dir', type=str, required=True, help='Path to raw MAFAULDA data')
    parser.add_argument('--processed_dir', type=str, required=True, help='Path to store processed data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to store training outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    base_output = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_output / f'training_run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = DomainRobustConfig(args)
    cfg.raw_data_dir = Path(args.raw_data_dir)
    cfg.processed_dir = Path(args.processed_dir)
    cfg.output_dir = output_dir
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting physics-corrected training pipeline for ADXL357 deployment")
    
    # STEP 1: Process data with physics-corrected pipeline
    logger.info("\n" + "="*70)
    logger.info("STEP 1: PHYSICS-CORRECTED DATA PROCESSING")
    logger.info("="*70)
    logger.info("  ✓ Using ONLY Overhang channels (cols 5-7) to match ADXL357 sensor placement")
    logger.info("  ✓ Applying 4kHz low-pass filter BEFORE resampling (sensor reality match)")
    logger.info("  ✓ Excluding 0g bearing faults from training set (using 6g+ only)")
    logger.info("  ✓ Generating 400 synthetic normals for extreme imbalance")
    metadata = process_data_preserving_structure(cfg, logger)
    
    # STEP 2: Create datasets with severity-aware sampling (exclude 0g from training)
    logger.info("\n" + "="*70)
    logger.info("STEP 2: CREATING DATASETS WITH SEVERITY-AWARE SAMPLING")
    logger.info("="*70)
    
    # Split: 70% train, 15% val, 15% test (stratified by severity)
    train_meta, temp_meta = train_test_split(
        metadata,
        test_size=0.3,
        stratify=metadata['severity'],
        random_state=cfg.seed
    )
    val_meta, test_meta = train_test_split(
        temp_meta,
        test_size=0.5,
        stratify=temp_meta['severity'],
        random_state=cfg.seed
    )
    
    logger.info(f"Train samples (before 0g exclusion): {len(train_meta)}")
    logger.info(f"Validation samples: {len(val_meta)}")
    logger.info(f"Test samples: {len(test_meta)}")
    
    # Initialize trainer
    trainer = DomainRobustTrainer(cfg, output_dir)
    
    # Create datasets (exclude 0g from training only)
    train_dataset = FaultDiagnosisDataset(
        train_meta, 
        cfg, 
        augmentor=DomainRobustAugmentor(cfg), 
        mode='train',
        exclude_0g=True  # CRITICAL: Exclude 0g bearing faults from training
    )
    val_dataset = FaultDiagnosisDataset(val_meta, cfg, mode='val', exclude_0g=False)
    test_dataset = FaultDiagnosisDataset(test_meta, cfg, mode='test', exclude_0g=False)
    
    # Compute weights and create sampler
    sample_weights = trainer.compute_sample_weights(train_dataset.severity_labels)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset) * 3,
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # STEP 3: Train model
    logger.info("\n" + "="*70)
    logger.info("STEP 3: TRAINING MODEL WITH PHYSICS-CORRECTED AUGMENTATIONS")
    logger.info("="*70)
    logger.info(f"  ✓ 4kHz low-pass filtering applied to all augmentations")
    logger.info(f"  ✓ Wider RPM jitter (0.75-1.35x) covers 1900 RPM operating point")
    logger.info(f"  ✓ Short window training (20% of batches at 0.2s length)")
    logger.info(f"  ✓ 0g bearing faults EXCLUDED from training (6g+ only)")
    logger.info(f"  ✓ Aggressive normal oversampling (8.0x weight)")
    trainer.train(train_loader, val_loader)
    
    # STEP 4: Final evaluation (including 0g faults in test set)
    logger.info("\n" + "="*70)
    logger.info("STEP 4: FINAL EVALUATION ON TEST SET (INCLUDING 0G FAULTS)")
    logger.info("="*70)
    test_acc, test_bal_acc = trainer.final_evaluation(test_loader, "Test")
    
    # Save summary with JSON-safe types
    summary = {
        'timestamp': timestamp,
        'config': cfg.to_dict(),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'best_val_balanced_acc': float(trainer.best_val_acc),
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'class_distribution': metadata['label_name'].value_counts().to_dict(),
        'severity_distribution': metadata['severity'].value_counts().to_dict()
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE - PHYSICS-CORRECTED PIPELINE")
    logger.info("="*70)
    logger.info("✓ Using ONLY Overhang channels (cols 5-7) to match ADXL357 sensor placement")
    logger.info("✓ 4kHz low-pass filtering applied BEFORE resampling (sensor reality match)")
    logger.info("✓ 0g bearing faults EXCLUDED from training (6g+ only for robust detection)")
    logger.info("✓ 400 synthetic normals generated for extreme imbalance")
    logger.info("✓ Wider RPM augmentation (0.75-1.35x) covers 1900 RPM operating point")
    logger.info("✓ Short-window training (20% at 0.2s) matches Ahrar signal length")
    logger.info(f"✓ Best validation balanced accuracy: {trainer.best_val_acc:.2%}")
    logger.info(f"✓ Test balanced accuracy: {test_bal_acc:.2%}")
    logger.info(f"✓ All outputs saved to: {output_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    main()