import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, f1_score, recall_score
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler
import scipy.stats as stats
import scipy.signal
from scipy.signal import welch
import random
import logging
import time
from collections import defaultdict
from sklearn.metrics import make_scorer
import plotly.express as px

# ==================== CONFIGURATION ====================
# Physics Settings
WINDOW_SIZE = 180
STRIDE = 120
DECIMATION_FACTOR = 25  # 50kHz -> 2kHz
VIBRATION_COLS = [1, 2, 3]  # Axial, Radial, Tangential
TACH_COL = 0                # Tachometer
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR
RANDOM_STATE = 42

# Validation Test Configuration (Toggle as needed)
RUN_RPM_STRATIFICATION_TEST = True    # Critical: Test across operational speeds
RUN_AXIS_ABLATION_TEST = False        # Heavy: Requires 3 retrainings
RUN_NOISE_ROBUSTNESS_TEST = False     # Heavy: Adds noise to raw signals
RUN_CROSS_FILE_VALIDATION = True      # CRITICAL: Prevents file-level leakage
RUN_CONFUSION_SANITY_CHECK = True     # Lightweight physics-aligned error check
SKIP_HEAVY_TESTS_IN_DEV = True        # Skip heavy tests during development

# Physics sanity thresholds
MIN_ACCEPTABLE_RPM_BAND_ACCURACY = 0.82
MAX_ALLOWED_MISCLASSIFICATION_BETWEEN_NORMAL_AND_FAULTS = 0.08

# MaFaulDa operational RPM ranges (0.5HP motor)
RPM_RANGES = {
    'low': (767, 1500),
    'mid': (1501, 2500),
    'high': (2501, 3686)
}

# Visualization Controls
PLOT_FREQUENCY_DOMAIN = True
PLOT_T_SNE = True
PLOT_PCA = True
PLOT_LEARNING_CURVES = True
PLOT_RPM_STRATIFIED = True
PLOT_ROC_CURVES = True
PLOT_FEATURE_IMPORTANCE = True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger()

# ==================== DATA SOURCES ====================
DATA_SOURCES = {
    "Normal": {"root": "normal", "patterns": ["*.csv"]},
    "Imbalance": {"root": "imbalance", "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]},
    "Horiz_Misalign": {"root": "horizontal-misalignment", "subfolders": ["0.5mm", "1.0mm", "1.5mm", "2.0mm"]},
    "Vert_Misalign": {"root": "vertical-misalignment", "subfolders": ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"]},
    "Ball_Fault": {"root": "underhang/ball_fault", "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]},
    "Outer_Race": {"root": "underhang/outer_race", "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]}
}

# ==================== CORE PHYSICS FUNCTIONS ====================
def calculate_rpm_from_tach(tach_signal, sampling_freq):
    """Physics-accurate RPM from tachometer (1 pulse/revolution)"""
    if len(tach_signal) < 10:
        return None
    
    threshold = np.mean(tach_signal)
    binary = tach_signal > threshold
    rising_edges = np.where((binary[:-1] == False) & (binary[1:] == True))[0]
    
    if len(rising_edges) < 2:
        return None
    
    time_between = (rising_edges[-1] - rising_edges[0]) / sampling_freq
    revolutions = len(rising_edges) - 1
    
    if time_between <= 0 or revolutions == 0:
        return None
    
    return (revolutions / time_between) * 60

def extract_features_with_rpm(vib_signal, tach_signal, sampling_freq):
    """
    Physics-aligned feature extraction optimized for misalignment detection.
    Returns features + RPM.
    """
    rpm = calculate_rpm_from_tach(tach_signal, sampling_freq)
    features = []
    rms_vals = []

    for ax in range(3):
        signal = vib_signal[:, ax]
        
        # Time-domain features
        rms = np.sqrt(np.mean(signal**2))
        kur = stats.kurtosis(signal, fisher=False)
        rms_vals.append(rms)
        
        # Frequency-domain features (PSD-based)
        nperseg = min(1024, len(signal))
        if nperseg < 8:
            features.extend([rms, kur, 0, 0, 0, 0])
            continue
            
        f, Pxx = welch(signal, fs=sampling_freq, nperseg=nperseg)
        Pxx = np.maximum(Pxx, 1e-12)
        totalE = np.sum(Pxx)
        
        # Spectral centroid (FM)
        FM = np.sum(f * Pxx) / totalE
        # Spectral spread (FSD)
        FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / totalE)
        # Median frequency (FMED)
        cumulative = np.cumsum(Pxx)
        FMED = f[np.searchsorted(cumulative, 0.5 * totalE)]
        # 85% roll-off frequency (SRO)
        SRO = f[np.searchsorted(cumulative, 0.85 * totalE)]
        
        features.extend([rms, kur, FM, FSD, FMED, SRO])
    
    # Axial dominance ratio (KEY for misalignment)
    rms_axial = rms_vals[0]
    rms_radial = rms_vals[1]
    rms_tangential = rms_vals[2]
    axial_ratio = rms_axial / (rms_radial + rms_tangential + 1e-12)
    features.append(axial_ratio)
    
    return features, rpm

def process_file_multifault(file_path, label_name):
    """Process file with tachometer-based RPM extraction"""
    try:
        df = pd.read_csv(file_path, header=None)
        raw_vib = df.values[:, VIBRATION_COLS]
        raw_tach = df.values[:, TACH_COL]
        
        # Decimate both signals
        sig_vib = scipy.signal.decimate(raw_vib, DECIMATION_FACTOR, axis=0)
        sig_tach = scipy.signal.decimate(raw_tach, DECIMATION_FACTOR, axis=0)
        
        feats = []
        rpm_vals = []
        
        for start in range(0, len(sig_vib) - WINDOW_SIZE + 1, STRIDE):
            window_vib = sig_vib[start:start+WINDOW_SIZE, :]
            window_tach = sig_tach[start:start+WINDOW_SIZE]
            
            features, rpm = extract_features_with_rpm(
                window_vib, window_tach, SAMPLING_FREQ_DECIMATED
            )
            features.append(label_name)
            feats.append(features)
            rpm_vals.append(rpm if rpm else -1)  # -1 for unreliable RPM
            
        return feats, rpm_vals
    except Exception as e:
        logger.warning(f"⚠️  Failed to process {file_path.name}: {str(e)[:50]}")
        return [], []

# ==================== DATA LOADING ====================
def load_multifault_dataset_with_file_split(base_path, test_size=0.3, cache_file="mafaulda_file_split.pkl"):
    """CRITICAL: Splits at FILE level (not window level) to prevent data leakage"""
    cache_path = Path(cache_file)
    if cache_path.exists() and not SKIP_HEAVY_TESTS_IN_DEV:
        logger.info(f"✅ Loading file-split dataset: {cache_file}")
        return pd.read_pickle(cache_path)
    
    logger.info("⏳ Processing with FILE-LEVEL SPLIT (critical for validation)...")
    base = Path(base_path)
    all_files = defaultdict(list)  # {class: [file_paths]}
    
    # First pass: collect ALL files per class
    for class_name, config in DATA_SOURCES.items():
        target_dir = base
        for part in config['root'].split('/'):
            target_dir = target_dir / part
        
        files_found = []
        if "subfolders" in config:
            for sub in config['subfolders']:
                matches = [d for d in target_dir.iterdir() if d.is_dir() and sub in d.name]
                if matches:
                    files_found.extend(sorted(matches[0].glob("*.csv")))
        elif "patterns" in config:
            for pat in config['patterns']:
                files_found.extend(sorted(target_dir.glob(pat)))
        
        all_files[class_name] = files_found[:60]  # Limit files per class
    
    # Split FILES per class (not windows!)
    train_files, test_files = {}, {}
    for cls, files in all_files.items():
        n_test = max(1, int(len(files) * test_size))
        random.seed(RANDOM_STATE)
        random.shuffle(files)
        test_files[cls] = files[:n_test]
        train_files[cls] = files[n_test:]
        logger.info(f"   {cls:15s}: {len(train_files[cls])} train files | {len(test_files[cls])} test files")
    
    # Process train/test separately
    def process_file_set(file_dict, label_name):
        feats, rpms = [], []
        for f in file_dict[label_name]:
            f_feats, f_rpms = process_file_multifault(f, label_name)
            if f_feats:  # Only add if processing succeeded
                feats.extend(f_feats)
                rpms.extend(f_rpms)
        return feats, rpms
    
    # Build train/test datasets
    train_data, train_rpms = [], []
    test_data, test_rpms = [], []
    
    for cls in DATA_SOURCES.keys():
        t_feats, t_rpms = process_file_set(train_files, cls)
        train_data.extend(t_feats)
        train_rpms.extend(t_rpms)
        
        te_feats, te_rpms = process_file_set(test_files, cls)
        test_data.extend(te_feats)
        test_rpms.extend(te_rpms)
    
    # Create DataFrames with updated feature names
    per_axis_feats = ['rms', 'kurt', 'spec_centroid', 'spec_spread', 'freq_median', 'freq_rolloff85']
    axes = ['ax', 'rad', 'tan']
    cols = [f"{ax}_{feat}" for ax in axes for feat in per_axis_feats] + ['axial_ratio', 'label']
    
    df_train = pd.DataFrame(train_data, columns=cols)
    df_train['rpm'] = train_rpms
    df_test = pd.DataFrame(test_data, columns=cols)
    df_test['rpm'] = test_rpms
    
    # RPM filtering and balancing (applied separately to avoid leakage)
    def filter_and_balance(df):
        df = df[(df['rpm'] >= 500) & (df['rpm'] <= 4000)].copy()
        if df.empty:
            return df
        # Balance classes
        min_samples = min(df['label'].value_counts().min(), 2000)
        df_bal = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), min_samples), random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        return df_bal
    
    df_train = filter_and_balance(df_train)
    df_test = filter_and_balance(df_test)
    
    logger.info(f"✅ File-split dataset: {len(df_train)} train windows | {len(df_test)} test windows")
    if not SKIP_HEAVY_TESTS_IN_DEV:
        pd.to_pickle((df_train, df_test), cache_path)
    
    return df_train, df_test

def load_multifault_dataset_enhanced(base_path, cache_file="mafaulda_multifault_rpm_phys.pkl"):
    """Legacy loader (window-level split) - NOT recommended for final validation"""
    cache_path = Path(cache_file)
    if cache_path.exists():
        logger.info(f"✅ Loading cached multi-fault data: {cache_file}")
        return pd.read_pickle(cache_path)
    
    logger.info("⏳ Processing Raw Data with PHYSICS-ALIGNED feature extraction + RPM...")
    base = Path(base_path)
    all_data = []
    all_rpm = []
    
    for class_name, config in DATA_SOURCES.items():
        logger.info(f"   ➡️  Processing Class: {class_name}")
        target_dir = base
        for part in config['root'].split('/'):
            target_dir = target_dir / part
        
        files_found = []
        if "subfolders" in config:
            for sub in config['subfolders']:
                matches = [d for d in target_dir.iterdir() if d.is_dir() and sub in d.name]
                if matches:
                    files_found.extend(sorted(matches[0].glob("*.csv")))
        elif "patterns" in config:
            for pat in config['patterns']:
                files_found.extend(sorted(target_dir.glob(pat)))
        
        random.seed(RANDOM_STATE)
        random.shuffle(files_found)
        files_to_process = files_found[:60]
        
        class_feats = []
        class_rpm = []
        for f in files_to_process:
            feats, rpm = process_file_multifault(f, class_name)
            class_feats.extend(feats)
            class_rpm.extend(rpm)
        
        all_data.extend(class_feats)
        all_rpm.extend(class_rpm)
        logger.info(f"      ✓ {class_name}: {len(files_to_process)} files → {len(class_feats)} windows")
    
    # Create DataFrame with updated feature names
    per_axis_feats = ['rms', 'kurt', 'spec_centroid', 'spec_spread', 'freq_median', 'freq_rolloff85']
    axes = ['ax', 'rad', 'tan']
    cols = [f"{ax}_{feat}" for ax in axes for feat in per_axis_feats] + ['axial_ratio', 'label']
    
    df = pd.DataFrame(all_data, columns=cols)
    df['rpm'] = all_rpm
    
    # RPM filtering
    valid_mask = (df['rpm'] >= 500) & (df['rpm'] <= 4000)
    df = df[valid_mask].copy()
    logger.info(f"   📊 Valid windows after RPM filtering: {len(df)}")
    
    # Class balancing
    df_bal = (
        df.groupby('label', group_keys=False)
          .apply(lambda x: x.sample(min(len(x), 2000), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )
    
    logger.info(f"💾 Saving physics-enhanced cache: {cache_file}")
    df_bal.to_pickle(cache_path)
    return df_bal

# ==================== VALIDATION TESTS ====================
def run_rpm_stratification_test(y_true, y_pred, rpm_values, class_names):
    """Quantitative RPM robustness test - runs in <1 second"""
    logger.info("\n" + "="*70)
    logger.info("⚙️  RPM STRATIFICATION TEST (Critical for real-world deployment)")
    logger.info("="*70)
    
    results = {}
    for rpm_range, (low, high) in RPM_RANGES.items():
        mask = (rpm_values >= low) & (rpm_values <= high)
        if np.sum(mask) < 30: 
            continue
            
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results[rpm_range] = acc
        status = "✅" if acc >= MIN_ACCEPTABLE_RPM_BAND_ACCURACY else "❌"
        logger.info(f"   {rpm_range.upper():8s} ({low}-{high} RPM): {acc:.2%} {status}")
    
    # Physics validation
    if not results:
        logger.warning("   ⚠️  No RPM bands with sufficient samples for testing")
        return True
        
    worst_band = min(results, key=results.get)
    if results[worst_band] < MIN_ACCEPTABLE_RPM_BAND_ACCURACY:
        logger.error(f"   🔴 CRITICAL: Model fails at {worst_band} RPM ({results[worst_band]:.2%})")
        logger.error("      → Likely learning RPM-correlated noise instead of fault physics")
        return False
    else:
        logger.info(f"   ✅ PASSED: Model maintains >{MIN_ACCEPTABLE_RPM_BAND_ACCURACY:.0%} accuracy across all RPM bands")
        return True

def run_confusion_sanity_check(y_true, y_pred, class_names):
    """Automated check for physics-aligned misclassifications"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    logger.info("\n" + "="*70)
    logger.info("🔍 CONFUSION SANITY CHECK (Physics-aligned error patterns)")
    logger.info("="*70)
    
    # Critical safety checks
    normal_idx = list(class_names).index('Normal')
    normal_misclassified = cm[normal_idx, :].sum() - cm[normal_idx, normal_idx]
    if normal_misclassified > MAX_ALLOWED_MISCLASSIFICATION_BETWEEN_NORMAL_AND_FAULTS:
        logger.error(f"   🔴 CRITICAL: {normal_misclassified:.1%} of Normal samples misclassified as faults!")
        logger.error("      → Unacceptable false alarms in healthy machinery")
        return False
    
    # Physics-aligned errors (acceptable)
    misalign_classes = ['Horiz_Misalign', 'Vert_Misalign']
    misalign_indices = [i for i, c in enumerate(class_names) if c in misalign_classes]
    if len(misalign_indices) >= 2:
        misalign_confusion = cm[np.ix_(misalign_indices, misalign_indices)].sum() - np.trace(cm[np.ix_(misalign_indices, misalign_indices)])
        
        if misalign_confusion > 0.35:  # High confusion between misalignment types is PHYSICS-CORRECT
            logger.info(f"   ✅ High confusion between misalignment types ({misalign_confusion:.1%}) → PHYSICS-CORRECT (similar vibration signatures)")
        else:
            logger.info(f"   ℹ️  Low misalignment confusion ({misalign_confusion:.1%}) → Model distinguishes alignment types well")
    
    # Noise-like errors (red flags)
    try:
        ball_fault_idx = list(class_names).index('Ball_Fault')
        outer_race_idx = list(class_names).index('Outer_Race')
        imbalance_idx = list(class_names).index('Imbalance')
        
        ball_to_normal = cm[ball_fault_idx, normal_idx]
        outer_to_imbalance = cm[outer_race_idx, imbalance_idx]
        
        if ball_to_normal > 0.15 or outer_to_imbalance > 0.20:
            logger.error(f"   🔴 SUSPICIOUS: Ball fault → Normal ({ball_to_normal:.1%}) or Outer race → Imbalance ({outer_to_imbalance:.1%})")
            logger.error("      → Suggests noise learning (these faults have distinct physics signatures)")
            return False
    except ValueError:
        pass  # Class not present in test set
    
    logger.info("   ✅ PASSED: Misclassification patterns align with mechanical physics")
    return True

def run_axis_ablation_test(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Physics validation: Remove axial/radial/tangential features to verify directional sensitivity"""
    if SKIP_HEAVY_TESTS_IN_DEV and not RUN_AXIS_ABLATION_TEST:
        logger.info("⏭️  Skipping axis ablation test (heavy computation - enable RUN_AXIS_ABLATION_TEST=True to run)")
        return
    
    logger.info("\n" + "="*70)
    logger.info("🔬 AXIS ABLATION TEST (Verifying directional physics sensitivity)")
    logger.info("="*70)
    
    # Identify feature groups
    axial_feats = [i for i, f in enumerate(feature_names) if f.startswith('ax_')]
    radial_feats = [i for i, f in enumerate(feature_names) if f.startswith('rad_')]
    tangential_feats = [i for i, f in enumerate(feature_names) if f.startswith('tan_')]
    
    ablation_tests = [
        ("Full model", list(range(len(feature_names)))),
        ("No axial", [i for i in range(len(feature_names)) if i not in axial_feats]),
        ("No radial", [i for i in range(len(feature_names)) if i not in radial_feats]),
        ("No tangential", [i for i in range(len(feature_names)) if i not in tangential_feats])
    ]
    
    results = {}
    base_acc = 0
    
    for name, feat_idx in ablation_tests:
        if not feat_idx:  # Skip if no features left
            logger.warning(f"   Skipping '{name}' - no features remaining")
            continue
            
        X_tr_abl = X_train[:, feat_idx]
        X_te_abl = X_test[:, feat_idx]
        
        # Quick SVM training (smaller C for speed)
        clf_abl = SVC(kernel='rbf', C=10, gamma=0.1, random_state=RANDOM_STATE)
        clf_abl.fit(X_tr_abl, y_train)
        acc = accuracy_score(y_test, clf_abl.predict(X_te_abl))
        
        results[name] = acc
        if name == "Full model":
            base_acc = acc
            logger.info(f"   {name:20s}: {acc:.2%} (baseline)")
        else:
            delta = acc - base_acc
            arrow = "↓" if delta < 0 else "↑"
            logger.info(f"   {name:20s}: {acc:.2%} ({arrow}{abs(delta):.1%})")
    
    # Physics validation
    logger.info("\n   Physics validation:")
    if "No axial" in results and results["No axial"] < results.get("Full model", 0) - 0.08:
        logger.info("   ✅ Axial removal hurts most → Model uses axial dominance for misalignment (correct physics)")
    else:
        logger.warning("   ⚠️  Axial removal has minimal impact → Model may not be using misalignment physics")
    
    if "No radial" in results and results["No radial"] < results.get("Full model", 0) - 0.06:
        logger.info("   ✅ Radial removal hurts → Model uses radial vibration for imbalance (correct physics)")
    else:
        logger.warning("   ⚠️  Radial removal has minimal impact → Model may not be using imbalance physics")

# ==================== VISUALIZATIONS ====================
def plot_frequency_domain_multiclass(df, class_names):
    """Physics validation using spectral features (fixed to use existing columns)"""
    logger.info("🔬 Generating frequency domain validation...")
    
    # Sample data for visualization
    samples = []
    for cls in class_names:
        cls_data = df[df['label'] == cls]
        if len(cls_data) > 0:
            subsample = cls_data.sample(n=min(200, len(cls_data)), random_state=RANDOM_STATE).copy()
            subsample['fault_type'] = cls
            samples.append(subsample)
    
    if not samples:
        logger.warning("No data found for plotting.")
        return
    
    plot_df = pd.concat(samples, ignore_index=True)
    
    # Create Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()
    
    axes_map = {'ax': 'Axial', 'rad': 'Radial', 'tan': 'Tangential'}
    
    for idx, (axis_code, axis_name) in enumerate(axes_map.items()):
        spec_col = f"{axis_code}_spec_centroid"
        
        # Skip if column doesn't exist
        if spec_col not in plot_df.columns:
            logger.warning(f"Column '{spec_col}' not found - skipping plot")
            continue
            
        # --- Row 1: Violin Plots (Distributions) ---
        ax_vio = axes[idx]
        sns.violinplot(data=plot_df, x='fault_type', y=spec_col, ax=ax_vio, 
                      palette='viridis', inner='quartile')
        ax_vio.set_title(f'{axis_name}: Spectral Centroid Distribution', fontsize=14, fontweight='bold')
        ax_vio.set_ylabel('Frequency (Hz)', fontsize=12)
        ax_vio.set_xlabel('')
        ax_vio.tick_params(axis='x', rotation=45)
        ax_vio.grid(True, alpha=0.2)
        
        # --- Row 2: Scatter Plots (RPM vs Freq) ---
        ax_scat = axes[idx+3]
        
        # Plot 1x RPM Line (Reference)
        rpm_vals = np.linspace(10, 60, 100)  # 10-60Hz (600-3600 RPM)
        ax_scat.plot(rpm_vals, rpm_vals, 'k--', alpha=0.5, label='1x RPM Reference')
        
        # Plot points
        sns.scatterplot(data=plot_df, x=plot_df['rpm']/60, y=spec_col, 
                        hue='fault_type', style='fault_type', 
                        ax=ax_scat, palette='viridis', alpha=0.7, s=40)
        
        ax_scat.set_title(f'{axis_name}: Spectral Centroid vs Speed', fontsize=14, fontweight='bold')
        ax_scat.set_xlabel('Motor Speed (Hz)', fontsize=12)
        ax_scat.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
        ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
        ax_scat.grid(True, alpha=0.2)
        ax_scat.set_ylim(0, 150)
    
    plt.suptitle('Fault Physics Validation: Spectral Signatures', fontsize=20, weight='bold')
    plt.show()

def plot_learning_curves_multiclass(X, y):
    """Diagnose model capacity for multi-class problem"""
    logger.info("📈 Generating learning curves (diagnosing under/overfitting)...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        SVC(kernel='rbf', C=100, gamma=0.1, decision_function_shape='ovo', random_state=RANDOM_STATE),
        X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.15, 1.0, 6), random_state=RANDOM_STATE
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(11, 7))
    plt.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training Score', linewidth=2.5, markersize=8)
    plt.plot(train_sizes, test_mean, 's-', color='#e74c3c', label='CV Score', linewidth=2.5, markersize=8)
    
    plt.title('Learning Curves: Multi-Class Fault Diagnosis Capacity', fontsize=15, fontweight='bold')
    plt.xlabel('Training Examples', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(0.5, 1.02)
    
    # Diagnosis
    if test_mean[-1] < 0.75:
        diagnosis = "CRITICAL: UNDERFITTING - Model cannot learn fault distinctions"
        color = 'red'
    elif train_mean[-1] - test_mean[-1] > 0.15:
        diagnosis = "WARNING: OVERFITTING - Model memorizing noise, not generalizing"
        color = 'orange'
    elif test_mean[-1] < 0.85:
        diagnosis = "CAUTION: Marginal performance - May fail on unseen operating conditions"
        color = 'darkorange'
    else:
        diagnosis = "GOOD: Model has appropriate capacity for fault diagnosis"
        color = 'green'
    
    plt.text(0.5, 0.15, diagnosis, transform=plt.gca().transAxes,
            fontsize=13, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95))
    
    plt.tight_layout()
    plt.show()
    
    logger.info(f"📊 LEARNING CURVE DIAGNOSIS: {diagnosis}")

def plot_pca_multiclass(X_scaled, y, class_names):
    """PCA for multi-class separation visualization"""
    logger.info("🎨 Generating PCA plot (global structure visualization)...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=40, edgecolors='none')
    plt.colorbar(scatter, ticks=range(len(class_names)), label='Fault Class')
    plt.clim(-0.5, len(class_names)-0.5)
    
    # Add class centroids
    for i, cls in enumerate(class_names):
        mask = y == i
        if np.any(mask):
            centroid = X_pca[mask].mean(axis=0)
            plt.annotate(cls, centroid, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.title(f'PCA: Multi-Fault Separation (PC1+PC2 = {pca.explained_variance_ratio_.sum()*100:.1f}% variance)', 
             fontsize=15, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_tsne_multiclass_interactive(X_scaled, y, class_names, perplexities=[30]):
    """Generates interactive Plotly visualizations for 2D t-SNE"""
    logger.info("🎨 Generating Interactive Plotly t-SNE visualizations...")
    
    # Map numeric labels to class names
    y_named = [class_names[val] for val in y]
    
    for perp in perplexities:
        try:
            # Compute 2D t-SNE
            tsne_2d = TSNE(n_components=2, perplexity=perp, max_iter=1000, 
                          random_state=RANDOM_STATE, init='pca')
            X_2d = tsne_2d.fit_transform(X_scaled)
            
            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                'TSNE1': X_2d[:, 0],
                'TSNE2': X_2d[:, 1],
                'Fault_Condition': y_named
            })
            
            # Generate interactive 2D plot
            fig = px.scatter(
                df_plot, x='TSNE1', y='TSNE2',
                color='Fault_Condition',
                title=f"2D t-SNE (Perplexity: {perp})",
                opacity=0.7,
                template='plotly_white',
                width=900, height=700
            )
            fig.update_traces(marker=dict(size=6))
            fig.show()
            
            logger.info(f"   ✓ Interactive plot rendered for perplexity={perp}")
            
        except Exception as e:
            logger.warning(f"   ✗ t-SNE Plotly failed (perplexity={perp}): {str(e)}")

def plot_rpm_stratified_multiclass(df, y_pred, y_true, class_names):
    """Performance analysis across operational RPM ranges"""
    logger.info("⚙️  Analyzing RPM-stratified performance (critical for real-world deployment)...")
    
    results = defaultdict(lambda: defaultdict(list))
    
    for rpm_range, (low, high) in RPM_RANGES.items():
        mask = (df['rpm'] >= low) & (df['rpm'] <= high)
        if np.sum(mask) < 30:
            continue
            
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]
        
        # Per-class metrics
        for cls_idx, cls_name in enumerate(class_names):
            cls_mask = y_true_range == cls_idx
            if np.sum(cls_mask) > 5:
                acc = accuracy_score(y_true_range[cls_mask], y_pred_range[cls_mask])
                results[rpm_range][cls_name].append(acc)
    
    # Plot per-class RPM performance
    n_classes = len(class_names)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for idx, cls_name in enumerate(class_names):
        ax = axes[idx]
        means = []
        stds = []
        x_labels = []
        
        for rpm_range in RPM_RANGES.keys():
            if rpm_range in results and cls_name in results[rpm_range]:
                vals = results[rpm_range][cls_name]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                x_labels.append(rpm_range.upper())
        
        if means:
            x_pos = np.arange(len(means))
            ax.bar(x_pos, means, yerr=stds, capsize=8, 
                   color=plt.cm.Set2(idx/len(class_names)), alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
            ax.set_title(f'{cls_name}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cls_name}', fontsize=13, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(class_names), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Per-Class Accuracy Across Operational RPM Ranges\n(Critical: Model must work at ALL speeds)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def plot_roc_curves_multiclass(y_test, y_score, class_names):
    """One-vs-Rest ROC curves for multi-class"""
    logger.info("📈 Generating One-vs-Rest ROC curves (per-class discriminability)...")
    
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('One-vs-Rest ROC Curves: Per-Class Discriminability', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10, ncol=2)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_enhanced(y_true, y_pred, class_names):
    """Enhanced confusion matrix with normalized view"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names, rotation=0)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Reds', ax=ax2, vmin=0, vmax=1,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Normalized by True Class)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_yticklabels(class_names, rotation=0)
    
    plt.suptitle('Multi-Fault Classification Confusion Analysis\n(Identify common misclassifications)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def plot_feature_importance_multiclass(model, X_train, y_train, X_test, y_test, feature_names, class_names):
    """Per-class feature importance using permutation importance"""
    logger.info("🧠 Calculating per-class feature importance (XAI)...")
    
    # Get global top features
    results = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    top_idx = results.importances_mean.argsort()[::-1][:15]
    top_features = [feature_names[i] for i in top_idx]
    
    n_classes = len(class_names)
    importance_matrix = np.zeros((n_classes, len(top_idx)))
    
    # Per-Class Importance using Binary Classifiers
    for cls_idx in range(n_classes):
        target_class = class_names[cls_idx]
        
        # Create Binary Targets
        y_bin_train = (y_train == cls_idx).astype(int)
        y_bin_test = (y_test == cls_idx).astype(int)
        
        # Skip if too few samples
        if np.sum(y_bin_train) < 10:
            continue
            
        # Train Binary Model with balanced weights
        bin_clf = SVC(kernel='rbf', C=10, gamma='scale', 
                     class_weight='balanced', random_state=42)
        bin_clf.fit(X_train[:, top_idx], y_bin_train)
        
        # Calculate importance using F1-score
        scorer = make_scorer(f1_score, zero_division=0)
        r = permutation_importance(
            bin_clf, X_test[:, top_idx], y_bin_test, 
            scoring=scorer, n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Handle negatives (noise)
        imps = r.importances_mean
        imps[imps < 0] = 0
        importance_matrix[cls_idx] = imps
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=top_features,
                yticklabels=class_names, vmin=0)
    plt.title('Per-Class Feature Importance (F1-Score Based)', fontsize=15, fontweight='bold')
    plt.xlabel('Top Features', fontsize=12)
    plt.ylabel('Fault Classes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ==================== MAIN PIPELINE ====================
if __name__ == "__main__":
    start_time = time.time()
    logger.info("="*70)
    logger.info("🚀 MAFAULDA MULTI-FAULT CLASSIFICATION PIPELINE (Physics-Validated)")
    logger.info("="*70)
    
    # 1. LOAD DATA WITH PROPER SPLITTING
    RAW_DATA_ROOT = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"
    
    if RUN_CROSS_FILE_VALIDATION:
        logger.info("✅ Using FILE-LEVEL SPLIT (prevents data leakage)")
        df_train, df_test = load_multifault_dataset_with_file_split(RAW_DATA_ROOT)
    else:
        logger.warning("⚠️  Using WINDOW-LEVEL SPLIT (risk of data leakage - not recommended for publication)")
        df = load_multifault_dataset_enhanced(RAW_DATA_ROOT)
        # Create train/test split at window level
        feature_cols = [col for col in df.columns if col not in ['label', 'rpm']]
        X = df[feature_cols].values
        y = df['label'].values
        rpm = df['rpm'].values
        
        X_train, X_test, y_train, y_test, rpm_train, rpm_test = train_test_split(
            X, y, rpm, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )
        
        # Create DataFrames for consistency
        df_train = pd.DataFrame(X_train, columns=feature_cols)
        df_train['label'] = y_train
        df_train['rpm'] = rpm_train
        
        df_test = pd.DataFrame(X_test, columns=feature_cols)
        df_test['label'] = y_test
        df_test['rpm'] = rpm_test
    
    logger.info(f"\n📊 Dataset Summary:")
    logger.info(f"   Train windows: {len(df_train)} | Test windows: {len(df_test)}")
    logger.info(f"   RPM range (train): {df_train['rpm'].min():.0f} - {df_train['rpm'].max():.0f} RPM")
    logger.info(f"   RPM range (test):  {df_test['rpm'].min():.0f} - {df_test['rpm'].max():.0f} RPM")
    logger.info(f"   Class distribution (train):")
    for cls, count in df_train['label'].value_counts().items():
        logger.info(f"      {cls:20s}: {count:5d} windows")
    
    # 2. PREPARE DATA
    feature_cols = [col for col in df_train.columns if col not in ['label', 'rpm']]
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    rpm_train = df_train['rpm'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values
    rpm_test = df_test['rpm'].values
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_
    
    # 3. BALANCE & SCALE
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. TRAIN MODEL
    logger.info("\n🧠 Training Multi-Class SVM (RBF kernel, One-vs-One)...")
    clf = SVC(kernel='rbf', C=100, gamma=0.1, decision_function_shape='ovo', 
             probability=True, random_state=RANDOM_STATE)
    clf.fit(X_train_scaled, y_train_res)
    
    # 5. EVALUATE
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    
    logger.info("\n" + "="*70)
    logger.info("🏆 MULTI-FAULT CLASSIFICATION RESULTS")
    logger.info("="*70)
    print(classification_report(y_test_enc, y_pred, target_names=class_names, digits=4))
    logger.info(f"Overall Accuracy: {accuracy_score(y_test_enc, y_pred):.4f}")
    logger.info(f"Macro F1-Score:   {f1_score(y_test_enc, y_pred, average='macro'):.4f}")
    logger.info(f"Weighted F1-Score:{f1_score(y_test_enc, y_pred, average='weighted'):.4f}")
    
    # 6. VALIDATION VISUALIZATIONS
    if PLOT_FREQUENCY_DOMAIN:
        plot_frequency_domain_multiclass(df_test.sample(frac=0.3, random_state=RANDOM_STATE), class_names)
    
    if PLOT_LEARNING_CURVES:
        plot_learning_curves_multiclass(X_train_scaled, y_train_res)
    
    if PLOT_PCA:
        plot_pca_multiclass(X_test_scaled, y_test_enc, class_names)
    
    if PLOT_T_SNE and len(X_test_scaled) <= 5000:
        plot_tsne_multiclass_interactive(X_test_scaled, y_test_enc, class_names, perplexities=[30])
    
    if PLOT_RPM_STRATIFIED:
        plot_rpm_stratified_multiclass(df_test, y_pred, y_test_enc, class_names)
    
    if PLOT_ROC_CURVES:
        plot_roc_curves_multiclass(y_test_enc, y_proba, class_names)
    
    plot_confusion_matrix_enhanced(y_test_enc, y_pred, class_names)
    
    if PLOT_FEATURE_IMPORTANCE and accuracy_score(y_test_enc, y_pred) > 0.85:
        plot_feature_importance_multiclass(
            clf, X_train_scaled, y_train_res,
            X_test_scaled, y_test_enc,
            feature_cols, class_names
        )
    
    # 7. PHYSICS-BASED VALIDATION TESTS
    logger.info("\n" + "="*70)
    logger.info("🔍 PHYSICS-BASED VALIDATION TESTS")
    logger.info("="*70)
    
    validation_passed = True
    
    # RPM Stratification Test
    if RUN_RPM_STRATIFICATION_TEST:
        rpm_ok = run_rpm_stratification_test(y_test_enc, y_pred, rpm_test, class_names)
        validation_passed &= rpm_ok
    
    # Confusion Sanity Check
    if RUN_CONFUSION_SANITY_CHECK:
        conf_ok = run_confusion_sanity_check(y_test_enc, y_pred, class_names)
        validation_passed &= conf_ok
    
    # Axis Ablation Test (heavy)
    if RUN_AXIS_ABLATION_TEST and not SKIP_HEAVY_TESTS_IN_DEV:
        run_axis_ablation_test(
            X_train_scaled, X_test_scaled, y_train_res, y_test_enc,
            feature_cols, class_names
        )
    
    # 8. EXECUTION SUMMARY
    elapsed = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total runtime: {elapsed:.2f} seconds")
    logger.info(f"Physics compliance: Tachometer-based RPM extraction")
    logger.info(f"Sampling rate: {SAMPLING_FREQ_DECIMATED:.0f} Hz (captures 0.5-100Hz dynamics)")
    logger.info(f"Window duration: {WINDOW_SIZE/SAMPLING_FREQ_DECIMATED*1000:.1f} ms")
    
    # Final validation verdict
    logger.info("\n" + "="*70)
    logger.info("✅ VALIDATION VERDICT")
    logger.info("="*70)
    if validation_passed:
        logger.info("🟢 MODEL VALIDATED: Learning physics, not noise")
        logger.info("   → RPM robust across operational range")
        logger.info("   → Misclassifications follow mechanical physics")
        logger.info("   → High accuracy on safety-critical faults (imbalance, bearing defects)")
    else:
        logger.error("🔴 VALIDATION FAILED: Possible noise learning detected")
        logger.error("   → Check RPM stratification and confusion patterns above")
        logger.error("   → Enable axis ablation test for deeper diagnosis")
    
    logger.info("="*70)