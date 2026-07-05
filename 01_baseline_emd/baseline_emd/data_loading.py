# utils/data_loading.py
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

import scipy
from .feature_extraction import (
    extract_features_with_rpm,
    DECIMATION_FACTOR,
    VIBRATION_COLS,
    TACH_COL,
    SAMPLING_FREQ_DECIMATED,
    FEATURE_COLUMNS,
    STRIDE
)

logger = logging.getLogger(__name__)

# === DATA SOURCES ===
DATA_SOURCES = {
    "Normal": {"root": "normal", "patterns": ["*.csv"]},
    "Imbalance": {"root": "imbalance", "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]},
    "Horiz_Misalign": {"root": "horizontal-misalignment", "subfolders": ["0.5mm", "1.0mm", "1.5mm", "2.0mm"]},
    "Vert_Misalign": {"root": "vertical-misalignment", "subfolders": ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"]},
    "Ball_Fault": {"root": "underhang/ball_fault", "subfolders": ["6g", "20g", "35g"]},
    "Outer_Race": {"root": "underhang/outer_race", "subfolders": ["6g", "20g", "35g"]},
    # "Cage_Fault": {"root": "underhang/cage_fault", "subfolders": ["6g", "20g", "35g"]}
}


def parse_severity_from_path(path_str):
    path_lower = str(path_str).lower()
    if 'overhang' in path_lower or 'underhang' in path_lower:
        m = re.search(r'(\d+)g', path_lower)
        if m: return ('bearing_g', int(m.group(1)))
    if 'imbalance' in path_lower:
        m = re.search(r'(\d+)g', path_lower)
        if m: return ('imbalance_g', int(m.group(1)))
    if 'misalign' in path_lower:
        m = re.search(r'([\d.]+)mm', path_lower)
        if m: return ('misalign_mm', float(m.group(1)))
    if 'normal' in path_lower:
        return ('normal', None)
    return ('unknown', None)


def process_file_multifault(file_path, label_name, file_id_counter):
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 4:
            return [], [], [], []
        
        skip_samples = int(50000 * 1.0) # 1 second at 50kHz
        if len(df) <= skip_samples + 4096:
            return [], [], [], None # Skip if file is too short
        
        raw_vib = df.values[skip_samples:, VIBRATION_COLS]
        raw_tach = df.values[skip_samples:, TACH_COL]

        sig_vib = scipy.signal.decimate(raw_vib, DECIMATION_FACTOR, axis=0, zero_phase=True)
        sig_tach = scipy.signal.decimate(raw_tach, DECIMATION_FACTOR, axis=0, zero_phase=True)

        feats = []
        rpm_vals = []
        file_ids = []

        for start in range(0, len(sig_vib) - 4096 + 1, STRIDE):
            window_vib = sig_vib[start:start+4096, :]
            window_tach = sig_tach[start:start+4096]
            features, rpm = extract_features_with_rpm(window_vib, window_tach, SAMPLING_FREQ_DECIMATED)
            if features:
                feats.append(features)  # ← NO LABEL APPENDED HERE
                rpm_vals.append(rpm if rpm else -1)
                file_ids.append(file_id_counter)
        return feats, rpm_vals, file_ids, label_name
    except Exception as e:
        logger.warning(f"⚠️ Failed to process {file_path.name}: {str(e)[:50]}")
        return [], [], [], None


def load_mafaulda_dataset_with_groups(base_path, cache_file="grad.pkl"):
    cache_path = Path(cache_file)
    if cache_path.exists():
        df = pd.read_pickle(cache_path)
        
        # Safely check if the cache has the correct number of features and metadata
        metadata_cols = ['label', 'rpm', 'file_id', 'severity_type', 'severity_value']
        cached_features = [col for col in df.columns if col not in metadata_cols]
        
        if 'severity_type' not in df.columns or len(cached_features) != len(FEATURE_COLUMNS):
            logger.info(f"⚠️ Outdated cache detected (Found {len(cached_features)} features, expected {len(FEATURE_COLUMNS)}). Rebuilding...")
            cache_path.unlink()  # Deletes the old file automatically
            return load_mafaulda_dataset_with_groups(base_path, cache_file)
            
        return df

    logger.info("⏳ Processing MaFaulDa Dataset with Physics-Aligned Features + Severity Metadata...")
    base = Path(base_path)
    all_feats = []
    all_rpms = []
    all_file_ids = []
    all_labels = []
    all_severity_types = []
    all_severity_values = []
    global_file_counter = 0

    for class_name, config in DATA_SOURCES.items():
        target_dir = base
        for part in config['root'].split('/'):
            target_dir = target_dir / part

        files_found = []
        if "subfolders" in config:
            for sub in config['subfolders']:
                severity_type, severity_value = parse_severity_from_path(f"{config['root']}/{sub}")
                
                subfolder_path = None
                for d in target_dir.iterdir():
                    if d.is_dir() and sub in d.name:
                        subfolder_path = d
                        break
                if subfolder_path:
                    csv_files = sorted(subfolder_path.glob("*.csv"))
                    for f in csv_files:
                        files_found.append((f, severity_type, severity_value))
                else:
                    logger.warning(f"   ⚠️ Subfolder '{sub}' not found in {target_dir}")
        elif "patterns" in config:
            for pat in config['patterns']:
                for f in sorted(target_dir.glob(pat)):
                    files_found.append((f, 'normal', None))

        logger.info(f"   📂 Processing {class_name}: {len(files_found)} files")
        for file_info in files_found:
            f_path, severity_type, severity_value = file_info
            f_feats, f_rpms, f_ids, _ = process_file_multifault(f_path, class_name, global_file_counter)
            if f_feats:
                all_feats.extend(f_feats)
                all_rpms.extend(f_rpms)
                all_file_ids.extend(f_ids)
                all_labels.extend([class_name] * len(f_feats))
                for _ in range(len(f_feats)):
                    all_severity_types.append(severity_type)
                    all_severity_values.append(severity_value)
                global_file_counter += 1

    df = pd.DataFrame(all_feats, columns=FEATURE_COLUMNS)
    df['label'] = all_labels
    df['rpm'] = all_rpms
    df['file_id'] = all_file_ids
    df['severity_type'] = all_severity_types
    df['severity_value'] = all_severity_values

    initial_len = len(df)
    df = df[(df['rpm'] >= 600) & (df['rpm'] <= 5000)].copy()
    logger.info(f"   🧹 RPM Filtering: Removed {initial_len - len(df)} windows ({len(df)} remaining)")

    df.to_pickle(cache_path)
    logger.info(f"✅ Dataset cached: {cache_file}")
    return df