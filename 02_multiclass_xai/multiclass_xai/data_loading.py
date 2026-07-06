import logging
import re
from pathlib import Path

import pandas as pd
import scipy.signal

from .feature_extraction import (
    DECIMATION_FACTOR,
    FEATURE_COLUMNS,
    SAMPLING_FREQ_DECIMATED,
    STRIDE,
    TACH_COL,
    VIBRATION_COLS,
    extract_features_with_rpm,
)


logger = logging.getLogger(__name__)

DATA_SOURCES = {
    "Normal": {"root": "normal", "patterns": ["*.csv"]},
    "Imbalance": {"root": "imbalance", "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]},
    "Horiz_Misalign": {"root": "horizontal-misalignment", "subfolders": ["0.5mm", "1.0mm", "1.5mm", "2.0mm"]},
    "Vert_Misalign": {"root": "vertical-misalignment", "subfolders": ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"]},
    "Ball_Fault": {"root": "underhang/ball_fault", "subfolders": ["6g", "20g", "35g"]},
    "Outer_Race": {"root": "underhang/outer_race", "subfolders": ["6g", "20g", "35g"]},
}


def parse_severity_from_path(path_str):
    path_lower = str(path_str).lower()
    if "overhang" in path_lower or "underhang" in path_lower:
        match = re.search(r"(\d+)g", path_lower)
        if match:
            return "bearing_g", int(match.group(1))
    if "imbalance" in path_lower:
        match = re.search(r"(\d+)g", path_lower)
        if match:
            return "imbalance_g", int(match.group(1))
    if "misalign" in path_lower:
        match = re.search(r"([\d.]+)mm", path_lower)
        if match:
            return "misalign_mm", float(match.group(1))
    if "normal" in path_lower:
        return "normal", None
    return "unknown", None


def process_file_multifault(file_path, label_name, file_id_counter):
    try:
        dataframe = pd.read_csv(file_path, header=None)
        if dataframe.shape[1] <= max([TACH_COL] + VIBRATION_COLS):
            return [], [], [], None

        skip_samples = int(50000 * 1.0)
        if len(dataframe) <= skip_samples + 4096:
            return [], [], [], None

        raw_vib = dataframe.values[skip_samples:, VIBRATION_COLS]
        raw_tach = dataframe.values[skip_samples:, TACH_COL]

        sig_vib = scipy.signal.decimate(raw_vib, DECIMATION_FACTOR, axis=0, zero_phase=True)
        sig_tach = scipy.signal.decimate(raw_tach, DECIMATION_FACTOR, axis=0, zero_phase=True)

        feats = []
        rpm_vals = []
        file_ids = []

        for start in range(0, len(sig_vib) - 4096 + 1, STRIDE):
            window_vib = sig_vib[start : start + 4096, :]
            window_tach = sig_tach[start : start + 4096]
            features, rpm = extract_features_with_rpm(window_vib, window_tach, SAMPLING_FREQ_DECIMATED)
            if features:
                feats.append(features)
                rpm_vals.append(rpm if rpm else -1)
                file_ids.append(file_id_counter)

        return feats, rpm_vals, file_ids, label_name
    except Exception as exc:
        logger.warning("Failed to process %s: %s", Path(file_path).name, str(exc)[:80])
        return [], [], [], None


def load_mafaulda_dataset_with_groups(base_path, cache_file="grad.pkl"):
    cache_path = Path(cache_file)
    if cache_path.exists():
        dataframe = pd.read_pickle(cache_path)
        metadata_cols = ["label", "rpm", "file_id", "severity_type", "severity_value"]
        cached_features = [col for col in dataframe.columns if col not in metadata_cols]

        if "severity_type" not in dataframe.columns or len(cached_features) != len(FEATURE_COLUMNS):
            logger.info(
                "Outdated cache detected: found %s features, expected %s. Rebuilding.",
                len(cached_features),
                len(FEATURE_COLUMNS),
            )
            cache_path.unlink()
            return load_mafaulda_dataset_with_groups(base_path, cache_file)

        return dataframe

    logger.info("Processing MaFaulDa dataset with physics-aligned features and severity metadata.")
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
        for part in config["root"].split("/"):
            target_dir = target_dir / part

        files_found = []
        if not target_dir.exists():
            logger.warning("Class directory not found for %s: %s", class_name, target_dir)
            continue

        if "subfolders" in config:
            for subfolder in config["subfolders"]:
                severity_type, severity_value = parse_severity_from_path(f"{config['root']}/{subfolder}")
                subfolder_path = None
                for directory in target_dir.iterdir():
                    if directory.is_dir() and subfolder in directory.name:
                        subfolder_path = directory
                        break

                if subfolder_path:
                    files_found.extend((file_path, severity_type, severity_value) for file_path in sorted(subfolder_path.glob("*.csv")))
                else:
                    logger.warning("Subfolder %s not found in %s", subfolder, target_dir)
        elif "patterns" in config:
            for pattern in config["patterns"]:
                files_found.extend((file_path, "normal", None) for file_path in sorted(target_dir.glob(pattern)))

        logger.info("Processing %s: %s files", class_name, len(files_found))
        for file_path, severity_type, severity_value in files_found:
            f_feats, f_rpms, f_ids, _ = process_file_multifault(file_path, class_name, global_file_counter)
            if f_feats:
                all_feats.extend(f_feats)
                all_rpms.extend(f_rpms)
                all_file_ids.extend(f_ids)
                all_labels.extend([class_name] * len(f_feats))
                all_severity_types.extend([severity_type] * len(f_feats))
                all_severity_values.extend([severity_value] * len(f_feats))
                global_file_counter += 1

    dataframe = pd.DataFrame(all_feats, columns=FEATURE_COLUMNS)
    dataframe["label"] = all_labels
    dataframe["rpm"] = all_rpms
    dataframe["file_id"] = all_file_ids
    dataframe["severity_type"] = all_severity_types
    dataframe["severity_value"] = all_severity_values

    initial_len = len(dataframe)
    dataframe = dataframe[(dataframe["rpm"] >= 600) & (dataframe["rpm"] <= 5000)].copy()
    logger.info("RPM filtering removed %s windows (%s remaining).", initial_len - len(dataframe), len(dataframe))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(cache_path)
    logger.info("Dataset cached: %s", cache_path)
    return dataframe
