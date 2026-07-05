# train.py
"""
Main training script for MaFaulDa fault diagnosis.
Orchestrates data loading, model training, and physics validation.
All heavy lifting delegated to utils modules.
"""

import logging
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GroupShuffleSplit

# Import modular utilities
from .data_loading import load_mafaulda_dataset_with_groups
from .feature_extraction import FEATURE_COLUMNS
from .physics_validation import (
    validate_rpm_stratification,
    validate_normal_false_alarms,
    run_axis_ablation_test_mafulda,
    validate_bearing_physics_mafulda,
    run_correct_severity_validation
)
from .explainability import compute_permutation_feature_importance
from scipy.spatial.distance import cosine
from itertools import combinations
from .paths import DATA_ROOT, FIGURES_DIR, MODELS_DIR, PLAYGROUND_MODELS_DIR
from .visualizations import (
    plot_confusion_matrix_academic,
    plot_pca_with_rpm_coloring,
    # plot_professional_vibration_signatures,
    cepstrum_analysis,
    waterfall_order_analysis,
    order_analysis_spectrogram,
    cosine_similarity_signal,
    fft_cosine_similarity,
    plot_pca_3d_interactive
    # plot_tsne_multiclass_interactive_with_severity,
    # plot_time_frequency_signatures,
    # visualize_fault_harmonics_mafulda,
    # plot_partial_dependence
)
# ==================== CONFIGURATION ====================
RANDOM_STATE = 42
WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 13
VIBRATION_COLS = [1,2,3]
TACH_COL = 0
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

# MaFaulDa RPM ranges
RPM_RANGES = {
    'low': (600, 1500),
    'mid': (1501, 2500),
    'high': (2501, 3800)
}
MIN_ACCEPTABLE_RPM_BAND_ACCURACY = 0.90
MAX_ALLOWED_NORMAL_FALSE_ALARM = 0.05

# Validation flags (set to True for full thesis validation)
RUN_AXIS_ABLATION_TEST = True
RUN_BEARING_FREQ_VALIDATION = True
RUN_SEVERITY_VALIDATION = True
PLOT_CONFUSION_MATRIX = True
COMPUTE_FEATURE_IMPORTANCE = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger()

# ==================== MAIN PIPELINE ====================
def main():
    start_time = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYGROUND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("🎓 MAFAULDA FAULT DIAGNOSIS: GRADUATION PROJECT PIPELINE")
    logger.info("=" * 70)

    RAW_DATA_ROOT = DATA_ROOT
    CACHE_FILE = PLAYGROUND_MODELS_DIR / "mafaulda_physics_validated_13_Decimation_new_____2026_with_Severity_2.pkl"

    # 1. Load dataset (features + metadata)
    df = load_mafaulda_dataset_with_groups(RAW_DATA_ROOT, cache_file=CACHE_FILE)
    
    logger.info(f"\n📊 Dataset Summary:")
    logger.info(f"   Total windows: {len(df)}")
    logger.info(f"   Unique files: {df['file_id'].nunique()}")
    logger.info(f"   Class distribution:")
    for cls, count in df['label'].value_counts().items():
        logger.info(f"      {cls:20s}: {count:5d} ({count/len(df)*100:.1f}%)")

    # Ensure exactly 23 numeric features
    metadata_cols = ['label', 'rpm', 'file_id', 'severity_type', 'severity_value']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # CHANGE TO 23!
    assert len(feature_cols) == 23, f"Expected 23 features, got {len(feature_cols)}"

    # 2. Prepare arrays
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df['label'].values
    groups = df['file_id'].values
    rpm_values = df['rpm'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    # 3. Leakage-proof split (file-level separation)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    rpm_test = rpm_values[test_idx]

    logger.info(f"\n✂️  Train/Test Split:")
    logger.info(f"   Train: {len(X_train)} windows from {len(np.unique(groups[train_idx]))} files")
    logger.info(f"   Test:  {len(X_test)} windows from {len(np.unique(groups[test_idx]))} files")

    # 4. Build and train pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', RandomOverSampler(random_state=RANDOM_STATE)),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE))
    ])

    logger.info("🧠 Training model...")
    pipeline.fit(X_train, y_train)

    # Extract components for downstream use
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['svm']
    X_test_scaled = scaler.transform(X_test)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    logger.info("\n" + "="*70)
    logger.info("🏆 MODEL PERFORMANCE")
    logger.info("="*70)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    logger.info(f"Overall Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Macro F1-Score:    {f1_score(y_test, y_pred, average='macro'):.4f}")

    # 6. Physics Validations
    logger.info("\n" + "="*70)
    logger.info("🔬 PHYSICS VALIDATIONS")
    logger.info("="*70)

    # RPM robustness
    rpm_valid = validate_rpm_stratification(y_test, y_pred, rpm_test, RPM_RANGES, MIN_ACCEPTABLE_RPM_BAND_ACCURACY)

    # Normal false alarms
    normal_false_alarm_rate = validate_normal_false_alarms(y_test, y_pred, class_names, MAX_ALLOWED_NORMAL_FALSE_ALARM)

    # Axis ablation (directional physics)
    if RUN_AXIS_ABLATION_TEST:
        X_train_scaled = scaler.transform(X_train)
        X_train_res, y_train_res = pipeline.named_steps['sampler'].fit_resample(X_train_scaled, y_train)

        physics_valid = run_axis_ablation_test_mafulda(
            X_train_res, X_test_scaled, y_train_res, y_test, feature_cols, class_names
        )
    else:
        physics_valid = True

    # Bearing frequency alignment
    if RUN_BEARING_FREQ_VALIDATION:
        freq_valid = validate_bearing_physics_mafulda(df.iloc[test_idx].copy(), class_names)
    else:
        freq_valid = True

    # Severity progression
    if RUN_SEVERITY_VALIDATION:
        run_correct_severity_validation(df.iloc[test_idx].copy(), class_names)

    # 7. Explainability & Visualization
    if COMPUTE_FEATURE_IMPORTANCE:
        compute_permutation_feature_importance(
            clf,
            X_test_scaled,
            y_test,
            feature_cols,
            class_names,
            save_path=str(FIGURES_DIR / "feature_importance_physics_validated.png"),
        )

    if PLOT_CONFUSION_MATRIX:
        plot_pca_3d_interactive(X_scaled, y_enc, rpm_values, class_names, le)

        vert_misalign = df[(df["label"] == "Vert_Misalign") & (df["severity_value"] == 1.90)]
        outer_race = df[(df["label"] == "Outer_Race") & (df["severity_value"] == 1.90)]
        horiz_misalign = df[(df["label"] == "Horiz_Misalign") & (df["severity_value"] == 1.0)]
        ball_fault = df[(df["label"] == "Ball_Fault") & (df["severity_value"] == 20.0)]
        normal = df[(df["label"] == "Normal")]

        samples = {
            "Vert_Misalign": vert_misalign,
            "Outer_Race": outer_race,
            "Horiz_Misalign": horiz_misalign,
            "Ball_Fault": ball_fault,
            "Normal": normal
        }

        representative_rows = {}
        for label, subset in samples.items():
            if subset.empty:
                logger.warning(f"⚠️ No samples found for {label} at the selected severity.")
                continue
            representative_rows[label] = subset.iloc[0]

        if len(representative_rows) >= 2:
            print("\n" + "=" * 60)
            print("📊 FEATURE SIMILARITY: ONE SEVERITY PER CLASS, ALL PAIRS")
            print("=" * 60)
            for label_a, label_b in combinations(representative_rows.keys(), 2):
                row_a = representative_rows[label_a]
                row_b = representative_rows[label_b]
                features_a = row_a[feature_cols].values.astype(float)
                features_b = row_b[feature_cols].values.astype(float)
                sim = 1 - cosine(features_a, features_b)

                sev_a = row_a.get("severity_value", "N/A")
                sev_b = row_b.get("severity_value", "N/A")
                print(
                    f"{label_a} (severity={sev_a}) vs {label_b} "
                    f"(severity={sev_b}): similarity={sim:.4f}"
                )
            print("=" * 60 + "\n")
        else:
            print("⚠️ Not enough representative classes to compute pairwise similarities.")
        # cepstrum_analysis(RAW_DATA_ROOT, "Ball_Fault", "20g")


        # plot_confusion_matrix_academic(y_test, y_pred, class_names)
        plot_pca_with_rpm_coloring(X_scaled, y_enc, rpm_values, class_names,le)


    # 8. Final verdict
    all_valid = (
        rpm_valid and
        (normal_false_alarm_rate <= MAX_ALLOWED_NORMAL_FALSE_ALARM) and
        physics_valid and
        freq_valid
    )

    logger.info("\n" + "="*70)
    logger.info("✅ FINAL VERDICT")
    logger.info("="*70)
    if all_valid:
        logger.info("🟢 MODEL VALIDATED: LEARNING PHYSICS, NOT NOISE")
    else:
        logger.warning("⚠️ Some validations failed — check logs.")

    # Save model for deployment
    saved_pipeline = {
        'scaler': scaler,
        'model': clf,
        'label_encoder': le,
        'feature_names': feature_cols,
        'config': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'decimation_factor': DECIMATION_FACTOR
        }
    }
    model_path = MODELS_DIR / "svm_pipeline_physics_validated.pkl"
    joblib.dump(saved_pipeline, model_path)
    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"\nTotal Runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
