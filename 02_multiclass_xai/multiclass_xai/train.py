"""
Main training script for MaFaulDa multi-class fault diagnosis.
"""

import logging
import time
from itertools import combinations

import joblib
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .data_loading import load_mafaulda_dataset_with_groups
from .explainability import compute_permutation_feature_importance
from .feature_extraction import DECIMATION_FACTOR, FEATURE_COLUMNS, STRIDE, WINDOW_SIZE
from .paths import DATA_ROOT, FIGURES_DIR, MODELS_DIR, PLAYGROUND_MODELS_DIR
from .physics_validation import (
    run_axis_ablation_test_mafulda,
    run_correct_severity_validation,
    validate_bearing_physics_mafulda,
    validate_normal_false_alarms,
    validate_rpm_stratification,
)
from .visualizations import (
    plot_pca_3d_interactive,
    plot_pca_with_rpm_coloring,
)


RANDOM_STATE = 42
RPM_RANGES = {
    "low": (600, 1500),
    "mid": (1501, 2500),
    "high": (2501, 3800),
}
MIN_ACCEPTABLE_RPM_BAND_ACCURACY = 0.90
MAX_ALLOWED_NORMAL_FALSE_ALARM = 0.05

RUN_AXIS_ABLATION_TEST = True
RUN_BEARING_FREQ_VALIDATION = True
RUN_SEVERITY_VALIDATION = True
PLOT_CONFUSION_MATRIX = True
COMPUTE_FEATURE_IMPORTANCE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger()


def _select_representative_rows(dataframe, feature_cols):
    samples = {
        "Vert_Misalign": dataframe[(dataframe["label"] == "Vert_Misalign") & (dataframe["severity_value"] == 1.90)],
        "Outer_Race": dataframe[(dataframe["label"] == "Outer_Race") & (dataframe["severity_value"] == 20.0)],
        "Horiz_Misalign": dataframe[(dataframe["label"] == "Horiz_Misalign") & (dataframe["severity_value"] == 1.0)],
        "Ball_Fault": dataframe[(dataframe["label"] == "Ball_Fault") & (dataframe["severity_value"] == 20.0)],
        "Normal": dataframe[dataframe["label"] == "Normal"],
    }

    representative_rows = {}
    for label, subset in samples.items():
        if subset.empty:
            fallback = dataframe[dataframe["label"] == label]
            if fallback.empty:
                logger.warning("No samples found for %s.", label)
                continue
            subset = fallback
        representative_rows[label] = subset.iloc[0]

    if len(representative_rows) < 2:
        print("Not enough representative classes to compute pairwise similarities.")
        return

    print("\n" + "=" * 60)
    print("FEATURE SIMILARITY: ONE SEVERITY PER CLASS, ALL PAIRS")
    print("=" * 60)
    for label_a, label_b in combinations(representative_rows.keys(), 2):
        row_a = representative_rows[label_a]
        row_b = representative_rows[label_b]
        features_a = row_a[feature_cols].values.astype(float)
        features_b = row_b[feature_cols].values.astype(float)
        similarity = 1 - cosine(features_a, features_b)
        print(
            f"{label_a} (severity={row_a.get('severity_value', 'N/A')}) vs "
            f"{label_b} (severity={row_b.get('severity_value', 'N/A')}): "
            f"similarity={similarity:.4f}"
        )
    print("=" * 60 + "\n")


def main():
    start_time = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYGROUND_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("%s", "=" * 70)
    logger.info("MAFAULDA FAULT DIAGNOSIS: GRADUATION PROJECT PIPELINE")
    logger.info("%s", "=" * 70)

    cache_file = PLAYGROUND_MODELS_DIR / "mafaulda_physics_validated_13_decimation_2026_with_severity.pkl"
    dataframe = load_mafaulda_dataset_with_groups(DATA_ROOT, cache_file=cache_file)

    logger.info("Dataset summary:")
    logger.info("  Total windows: %s", len(dataframe))
    logger.info("  Unique files: %s", dataframe["file_id"].nunique())
    logger.info("  Class distribution:")
    for class_name, count in dataframe["label"].value_counts().items():
        logger.info("    %-20s: %5d (%0.1f%%)", class_name, count, count / len(dataframe) * 100)

    metadata_cols = ["label", "rpm", "file_id", "severity_type", "severity_value"]
    feature_cols = [col for col in dataframe.columns if col not in metadata_cols]
    assert feature_cols == FEATURE_COLUMNS, "Feature columns do not match FEATURE_COLUMNS."
    assert len(feature_cols) == 23, f"Expected 23 features, got {len(feature_cols)}"

    X = dataframe[feature_cols].values.astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    y = dataframe["label"].values
    groups = dataframe["file_id"].values
    rpm_values = dataframe["rpm"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_encoded, groups=groups))
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]
    rpm_test = rpm_values[test_idx]

    logger.info("Train/test split:")
    logger.info("  Train: %s windows from %s files", len(X_train), len(np.unique(groups[train_idx])))
    logger.info("  Test:  %s windows from %s files", len(X_test), len(np.unique(groups[test_idx])))

    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ]
    )

    logger.info("Training model.")
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["svm"]
    X_test_scaled = scaler.transform(X_test)

    y_pred = pipeline.predict(X_test)

    logger.info("%s", "=" * 70)
    logger.info("MODEL PERFORMANCE")
    logger.info("%s", "=" * 70)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    logger.info("Overall Accuracy:  %.4f", accuracy_score(y_test, y_pred))
    logger.info("Macro F1-Score:    %.4f", f1_score(y_test, y_pred, average="macro"))

    logger.info("%s", "=" * 70)
    logger.info("PHYSICS VALIDATIONS")
    logger.info("%s", "=" * 70)

    rpm_valid = validate_rpm_stratification(
        y_test,
        y_pred,
        rpm_test,
        RPM_RANGES,
        MIN_ACCEPTABLE_RPM_BAND_ACCURACY,
    )
    normal_false_alarm_rate = validate_normal_false_alarms(
        y_test,
        y_pred,
        class_names,
        MAX_ALLOWED_NORMAL_FALSE_ALARM,
    )

    if RUN_AXIS_ABLATION_TEST:
        X_train_scaled = scaler.transform(X_train)
        X_train_resampled, y_train_resampled = pipeline.named_steps["sampler"].fit_resample(X_train_scaled, y_train)
        physics_valid = run_axis_ablation_test_mafulda(
            X_train_resampled,
            X_test_scaled,
            y_train_resampled,
            y_test,
            feature_cols,
            class_names,
        )
    else:
        physics_valid = True

    if RUN_BEARING_FREQ_VALIDATION:
        freq_valid = validate_bearing_physics_mafulda(dataframe.iloc[test_idx].copy(), class_names)
    else:
        freq_valid = True

    if RUN_SEVERITY_VALIDATION:
        run_correct_severity_validation(dataframe.iloc[test_idx].copy(), class_names)

    if COMPUTE_FEATURE_IMPORTANCE:
        compute_permutation_feature_importance(
            classifier,
            X_test_scaled,
            y_test,
            feature_cols,
            class_names,
            save_path=FIGURES_DIR / "feature_importance_physics_validated.png",
        )

    if PLOT_CONFUSION_MATRIX:
        plot_pca_3d_interactive(X_scaled, y_encoded, rpm_values, class_names, label_encoder)
        _select_representative_rows(dataframe, feature_cols)
        plot_pca_with_rpm_coloring(X_scaled, y_encoded, rpm_values, class_names, label_encoder)

    all_valid = (
        rpm_valid
        and normal_false_alarm_rate <= MAX_ALLOWED_NORMAL_FALSE_ALARM
        and physics_valid
        and freq_valid
    )

    logger.info("%s", "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("%s", "=" * 70)
    if all_valid:
        logger.info("MODEL VALIDATED: LEARNING PHYSICS, NOT NOISE")
    else:
        logger.warning("Some validations failed. Check the logs.")

    saved_pipeline = {
        "scaler": scaler,
        "model": classifier,
        "label_encoder": label_encoder,
        "feature_names": feature_cols,
        "config": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "decimation_factor": DECIMATION_FACTOR,
        },
    }
    model_path = MODELS_DIR / "svm_pipeline_physics_validated.pkl"
    joblib.dump(saved_pipeline, model_path)
    logger.info("Model saved to %s", model_path)
    logger.info("Total runtime: %.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
