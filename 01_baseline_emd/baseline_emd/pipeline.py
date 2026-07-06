from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import N_FOLDS
from .data import deterministic_file_folds, section_dataset
from .features import compose_feature_sets_from_reconstructed, segment_signal, sig_to_imf_paper
from .models import build_models
from .paths import ARTIFACTS_DIR, DATA_ROOT, FIGURES_DIR


CONFUSION_DIR = FIGURES_DIR / "confusion_matrices"


def build_window_dataset(healthy, faulty):
    files = []

    def process_file_signal(array, label, file_name, global_index):
        x_axis = array[:, 0]
        y_axis = array[:, 1]
        z_axis = array[:, 2]
        magnitude = np.sqrt(x_axis**2 + y_axis**2 + z_axis**2)
        magnitude = magnitude / (np.max(np.abs(magnitude)) + 1e-12)
        segments = segment_signal(magnitude)
        window_features = []
        for segment in segments:
            _, _, reconstructed = sig_to_imf_paper(segment)
            window_features.append(compose_feature_sets_from_reconstructed(reconstructed))
        return {
            "file_name": file_name,
            "label": label,
            "window_features": window_features,
            "n_windows": len(window_features),
            "global_index": global_index,
        }

    global_index = 0
    for index, file_name in enumerate(healthy["fileName"]):
        files.append(process_file_signal(healthy["x"][index], 0, file_name, global_index))
        global_index += 1

    for index, file_name in enumerate(faulty["fileName"]):
        files.append(process_file_signal(faulty["x"][index], 1, file_name, global_index))
        global_index += 1

    return files


def run_paper_cv(healthy, faulty):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    files = build_window_dataset(healthy, faulty)
    normal_count = len(healthy["fileName"])
    faulty_count = len(faulty["fileName"])

    normal_folds = deterministic_file_folds(normal_count, N_FOLDS)
    faulty_folds = deterministic_file_folds(faulty_count, N_FOLDS)
    normal_global = list(range(0, normal_count))
    faulty_global = list(range(normal_count, normal_count + faulty_count))
    normal_folds_global = [[normal_global[i] for i in fold] for fold in normal_folds]
    faulty_folds_global = [[faulty_global[i] for i in fold] for fold in faulty_folds]

    records = []
    models = build_models()

    for feature_set in ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]:
        for model_name, base_model in models.items():
            fold_accs = []
            fold_precs = []
            fold_recs = []
            fold_f1s = []
            total_tp = total_tn = total_fp = total_fn = 0

            for fold_index in range(N_FOLDS):
                test_file_indices = normal_folds_global[fold_index] + faulty_folds_global[fold_index]
                train_file_indices = [index for index in range(len(files)) if index not in test_file_indices]
                train_vectors = []
                train_labels = []
                test_file_windows = {}
                test_file_labels = {}

                for file_index in train_file_indices:
                    file_object = files[file_index]
                    for window_sets in file_object["window_features"]:
                        train_vectors.append(window_sets[feature_set])
                        train_labels.append(file_object["label"])

                for file_index in test_file_indices:
                    file_object = files[file_index]
                    window_vectors = [window_sets[feature_set] for window_sets in file_object["window_features"]]
                    test_file_windows[file_index] = np.vstack(window_vectors) if window_vectors else np.zeros((0, 1))
                    test_file_labels[file_index] = file_object["label"]

                X_train = np.vstack(train_vectors) if train_vectors else np.zeros((0, 1))
                y_train = np.array(train_labels, dtype=int)
                scaler = MinMaxScaler() if model_name == "SVM-Q" else StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                model = clone(base_model)
                model.fit(X_train_scaled, y_train)

                y_true_files = []
                y_pred_files = []
                for file_index in test_file_indices:
                    X_test_windows = test_file_windows[file_index]
                    if X_test_windows.size == 0:
                        continue
                    y_window = model.predict(scaler.transform(X_test_windows))
                    counts = np.bincount(y_window.astype(int))
                    if len(counts) == 0:
                        prediction = 0
                    elif len(counts) == 1:
                        prediction = int(np.argmax(counts))
                    elif counts[0] == counts[1]:
                        prediction = 1
                    else:
                        prediction = int(np.argmax(counts))
                    y_true_files.append(test_file_labels[file_index])
                    y_pred_files.append(prediction)

                y_true_files = np.array(y_true_files, dtype=int)
                y_pred_files = np.array(y_pred_files, dtype=int)
                acc = accuracy_score(y_true_files, y_pred_files)
                prec = precision_score(y_true_files, y_pred_files, zero_division=0)
                rec = recall_score(y_true_files, y_pred_files, zero_division=0)
                f1 = f1_score(y_true_files, y_pred_files, zero_division=0)
                fold_accs.append(acc)
                fold_precs.append(prec)
                fold_recs.append(rec)
                fold_f1s.append(f1)

                cm = confusion_matrix(y_true_files, y_pred_files)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = cm[0, 0] if cm.shape == (1, 1) else 0
                    fp = fn = tp = 0

                total_tp += int(tp)
                total_fp += int(fp)
                total_tn += int(tn)
                total_fn += int(fn)

                fold_df = pd.DataFrame(
                    {
                        "FeatureSet": [feature_set],
                        "Model": [model_name],
                        "Fold": [fold_index + 1],
                        "Accuracy": [acc],
                        "Precision": [prec],
                        "Recall": [rec],
                        "F1": [f1],
                        "TP": [int(tp)],
                        "TN": [int(tn)],
                        "FP": [int(fp)],
                        "FN": [int(fn)],
                    }
                )
                fold_df.to_csv(ARTIFACTS_DIR / f"{feature_set}_{model_name}_fold{fold_index + 1}.csv", index=False)

                disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Faulty"])
                disp.plot()
                plt.title(f"{feature_set} - {model_name} - Fold {fold_index + 1}")
                plt.savefig(CONFUSION_DIR / f"{feature_set}_{model_name}_cm_fold{fold_index + 1}.png")
                plt.close()

            record = {
                "FeatureSet": feature_set,
                "Model": model_name,
                "AvgAccuracy": float(np.mean(fold_accs)),
                "StdAccuracy": float(np.std(fold_accs)),
                "AvgPrecision": float(np.mean(fold_precs)),
                "AvgRecall": float(np.mean(fold_recs)),
                "AvgF1": float(np.mean(fold_f1s)),
                "TotalTP": int(total_tp),
                "TotalTN": int(total_tn),
                "TotalFP": int(total_fp),
                "TotalFN": int(total_fn),
                "NumFolds": N_FOLDS,
                "Samples": len(files),
            }
            records.append(record)
            print(f"Done: {feature_set} - {model_name} | F1={record['AvgF1']:.4f} Acc={record['AvgAccuracy']:.4f}")

    dataframe = pd.DataFrame(records).sort_values(by=["AvgF1", "AvgAccuracy"], ascending=False).reset_index(drop=True)
    dataframe.to_csv(ARTIFACTS_DIR / "full_model_feature_comparison.csv", index=False)
    joblib.dump(dataframe, ARTIFACTS_DIR / "full_model_feature_comparison.joblib")
    return dataframe


def main(data_root: Path = DATA_ROOT):
    healthy, faulty = section_dataset(data_root)
    print(f"Loaded {len(healthy['x'])} healthy files and {len(faulty['x'])} faulty files.")
    ranking = run_paper_cv(healthy, faulty)
    print(f"Saved final ranking CSV to: {ARTIFACTS_DIR / 'full_model_feature_comparison.csv'}")
    print(ranking.head(10))
