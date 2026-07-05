import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .config import DATA_SOURCES, DEVICE, MAX_FILES_PER_CLASS, RANDOM_STATE, TEST_SIZE
from .dataset import SpectralDataset
from .paths import ARTIFACTS_DIR, DATA_ROOT, FIGURES_DIR, MODELS_DIR
from .plotting import plot_confusion_matrix
from .preprocessing import SpectralOrderPreprocessor
from .training import train_pytorch_model


def collect_dataset_files(base_path: Path):
    all_files = []
    all_labels = []
    for class_name, config in DATA_SOURCES.items():
        target_dir = base_path / config["root"]
        if not target_dir.exists():
            continue
        if "subfolders" in config:
            for subfolder in config["subfolders"]:
                sub_dir = target_dir / subfolder
                if sub_dir.exists():
                    files = list(sub_dir.rglob("*.csv"))
                    all_files.extend(files)
                    all_labels.extend([class_name] * len(files))
        else:
            files = list(target_dir.rglob("*.csv"))
            all_files.extend(files)
            all_labels.extend([class_name] * len(files))
    return all_files, all_labels


def main(data_root: Path = DATA_ROOT):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_files, all_labels = collect_dataset_files(data_root)
    if not all_files:
        raise FileNotFoundError(f"Dataset not found at {data_root}")

    preprocessor = SpectralOrderPreprocessor()
    X, y, sources = preprocessor.fit_transform(all_files, all_labels, max_files_per_class=MAX_FILES_PER_CLASS)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(splitter.split(X, y_encoded, groups=sources))

    X_train_val = X[train_val_idx]
    y_train_val = y_encoded[train_val_idx]
    sources_train_val = sources[train_val_idx]
    X_test = X[test_idx]
    y_test = y_encoded[test_idx]

    best_model = train_pytorch_model(X_train_val, y_train_val, sources_train_val, label_encoder, FIGURES_DIR, MODELS_DIR)

    best_model.eval()
    test_loader = DataLoader(SpectralDataset(X_test, y_test, augment=False), batch_size=128, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels_numeric = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            outputs = best_model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs.extend(probabilities.cpu().numpy())
            all_preds.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
            all_labels_numeric.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels_numeric = np.array(all_labels_numeric)
    class_names = label_encoder.classes_

    plot_confusion_matrix(all_labels_numeric, all_preds, class_names, FIGURES_DIR)

    report_dict = classification_report(
        all_labels_numeric,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(ARTIFACTS_DIR / "classification_report.csv")

    prediction_frame = pd.DataFrame(
        {
            "True_Label": label_encoder.inverse_transform(all_labels_numeric),
            "Pred_Label": label_encoder.inverse_transform(all_preds),
            "Correct": all_labels_numeric == all_preds,
        }
    )
    for index, class_name in enumerate(class_names):
        prediction_frame[f"Prob_{class_name}"] = all_probs[:, index]
    prediction_frame.to_csv(ARTIFACTS_DIR / "test_predictions_detailed.csv", index=False)

    test_acc = accuracy_score(all_labels_numeric, all_preds)
    try:
        test_auc = roc_auc_score(all_labels_numeric, all_probs, multi_class="ovr")
    except Exception:
        test_auc = test_acc

    with open(ARTIFACTS_DIR / "final_metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump({"Test_Accuracy": float(test_acc), "Test_ROC_AUC": float(test_auc)}, file_handle, indent=4)
