# Updated pipeline: EMD extraction (drop IMF1), feature extraction, CV
# - Uses PyEMD correctly (emd.emd() then get_imfs_and_residue())
# - Drops IMF1, keeps IMF2..IMF10 (up to available IMFs)
# - Reconstructs signal = sum(IMF2..IMF10) + residue
# - Extracts time + spectral features from IMFs and reconstructed signal

import os
from lightgbm import LGBMClassifier
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
# from emd_imfs import sig_to_imf  # replaced by local implementation below
from emd_imfs import sig_to_imf
from load_data import section_dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone  # To reset the model in each fold
from scipy.signal import welch
from scipy.fft import fft, fftfreq


# ------------------ Load data ------------------
healthy_Dict, faulty_Dict = section_dataset()

# quick sanity check plot for one sample (optional)
x = faulty_Dict["x"][0][:, 0]
y = faulty_Dict["x"][0][:, 1]
z = faulty_Dict["x"][0][:, 2]
S = np.sqrt(x**2 + y**2 + z**2)

imfs, residue, reconstructed = sig_to_imf(S, max_imfs=10)

num_imfs = imfs.shape[0]
fig, axs = plt.subplots(num_imfs + 3, figsize=(12, 2 * (num_imfs + 3)))
axs[0].plot(S)
axs[0].set_title("Original Vibration Signal")
for i in range(num_imfs):
    axs[i + 1].plot(imfs[i])
    axs[i + 1].set_title(f"IMF {i + 2}")  # IMFs returned are IMF2.. so label accordingly
axs[num_imfs + 1].plot(residue)
axs[num_imfs + 1].set_title("Residue")
axs[num_imfs + 2].plot(reconstructed)
axs[num_imfs + 2].set_title("Reconstructed (IMF2.. + residue)")
plt.tight_layout()
plt.show()

# Set MAX_IMFS to number of IMFs used (IMF2..)
MAX_IMFS = num_imfs

# ------------------ Feature names ------------------
feature_names = [
    "RMS",
    "P2P",
    "Var",
    "Skew",
    "Kurt",
    "Crest",
    "Impulse",
    "TotalEnergy",
    "Centroid",
    "Rolloff",
    "MaxPxx",
    "SpectralKurtosis",
    "MedianFreq",
    "StdFreq",
]

# ------------------ Feature extractors ------------------

def safe_len(imfs):
    return 0 if imfs is None else (imfs.shape[0] if hasattr(imfs, 'shape') else len(imfs))


def time_features(imfs: list):
    # imfs: iterable of 1D arrays
    if len(imfs) == 0:
        return np.zeros(0)

    len_imfs = len(imfs)
    feature_vec = np.zeros((7, len_imfs))

    for idx, x in enumerate(imfs):
        rms = np.sqrt(np.mean(x**2))
        p2p = np.max(x) - np.min(x)
        var_val = np.var(x)
        skew_val = skew(x)
        kurt_val = kurtosis(x)
        abs_mean = np.mean(np.abs(x))

        crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
        impulse_factor = np.max(np.abs(x)) / (abs_mean + 1e-12)

        feature_vec[:, idx] = [
            rms,
            p2p,
            var_val,
            skew_val,
            kurt_val,
            crest_factor,
            impulse_factor,
        ]

    return feature_vec.flatten()


def spectral_features(imfs: list, fs=1000):
    if len(imfs) == 0:
        return np.zeros(0)

    len_imfs = len(imfs)
    feature_vec = np.zeros((7, len_imfs))

    for idx, item in enumerate(imfs):
        # ensure length > 0
        if item.size == 0:
            f = np.array([0.0])
            Pxx = np.array([1e-12])
        else:
            f, Pxx = welch(item, fs=fs, nperseg=min(1024, len(item)))
            Pxx = np.maximum(Pxx, 1e-12)

        total_energy = np.sum(Pxx)
        centroid = np.sum(f * Pxx) / (total_energy + 1e-12)

        cumulative = np.cumsum(Pxx)
        idx_roll = np.searchsorted(cumulative, 0.85 * total_energy)
        idx_roll = min(idx_roll, len(f) - 1)
        rolloff = f[idx_roll]

        spectral_kurt = kurtosis(Pxx)
        median_freq = np.median(f)
        std_freq = np.std(f)

        feature_vec[:, idx] = [
            total_energy,
            centroid,
            rolloff,
            np.max(Pxx),
            spectral_kurt,
            median_freq,
            std_freq,
        ]

    return feature_vec.flatten()


# ------------------ Extraction updated: use IMF2.. + reconstructed ------------------

def extract_features():
    features_healthy = []
    features_faulty = []

    for item in healthy_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)

        imfs_signal, residue, reconstructed = sig_to_imf(item_mag, max_imfs=10)

        # If no IMFs returned, use reconstructed only
        imfs_list = [imfs_signal[i] for i in range(imfs_signal.shape[0])] if imfs_signal.size > 0 else []

        # Features from IMFs (IMF2..IMF10)
        # feature1 = time_features(imfs_list)
        # feature2 = spectral_features(imfs_list)

        # Also extract features from reconstructed (preprocessed) signal
        feature3 = time_features([reconstructed])
        feature4 = spectral_features([reconstructed])

        feature = np.concatenate((feature3, feature4))
        features_healthy.append(feature)

    for item in faulty_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)

        imfs_signal, residue, reconstructed = sig_to_imf(item_mag, max_imfs=10)
        imfs_list = [imfs_signal[i] for i in range(imfs_signal.shape[0])] if imfs_signal.size > 0 else []

        # feature1 = time_features(imfs_list[:8])
        # feature2 = spectral_features(imfs_list)
        feature3 = time_features([reconstructed])
        feature4 = spectral_features([reconstructed])

        feature = np.concatenate(( feature3, feature4))
        features_faulty.append(feature)

    # Pad feature vectors to same length if some samples had fewer IMFs
    max_len = max([f.shape[0] for f in features_healthy + features_faulty])
    def pad_array(a, length):
        if a.shape[0] == length:
            return a
        padded = np.zeros(length)
        padded[: a.shape[0]] = a
        return padded

    features_healthy = np.array([pad_array(f, max_len) for f in features_healthy])
    features_faulty = np.array([pad_array(f, max_len) for f in features_faulty])

    return features_healthy, features_faulty


# ------------------ Cross validation (unchanged logic, updated paths) ------------------

def cross_validate_model(
    model_type="svm", n_splits=10, imfs_len=MAX_IMFS, feature_len=7, faulty_threshold=0.4
):

    feature_matrix_healthy, feature_matrix_faulty = extract_features()

    X = np.concatenate((feature_matrix_healthy, feature_matrix_faulty))
    Y = np.concatenate(
        (
            np.zeros(feature_matrix_healthy.shape[0]),  # Healthy
            np.ones(feature_matrix_faulty.shape[0]),  # Faulty
        )
    )

    if model_type == "svm":
        base_model = SVC(random_state=42, probability=True)
    elif model_type == "knn":
        base_model = KNeighborsClassifier()
    elif model_type == "lgbm":
        base_model = LGBMClassifier()
    elif model_type == "rf":
        base_model = RandomForestClassifier()
    else:
        base_model = LinearDiscriminantAnalysis()

    results_dir = f"Results/marwan/{model_type}/H1"
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = MinMaxScaler()

    total_TP = total_TN = total_FP = total_FN = 0
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = clone(base_model)
        model.fit(X_train_scaled, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of 'Faulty'
            y_pred = (y_prob >= faulty_threshold).astype(int)
        else:
            y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        total_TP += tp
        total_TN += tn
        total_FP += fp
        total_FN += fn

        fold_df = pd.DataFrame(
            {
                "Fold": [fold_idx + 1],
                "Accuracy": [acc],
                "Precision": [prec],
                "Recall": [rec],
                "F1": [f1],
                "TP": [tp],
                "TN": [tn],
                "FP": [fp],
                "FN": [fn],
            }
        )
        fold_df.to_csv(f"{results_dir}/fold_{fold_idx+1}.csv", index=False)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Healthy", "Faulty"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
        plt.savefig(f"{results_dir}/cm_fold_{fold_idx+1}.png")
        plt.close()

    summary_df = pd.DataFrame(
        {
            "Model Type": [model_type],
            "Faulty Threshold": [faulty_threshold],
            "Average Accuracy": [np.mean(accuracy_scores)],
            "Std Accuracy": [np.std(accuracy_scores)],
            "Average Precision": [np.mean(precision_scores)],
            "Average Recall": [np.mean(recall_scores)],
            "Average F1": [np.mean(f1_scores)],
            "Total TP": [total_TP],
            "Total TN": [total_TN],
            "Total FP": [total_FP],
            "Total FN": [total_FN],
            "IMFs Length": [imfs_len],
            "Features per IMF": [feature_len],
            "Total Features per Sample": [imfs_len * feature_len],
            "Feature Names": [", ".join(feature_names)],
            "Num Folds": [n_splits],
            "Total Samples": [len(Y)],
        }
    )
    summary_df.to_csv(f"{results_dir}/summary.csv", index=False)

    overall_cm = np.array([[total_TN, total_FP], [total_FN, total_TP]])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=overall_cm, display_labels=["Healthy", "Faulty"]
    )
    disp.plot()
    plt.title(f"Overall Confusion Matrix - {model_type} Thresholded")
    plt.savefig(f"{results_dir}/cm_overall.png")
    plt.close()


if __name__ == "__main__":
    cross_validate_model()
