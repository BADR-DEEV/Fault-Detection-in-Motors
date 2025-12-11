"""
pipeline_paper_exact.py

Paper-exact pipeline (F1..F7) for the MDPI Sensors 2021 dataset (sampling freq = 1000 Hz).
Saves per-fold CSVs, confusion matrices, and a final ranking CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier

# --- Replace with your real data loader ---
from load_data import section_dataset  # must return healthy_Dict, faulty_Dict

# --- Try to import user's fixed sig_to_imf if available ---
try:
    from emd_imfs import sig_to_imf  # expected signature: (imfs_filtered, residue, reconstructed)
except Exception:
    # fallback implementation using PyEMD (will drop IMF1)
    def sig_to_imf(sig, max_imfs=10):
        from PyEMD import EMD
        emd = EMD()
        emd.emd(sig)
        imfs, residue = emd.get_imfs_and_residue()
        if imfs is None or (hasattr(imfs, "size") and imfs.size == 0):
            return np.array([]), residue if residue is not None else np.zeros_like(sig), sig.copy()
        imfs = np.atleast_2d(imfs)[:max_imfs]  # imfs[0] = IMF1, etc.
        # Drop IMF1 and reconstruct IMF2.. + residue
        if imfs.shape[0] > 1:
            imfs_filtered = imfs[1:]  # IMF2..
            reconstructed = np.sum(imfs_filtered, axis=0) + (residue if residue is not None else 0.0)
        else:
            imfs_filtered = np.array([])
            reconstructed = residue if residue is not None else sig.copy()
        return imfs_filtered, residue if residue is not None else np.zeros_like(sig), reconstructed

# ---------------- CONFIG ----------------
FS = 1000  # sampling frequency (Hz) — per dataset / sensor spec
MAX_IMFS = 9  # IMF2..IMF10 -> up to 9 IMFs
RESULTS_DIR = "Results/paper_exact"
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 10

# ---------------- Feature implementations (paper definitions) ----------------

# --- Temporal features (7): M, SD, SK, KR, PP, RMS, E ---
def temporal_features(signal):
    """Return 7 temporal features in the paper order:
       Mean (M), Std (SD), Skew (SK), Kurt (KR), PeakToPeak (PP), RMS, Energy (E)
    """
    if signal.size == 0:
        return np.zeros(7, dtype=float)
    M = np.mean(signal)
    SD = np.std(signal)
    SK = float(skew(signal))
    KR = float(kurtosis(signal))
    PP = float(np.ptp(signal))
    RMS = float(np.sqrt(np.mean(signal ** 2)))
    E = float(np.sum(signal ** 2))
    return np.array([M, SD, SK, KR, PP, RMS, E], dtype=float)


# --- Frequency-domain helpers (use Welch PSD) ---
def compute_psd(signal, fs=FS, nperseg=1024):
    """Return f, Pxx using Welch; safe for short signals."""
    nperseg = min(nperseg, max(8, len(signal)))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    Pxx = np.maximum(Pxx, 1e-12)
    return f, Pxx


# --- Frequency A (6): FM, FSD, FSK, FKR, BPWR, FMED ---
def freqA_features(signal, fs=FS):
    """Frequency statistical features computed from PSD."""
    if signal.size == 0:
        return np.zeros(6, dtype=float)
    f, Pxx = compute_psd(signal, fs=fs)
    total_energy = np.sum(Pxx)
    # mean frequency (FM)
    FM = np.sum(f * Pxx) / (total_energy + 1e-12)
    # frequency std (FSD)
    FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / (total_energy + 1e-12))
    # frequency skewness (FSK)
    FSK = np.sum(((f - FM) ** 3) * Pxx) / ((total_energy + 1e-12) * (FSD ** 3 + 1e-12))
    # frequency kurtosis (FKR)
    FKR = np.sum(((f - FM) ** 4) * Pxx) / ((total_energy + 1e-12) * (FSD ** 4 + 1e-12))
    # band power BPWR - paper uses band 0..Nyquist; we'll compute total band power (0..fs/2)
    BPWR = total_energy
    # median frequency FMED
    cumulative = np.cumsum(Pxx)
    median_idx = np.searchsorted(cumulative, 0.5 * total_energy)
    FMED = float(f[min(median_idx, len(f) - 1)])
    return np.array([FM, FSD, FSK, FKR, BPWR, FMED], dtype=float)


# --- Frequency B (8): SC, SF, SRO, SFL, SCR, SDEC, SSL, SS ---
def freqB_features(signal, fs=FS):
    """Spectral features set (8 features) used in the paper."""
    if signal.size == 0:
        return np.zeros(8, dtype=float)
    f, Pxx = compute_psd(signal, fs=fs)
    total_energy = np.sum(Pxx)
    # Spectral centroid SC (same as mean_freq weighted by Pxx)
    SC = np.sum(f * Pxx) / (total_energy + 1e-12)
    # Spectral flux SF (sum squared difference between successive normalized magnitude spectra)
    # use normalized Pxx (divide by sum) then compute diff
    Pnorm = Pxx / (np.sum(Pxx) + 1e-12)
    SF = float(np.sum((np.diff(Pnorm, prepend=Pnorm[0])) ** 2))
    # Spectral roll-off SRO (85% rolloff)
    cumulative = np.cumsum(Pxx)
    idx_roll = np.searchsorted(cumulative, 0.85 * total_energy)
    SRO = float(f[min(idx_roll, len(f) - 1)])
    # Spectral flatness SFL (geometric mean / arithmetic mean)
    geo_mean = np.exp(np.mean(np.log(Pxx + 1e-12)))
    arith_mean = np.mean(Pxx)
    SFL = float(geo_mean / (arith_mean + 1e-12))
    # Spectral crest SCR (max / mean)
    SCR = float(np.max(Pxx) / (arith_mean + 1e-12))
    # Spectral decay / decrease SDEC (Ton & Pant approx used earlier)
    if len(Pxx) >= 2:
        n = np.arange(1, len(Pxx) + 1)
        diffs = (Pxx[:-1] - Pxx[1:]) / (n[:-1] + 1e-12)
        SDEC = float(np.sum(diffs) / (np.sum(Pxx) + 1e-12))
    else:
        SDEC = 0.0
    # Spectral slope SSL (linear fit slope of Pxx vs frequency)
    try:
        slope, intercept = np.polyfit(f, Pxx, 1)
    except Exception:
        slope = 0.0
    SSL = float(slope)
    # Spectral spread SS (variance around centroid)
    SS = float(np.sqrt(np.sum(((f - SC) ** 2) * Pxx) / (total_energy + 1e-12)))
    return np.array([SC, SF, SRO, SFL, SCR, SDEC, SSL, SS], dtype=float)


# ------------------ Feature set compositions (paper exact) ------------------
# F1 = temporal 7
# F2 = freqA 6
# F3 = freqB 8
# F4 = F1 + F2 = 13
# F5 = F2 + F3 = 14
# F6 = F1 + F3 = 15
# F7 = F1 + F2 + F3 = 21

ALL_FEATURE_SETS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]

def compose_features_from_reconstructed(reconstructed):
    t = temporal_features(reconstructed)      # length 7
    a = freqA_features(reconstructed)         # length 6
    b = freqB_features(reconstructed)         # length 8
    sets = {
        "F1": t,
        "F2": a,
        "F3": b,
        "F4": np.concatenate((t, a)),
        "F5": np.concatenate((a, b)),
        "F6": np.concatenate((t, b)),
        "F7": np.concatenate((t, a, b)),
    }
    return sets


# ------------------ Extraction over dataset ------------------
def extract_all_feature_sets(healthy_Dict, faulty_Dict, fs=FS, max_imfs=MAX_IMFS):
    """
    Returns dictionary: { 'F1': (X,y), 'F2': (X,y), ... }
    """
    features_by_set = {k: [] for k in ALL_FEATURE_SETS}
    labels_by_set = {k: [] for k in ALL_FEATURE_SETS}

    def process_sample(arr):
        # arr shape (N_samples, 3)
        x = arr[:, 0]; yv = arr[:, 1]; z = arr[:, 2]
        S = np.sqrt(x ** 2 + yv ** 2 + z ** 2)
        S = S / (np.max(np.abs(S)) + 1e-12)  # per-sample normalization
        # EMD decomposition and reconstruction must have dropped IMF1 already in sig_to_imf
        imfs_filtered, residue, reconstructed = sig_to_imf(S, max_imfs=max_imfs)
        # ensure reconstructed is np array same length
        if reconstructed is None:
            reconstructed = S.copy()
        sets = compose_features_from_reconstructed(reconstructed)
        for k, v in sets.items():
            features_by_set[k].append(v)

    # process healthy
    for item in healthy_Dict["x"]:
        process_sample(item)
        for k in ALL_FEATURE_SETS:
            labels_by_set[k].append(0)
    # process faulty
    for item in faulty_Dict["x"]:
        process_sample(item)
        for k in ALL_FEATURE_SETS:
            labels_by_set[k].append(1)

    outputs = {}
    for k in ALL_FEATURE_SETS:
        feats = features_by_set[k]
        # pad to same length if required (should be equal per set because each sample returns same length for a set)
        maxlen = max(f.shape[0] for f in feats)
        X = np.vstack([np.pad(f, (0, maxlen - f.shape[0]), 'constant') if f.shape[0] < maxlen else f for f in feats])
        y = np.array(labels_by_set[k], dtype=int)
        outputs[k] = (X, y)
    return outputs


# ------------------ Evaluation: CV + classifiers (paper params) ------------------
def evaluate_feature_sets(
    healthy_Dict,
    faulty_Dict,
    models=None,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    results_dir=RESULTS_DIR,
):
    if models is None:
        models = {
            # SVM-Quadratic as paper: C=1 (default), degree=2, gamma='scale', no probability=True
            "SVM-Q": SVC(kernel="poly", degree=2, gamma="scale", C=1.0, probability=False, random_state=random_state),
            "KNN-W": KNeighborsClassifier(weights="distance"),
            "LDA": LinearDiscriminantAnalysis(),
            "RF": RandomForestClassifier(random_state=random_state),
            "LGBM": LGBMClassifier(random_state=random_state),
        }

    outputs = extract_all_feature_sets(healthy_Dict, faulty_Dict)
    records = []

    os.makedirs(results_dir, exist_ok=True)

    for fs_name, (X, y) in outputs.items():
        for model_name, base_model in models.items():

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scaler = StandardScaler()

            accs = []; precs = []; recs = []; f1s = []
            total_TP = total_FP = total_TN = total_FN = 0

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # standardize per-fold (paper uses z-score)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = clone(base_model)
                model.fit(X_train, y_train)

                # get predictions
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    # paper uses classifier decision, not calibrated probability; threshold 0.5 is fine
                    y_pred = (y_prob >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

                cm = confusion_matrix(y_test, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = cm[0, 0]; fp = fn = tp = 0

                total_TP += tp; total_FP += fp; total_TN += tn; total_FN += fn

                # save per-fold CSV
                fold_df = pd.DataFrame({
                    "FeatureSet": [fs_name],
                    "Model": [model_name],
                    "Fold": [fold_idx],
                    "Accuracy": [acc],
                    "Precision": [prec],
                    "Recall": [rec],
                    "F1": [f1],
                    "TP": [tp],
                    "TN": [tn],
                    "FP": [fp],
                    "FN": [fn]
                })
                fold_path = os.path.join(results_dir, f"{fs_name}_{model_name}_fold{fold_idx}.csv")
                fold_df.to_csv(fold_path, index=False)

                # confusion matrix image
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Faulty"])
                disp.plot()
                plt.title(f"{fs_name} - {model_name} - Fold {fold_idx}")
                plt.savefig(os.path.join(results_dir, f"{fs_name}_{model_name}_cm_fold{fold_idx}.png"))
                plt.close()

            rec = {
                "FeatureSet": fs_name,
                "Model": model_name,
                "AvgAccuracy": float(np.mean(accs)),
                "StdAccuracy": float(np.std(accs)),
                "AvgPrecision": float(np.mean(precs)),
                "AvgRecall": float(np.mean(recs)),
                "AvgF1": float(np.mean(f1s)),
                "TotalTP": int(total_TP),
                "TotalTN": int(total_TN),
                "TotalFP": int(total_FP),
                "TotalFN": int(total_FN),
                "NumFolds": n_splits,
                "Samples": int(len(y)),
            }
            records.append(rec)
            # save intermediate CSV for this (optional)
            # pd.DataFrame([rec]).to_csv(os.path.join(results_dir, f"summary_{fs_name}_{model_name}.csv"), index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["AvgF1", "AvgAccuracy"], ascending=False).reset_index(drop=True)
    out_path = os.path.join(results_dir, "full_model_feature_comparison.csv")
    df.to_csv(out_path, index=False)
    return df


# ------------------ Main ------------------
if __name__ == "__main__":
    healthy_Dict, faulty_Dict = section_dataset()
    ranking = evaluate_feature_sets(healthy_Dict, faulty_Dict)
    print(f"Saved final ranking CSV to: {os.path.join(RESULTS_DIR, 'full_model_feature_comparison.csv')}")
    print(ranking.head(10))
