# pipeline_full_paperA.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.base import clone


# ---------------- MATLAB-accurate Quadratic SVM (SVM-Q) ----------------
from sklearn.svm import SVC
from sklearn.utils.validation import joblib

import numpy as np
from sklearn.svm import SVC

class MatlabSVMQ(SVC):
    """
    MATLAB-accurate Quadratic SVM:
    - Quadratic polynomial kernel
    - BoxConstraint = C = 1
    - KernelScale = auto → gamma = 1/(p * var(X))
    """
    def fit(self, X, y):
        # Compute MATLAB KernelScale (gamma)
        # MATLAB: gamma = 1 / ( num_features * var(X) )
        v = np.var(X, axis=0).mean() + 1e-12      # mean variance across dimensions
        p = X.shape[1]                            # num features
        gamma_matlab = 1.0 / (p * v)

        # Set kernel parameters EXACTLY like MATLAB
        self.gamma = gamma_matlab
        self.degree = 2          # quadratic
        self.coef0 = 1           # MATLAB uses (coef0 = 1)
        self.C = 1.0

        return super().fit(X, y)
# from utils.Matlab_SVC import MatlabSVMQ
# from utils.data_loader import section_dataset



healthy_Dict = {"fileName": [], "x": [], "label": 0, "predicted": []}
faulty_Dict = {"fileName": [], "x": [], "label": 1, "predicted": []}

def load_data_custom(file_path: str, directory: str):
    data = loadmat(f"../../data/paper_data/{directory}/{file_path}")
    return data


def section_dataset():
    healthy_dir = os.path.join(DATA_ROOT, "Healthy")
    faulty_dir = os.path.join(DATA_ROOT, "Faulty")
    for i in os.listdir(healthy_dir):
        healthy_Dict["fileName"].append(i)
        healthy_Dict["x"].append(load_data_custom(i, "Healthy")["H"].squeeze())
    for i in os.listdir(faulty_dir):
        faulty_Dict["fileName"].append(i)
        faulty_Dict["x"].append(load_data_custom(i, "Faulty")["H"].squeeze())
    return healthy_Dict, faulty_Dict

# ------------- User dataset configuration (edit paths if needed) -------------
DATA_ROOT = "../../data/paper_data"
HEALTHY_DIR = os.path.join(DATA_ROOT, "Healthy")
FAULTY_DIR  = os.path.join(DATA_ROOT, "Faulty")

RESULTS_DIR = "Results_pipeline_A"
os.makedirs(RESULTS_DIR, exist_ok=True)

FS = 1000                 # sampling frequency
WINDOW_LEN = 800          # window length in samples
WINDOW_STEP = 400         # 50% overlap -> step 400
MAX_IMFS = 10             # compute up to 10 IMFs (IMF1..IMF10), drop IMF1
N_FOLDS = 10




# ---------------- EMD helper: ensure up to MAX_IMFS, drop IMF1 ----------------
try:
    from utils.emd_processing import sig_to_imf as user_sig_to_imf
except Exception:
    user_sig_to_imf = None
    print("WARNING: EMD not found. Using PyEMD instead.")

def sig_to_imf_paper(sig, max_imfs=MAX_IMFS):
    """
    Ensure up to max_imfs IMFs are available (IMF1..IMFmax_imfs).
    Drop IMF1 and reconstruct using IMF2..IMFmax_imfs + residue.
    Return (imfs_filtered, residue, reconstructed)
    imfs_filtered has shape (max_imfs-1, len(sig)) if padding used.
    """
    if user_sig_to_imf is not None:
        try:
            imfs_f, residue, reconstructed = user_sig_to_imf(sig, max_imfs=max_imfs)
            # if user function returns reconstructed of correct length, trust it
            if reconstructed is not None and len(reconstructed) == len(sig):
                imfs_f = np.atleast_2d(imfs_f) if getattr(imfs_f, "size", 0) else np.array([])
                residue = residue if residue is not None else np.zeros_like(sig)
                return imfs_f, residue, reconstructed
        except Exception:
            pass

    # fallback using PyEMD
    from PyEMD import EMD
    emd = EMD()
    emd.emd(sig)
    imfs, residue = emd.get_imfs_and_residue()
    if imfs is None or (hasattr(imfs, "size") and imfs.size == 0):
        return np.array([]), residue if residue is not None else np.zeros_like(sig), sig.copy()
    imfs = np.atleast_2d(imfs)  # shape (n_imfs, L)
    # pad or truncate to exactly max_imfs (IMF1..IMFmax_imfs)
    n_have = imfs.shape[0]
    if n_have < max_imfs:
        pad_rows = np.zeros((max_imfs - n_have, imfs.shape[1]))
        imfs_padded = np.vstack([imfs, pad_rows])
    else:
        imfs_padded = imfs[:max_imfs]
    # drop IMF1 and sum IMF2..IMFmax_imfs
    if imfs_padded.shape[0] > 1:
        imfs_filtered = imfs_padded[1:max_imfs]  # IMF2..IMF10 -> up to 9 rows
        reconstructed = np.sum(imfs_filtered, axis=0) + (residue if residue is not None else 0.0)
    else:
        imfs_filtered = np.array([])
        reconstructed = residue if residue is not None else sig.copy()
    return imfs_filtered, residue if residue is not None else np.zeros_like(sig), reconstructed

# ---------------- windowing ----------------
def segment_signal(sig, window=WINDOW_LEN, step=WINDOW_STEP):
    segments = []
    starts = list(range(0, len(sig) - window + 1, step))
    for s in starts:
        segments.append(sig[s:s+window])
    return np.array(segments)  # shape (n_windows, window,)

# ---------------- feature functions (paper exact) ----------------
def temporal_features(sig):
    if sig.size == 0:
        return np.zeros(7)
    M = np.mean(sig)
    SD = np.std(sig)
    SK = float(skew(sig))
    KR = float(kurtosis(sig))
    PP = float(np.ptp(sig))
    RMS = float(np.sqrt(np.mean(sig**2)))
    E = float(np.sum(sig**2))
    return np.array([M, SD, SK, KR, PP, RMS, E], dtype=float)

def compute_psd(sig, fs=FS, nperseg=1024):
    nperseg = min(len(sig), nperseg)
    if nperseg < 8:
        return np.array([0.0]), np.array([np.sum(sig**2) + 1e-12])
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
    Pxx = np.maximum(Pxx, 1e-12)
    return f, Pxx

def freqA_features(sig, fs=FS):
    # FM, FSD, FSK, FKR, BPWR, FMED
    if sig.size == 0:
        return np.zeros(6)
    f, Pxx = compute_psd(sig, fs)
    totalE = np.sum(Pxx)
    FM = np.sum(f * Pxx) / (totalE + 1e-12)
    FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / (totalE + 1e-12))
    FSK = np.sum(((f - FM) ** 3) * Pxx) / ((totalE + 1e-12) * (FSD ** 3 + 1e-12))
    FKR = np.sum(((f - FM) ** 4) * Pxx) / ((totalE + 1e-12) * (FSD ** 4 + 1e-12))
    BPWR = totalE
    cumulative = np.cumsum(Pxx)
    median_idx = np.searchsorted(cumulative, 0.5 * totalE)
    FMED = float(f[min(median_idx, len(f) - 1)])
    return np.array([FM, FSD, FSK, FKR, BPWR, FMED], dtype=float)

def freqB_features(sig, fs=FS):
    # SC, SF, SRO, SFL, SCR, SDEC, SSL, SS
    if sig.size == 0:
        return np.zeros(8)
    f, Pxx = compute_psd(sig, fs)
    totalE = np.sum(Pxx)
    SC = np.sum(f * Pxx) / (totalE + 1e-12)
    Pnorm = Pxx / (np.sum(Pxx) + 1e-12)
    SF = float(np.sum((np.diff(Pnorm, prepend=Pnorm[0])) ** 2))
    cumulative = np.cumsum(Pxx)
    idx_roll = np.searchsorted(cumulative, 0.85 * totalE)
    SRO = float(f[min(idx_roll, len(f) - 1)])
    geo_mean = np.exp(np.mean(np.log(Pxx + 1e-12)))
    arith_mean = np.mean(Pxx)
    SFL = float(geo_mean / (arith_mean + 1e-12))
    SCR = float(np.max(Pxx) / (arith_mean + 1e-12))
    if len(Pxx) >= 2:
        n = np.arange(1, len(Pxx) + 1)
        diffs = (Pxx[:-1] - Pxx[1:]) / (n[:-1] + 1e-12)
        SDEC = float(np.sum(diffs) / (np.sum(Pxx) + 1e-12))
    else:
        SDEC = 0.0
    try:
        slope, _ = np.polyfit(f, Pxx, 1)
    except Exception:
        slope = 0.0
    SSL = float(slope)
    SS = float(np.sqrt(np.sum(((f - SC) ** 2) * Pxx) / (totalE + 1e-12)))
    return np.array([SC, SF, SRO, SFL, SCR, SDEC, SSL, SS], dtype=float)

# Compose feature sets for a single reconstructed window signal (composite S(t))
def compose_feature_sets_from_reconstructed(rec):
    t = temporal_features(rec)  # 7
    a = freqA_features(rec)     # 6
    b = freqB_features(rec)     # 8
    return {
        "F1": t,                     # 7
        "F2": a,                     # 6
        "F3": b,                     # 8
        "F4": np.concatenate((t, a)),           # 13
        "F5": np.concatenate((a, b)),           # 14
        "F6": np.concatenate((t, b)),           # 15
        "F7": np.concatenate((t, a, b)),        # 21
    }

# ---------------- Deterministic 10-fold file-level indices (round-robin) --------------
def deterministic_file_folds(n_files):
    folds = [[] for _ in range(N_FOLDS)]
    for i in range(n_files):
        folds[i % N_FOLDS].append(i)
    return folds

# ---------------- Build window-level dataset per file and mapping --------------------
def build_window_dataset(healthy, faulty):
    # outputs:
    # files = list of dicts with { 'file_name', 'label', 'window_features': list of feature dicts per set, 'index_in_global' }
    files = []
    # helper to process a file signal
    def process_file_signal(arr, label, fname, idx_global):
        x = arr[:,0]; y = arr[:,1]; z = arr[:,2]
        S = np.sqrt(x**2 + y**2 + z**2)
        S = S / (np.max(np.abs(S)) + 1e-12)
        segments = segment_signal(S, window=WINDOW_LEN, step=WINDOW_STEP)  # (n_windows, 800)
        # For each window compute EMD->reconstructed->features
        win_feature_sets = []
        for seg in segments:
            imfs_f, residue, rec = sig_to_imf_paper(seg, max_imfs=MAX_IMFS)
            sets = compose_feature_sets_from_reconstructed(rec)
            win_feature_sets.append(sets)
        return {
            "file_name": fname,
            "label": label,
            "window_features": win_feature_sets,
            "n_windows": len(win_feature_sets),
            "global_index": idx_global
        }
    idx = 0
    for i, fname in enumerate(healthy["fileName"]):
        arr = healthy["x"][i]
        files.append(process_file_signal(arr, 0, fname, idx)); idx += 1
    for i, fname in enumerate(faulty["fileName"]):
        arr = faulty["x"][i]
        files.append(process_file_signal(arr, 1, fname, idx)); idx += 1
    return files

# ---------------- Paper-style CV: train on window-level but test per-file majority vote --------------
def run_paper_cv(healthy, faulty):
    files = build_window_dataset(healthy, faulty)
    normal_count = len(healthy["fileName"])
    faulty_count = len(faulty["fileName"])
    # build fold indices for files separately then combine
    normal_folds = deterministic_file_folds(normal_count)
    faulty_folds = deterministic_file_folds(faulty_count)
    # map file global indices for normal and faulty
    normal_global = list(range(0, normal_count))
    faulty_global = list(range(normal_count, normal_count + faulty_count))
    normal_folds_global = [[normal_global[i] for i in fold] for fold in normal_folds]
    faulty_folds_global = [[faulty_global[i] for i in fold] for fold in faulty_folds]
    # models
    models = {
        "SVM-Q": MatlabSVMQ(kernel="poly"),
        "KNN-W": KNeighborsClassifier(weights="distance"),
        "LDA": LinearDiscriminantAnalysis(),
        "RF": RandomForestClassifier(random_state=0),
        "LGBM": LGBMClassifier(random_state=0),
    }
    records = []
    # For each feature set name produce X_windows, y_windows, and mapping windows->file idx
    # But we will build per-fold train/test sets dynamically to respect file-level folds
    for feat_set in ["F1","F2","F3","F4","F5","F6","F7"]:
        # Precompute per-file windows feature arrays for this feature set
        for model_name, base_model in models.items():
            fold_accs=[]; fold_precs=[]; fold_recs=[]; fold_f1s=[]
            total_TP = total_TN = total_FP = total_FN = 0
            for fold_idx in range(N_FOLDS):
                test_file_indices = normal_folds_global[fold_idx] + faulty_folds_global[fold_idx]
                train_file_indices = [i for i in range(len(files)) if i not in test_file_indices]
                # build train window-level dataset
                X_train_list=[]; y_train_list=[]
                file_index_of_test = []  # keep list of test file indices
                test_file_windows = {}   # map file_idx -> list of window feature vectors
                test_file_labels = {}    # map file_idx -> true label
                # train
                for fi in train_file_indices:
                    fobj = files[fi]
                    for wsets in fobj["window_features"]:
                        vec = wsets[feat_set]
                        X_train_list.append(vec)
                        y_train_list.append(fobj["label"])
                # test - keep windows grouped by file
                for fi in test_file_indices:
                    fobj = files[fi]
                    list_vecs = [wsets[feat_set] for wsets in fobj["window_features"]]
                    test_file_windows[fi] = np.vstack(list_vecs) if len(list_vecs)>0 else np.zeros((0, len(list_vecs[0]) if len(list_vecs)>0 else 0))
                    test_file_labels[fi] = fobj["label"]
                # convert to arrays
                X_train = np.vstack(X_train_list) if len(X_train_list)>0 else np.zeros((0,1))
                y_train = np.array(y_train_list, dtype=int)
                # scale per-fold
                if model_name == "SVM-Q":
                    scaler =  MinMaxScaler()
                else:
                    scaler = StandardScaler()

                X_train_s = scaler.fit_transform(X_train)
                # train model
                model = clone(base_model)
                model.fit(X_train_s, y_train)
                # predict windows per test file and majority vote per file
                y_true_files = []
                y_pred_files = []
                for fi in test_file_indices:
                    X_test_windows = test_file_windows[fi]  # shape (n_windows, feat_dim)
                    if X_test_windows.size == 0:
                        # fallback: if file had no windows (shouldn't happen), skip
                        continue
                    # scale
                    X_test_s = scaler.transform(X_test_windows)
                    # window predictions
                    if hasattr(model, "predict_proba"):
                        # use predicted class directly
                        y_win = model.predict(X_test_s)
                    else:
                        y_win = model.predict(X_test_s)
                    # majority vote (ties -> predict faulty (1) as conservative) or use mean threshold
                    counts = np.bincount(y_win.astype(int))
                    if len(counts) == 0:
                        pred_file = 0
                    elif len(counts) == 1:
                        pred_file = 0 if counts.shape[0]==1 and 0 in np.nonzero(counts)[0] else np.argmax(counts)
                    else:
                        if counts[0] == counts[1]:
                            pred_file = 1  # tie-break to faulty (paper not explicit — tie rare with 11 windows)
                        else:
                            pred_file = np.argmax(counts)
                    y_true_files.append(test_file_labels[fi])
                    y_pred_files.append(pred_file)
                # compute fold-level file metrics
                y_true_files = np.array(y_true_files, dtype=int)
                y_pred_files = np.array(y_pred_files, dtype=int)
                acc = accuracy_score(y_true_files, y_pred_files)
                prec = precision_score(y_true_files, y_pred_files, zero_division=0)
                rec = recall_score(y_true_files, y_pred_files, zero_division=0)
                f1 = f1_score(y_true_files, y_pred_files, zero_division=0)
                fold_accs.append(acc); fold_precs.append(prec); fold_recs.append(rec); fold_f1s.append(f1)
                cm = confusion_matrix(y_true_files, y_pred_files)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # degenerate (all same class)
                    tn = cm[0,0] if cm.shape==(1,1) else 0
                    fp = fn = tp = 0
                total_TP += int(tp); total_FP += int(fp); total_TN += int(tn); total_FN += int(fn)
                # save fold summary CSV
                fold_df = pd.DataFrame({
                    "FeatureSet":[feat_set],
                    "Model":[model_name],
                    "Fold":[fold_idx+1],
                    "Accuracy":[acc],
                    "Precision":[prec],
                    "Recall":[rec],
                    "F1":[f1],
                    "TP":[int(tp)],
                    "TN":[int(tn)],
                    "FP":[int(fp)],
                    "FN":[int(fn)]
                })
                fold_df.to_csv(os.path.join(RESULTS_DIR, f"{feat_set}_{model_name}_fold{fold_idx+1}.csv"), index=False)
                # save confusion matrix image
                disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Faulty"])
                disp.plot()
                plt.title(f"{feat_set} - {model_name} - Fold {fold_idx+1}")
                plt.savefig(os.path.join(RESULTS_DIR, f"{feat_set}_{model_name}_cm_fold{fold_idx+1}.png"))
                plt.close()
            # aggregated record across folds
            rec = {
                "FeatureSet": feat_set,
                "Model": model_name,
                "AvgAccuracy": float(np.mean(fold_accs)),
                "StdAccuracy": float(np.std(fold_accs)),
                "AvgPrecision": float(np.mean(fold_precs)),
                "AvgRecall": float(np.mean(fold_recs)),
                "AvgF1": float(np.mean(fold_f1s)),
                "TotalTP": int(total_TP),
                "TotalTN": int(total_TN),
                "TotalFP": int(total_FP),
                "TotalFN": int(total_FN),
                "NumFolds": N_FOLDS,
                "Samples": len(files)
            }
            records.append(rec)
            print(f"Done: {feat_set} - {model_name} | F1={rec['AvgF1']:.4f} Acc={rec['AvgAccuracy']:.4f}")
    df = pd.DataFrame(records)
    df = df.sort_values(by=["AvgF1","AvgAccuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "full_model_feature_comparison.csv"), index=False)
    joblib.dump(df, os.path.join(RESULTS_DIR, "full_model_feature_comparison.joblib"))
    return df

# ---------------- Main ----------------
if __name__ == "__main__":
    
    print("Loading dataset...")
    healthy, faulty = section_dataset()
    print(f"Loaded {len(healthy['x'])} healthy files and {len(faulty['x'])} faulty files.")
    ranking = run_paper_cv(healthy, faulty)
    print("Saved final ranking CSV to:", os.path.join(RESULTS_DIR, "full_model_feature_comparison.csv"))
    print(ranking.head(10))
