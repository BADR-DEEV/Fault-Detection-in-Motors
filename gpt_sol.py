"""
Vibration fault detection pipeline:
- Load .mat files (Healthy / Faulty)
- Compute features (time, freq, IMF)
- Train & evaluate KNN classifier
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
from PyEMD import EMD
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import csv
import joblib

# -------------------------
# 1) Config / paths
# -------------------------
DATA_ROOT = "Dataset"                 # change to your folder
HEALTHY_DIR = os.path.join(DATA_ROOT, "Healthy")
FAULTY_DIR  = os.path.join(DATA_ROOT, "Faulty")
MAT_VAR_NAME = None  # set to None to auto-detect (will use first array variable)
FS = 1000  # sampling frequency in Hz (change if different)

# -------------------------
# 2) helper functions
# -------------------------
def load_mat_file(path, mat_var_name=None):
    dd = loadmat(path)
    # auto-detect first ndarray of 1-2 dims (common)__version__
    if mat_var_name is None: 
        candidates = [k for k,v in dd.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        if len(candidates) == 0:
            raise ValueError(f"No ndarrays found in {path}")
        mat_var_name = candidates[0]
    arr = np.array(dd[mat_var_name])
    arr = arr.squeeze()
    # If shape is (3, N) transpose to (N,3)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] > 3:
        arr = arr.T
    # If 1D or other shapes, leave as-is
    return arr

def axis_signals_from_mat(arr):
    # Expect arr shape (N,3) or (N,) single axis
    if arr.ndim == 1:
        return arr, None, None
    if arr.ndim == 2 and arr.shape[1] >= 3:
        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]
        return x, y, z
    # fallback: if shape (3,N)
    if arr.ndim == 2 and arr.shape[0] >= 3:
        return arr[0,:], arr[1,:], arr[2,:]
    raise ValueError("Unexpected array shape: " + str(arr.shape))


# Time-domain features
def time_features(sig):
    sig = np.asarray(sig)
    rms = np.sqrt(np.mean(sig**2))
    meanv = np.mean(sig)
    stdv = np.std(sig)
    skewv = skew(sig)
    kurtv = kurtosis(sig)
    peak = np.max(np.abs(sig))
    p2p = np.ptp(sig)
    crest = peak / (rms + 1e-12)
    energy = np.sum(np.abs(sig)**2)
    # add new features

  
    return {"rms":rms, "mean":meanv, "std":stdv, "skew":skewv, "kurt":kurtv,
            "peak":peak, "p2p":p2p, "crest":crest, "energy":energy}

# # Frequency-domain features using PSD (Welch)
def spectral_features(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=min(4096, len(sig)))
    Pxx_norm = Pxx / (np.sum(Pxx) + 1e-12)
    # spectral centroid
    centroid = np.sum(f * Pxx_norm)
    # bandwidth (spectral spread)
    bw = np.sqrt(np.sum(((f - centroid)**2) * Pxx_norm))
    # spectral entropy
    spec_ent = entropy(Pxx_norm + 1e-12)
    # dominant frequency
    dom_f = f[np.argmax(Pxx)]
    # total band energy (sum of PSD)
    energy = np.sum(Pxx)
    return {"spec_centroid":centroid, "spec_bw":bw, "spec_entropy":spec_ent,
            "dom_freq":dom_f, "band_energy":energy}

# IMF features via EMD
def imf_features(sig, max_imfs=6):
    emd = EMD()
    imfs = emd.emd(sig)
    if imfs is None or imfs.size == 0:
        return {}
    

    energies = [np.sum(imf**2) for imf in imfs]

    
    total_e = np.sum(energies) + 1e-12
    energies_ratio = [e/total_e for e in energies]
    # keep up to max_imfs
    feats = {}
    for i in range(min(max_imfs, len(energies_ratio))):
        feats[f"imf{i+1}_energy"] = energies_ratio[i]
    # entropic measure of IMF energy distribution
    feats["imf_energy_entropy"] = entropy(np.array(energies_ratio) + 1e-12)
    # number of imfs
    feats["n_imfs"] = len(energies_ratio)
    return feats

# Combined feature extractor for one sample (3-axis array)
def extract_features_from_array(arr, fs=FS, use_combined=True):
    x,y,z = axis_signals_from_mat(arr)
    # decide which signal to use: combined magnitude or each axis
    features = {}
    signals_to_use = {}
    if use_combined:
        # if y or z is None, just use x
        if y is None:
            S = x
        else:
            if z is None:
                S = np.sqrt(x**2 + y**2)
            else:
                S = np.sqrt(x**2 + y**2 + z**2)
        signals_to_use["mag"] = S
    else:
        # per axis
        signals_to_use["x"] = x
        if y is not None: signals_to_use["y"] = y
        if z is not None: signals_to_use["z"] = z

    # compute features
    for name, sig in signals_to_use.items():
        tf = time_features(sig)
        sf = spectral_features(sig, fs)
        imff = imf_features(sig, max_imfs=6)
        # prefix with axis name
        for k,v in tf.items(): features[f"{name}_{k}"] = v
        for k,v in sf.items(): features[f"{name}_{k}"] = v
        for k,v in imff.items(): features[f"{name}_{k}"] = v
    return features

# -------------------------
# 3) Build dataset
# -------------------------
def build_dataset(healthy_dir, faulty_dir, mat_var=None, fs=FS, use_combined=True, max_files=None):
    rows = []
    labels = []
    # healthy
    healthy_files = sorted(glob.glob(os.path.join(healthy_dir, "*.mat")))
    faulty_files  = sorted(glob.glob(os.path.join(faulty_dir,  "*.mat")))
    if max_files:
        healthy_files = healthy_files[:max_files]
        faulty_files = faulty_files[:max_files]
    for p in healthy_files:
        arr = load_mat_file(p, mat_var)
        feats = extract_features_from_array(arr, fs=fs, use_combined=use_combined)
        feats["file"] = os.path.basename(p)
        rows.append(feats)
        labels.append(0)  # healthy = 0
    for p in faulty_files:
        arr = load_mat_file(p, mat_var)
        feats = extract_features_from_array(arr, fs=fs, use_combined=use_combined)
        feats["file"] = os.path.basename(p)
        rows.append(feats)
        labels.append(1)  # faulty = 1
    df = pd.DataFrame(rows).fillna(0)
    df["label"] = labels
    return df

# -------------------------
# 4) Train & Evaluate KNN
# -------------------------
def train_evaluate(df, save_model_path="knn_model.joblib"):
    X = df.drop(columns=["file","label"])
    y = df["label"].values
    feature_names = X.columns.tolist()
    X = X.values
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    # KNN + simple grid search to pick k
    param_grid = {"n_neighbors":[3,5,7,9], "weights":["uniform","distance"]}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_s, y_train)
    print("Best params:", grid.best_params_)
    best = grid.best_estimator_
    # Cross-val on train
    cv_scores = cross_val_score(best, X_train_s, y_train, cv=5)
    print("Train CV accuracy (5-fold):", np.round(cv_scores,3), "mean:", np.mean(cv_scores))
    # Test
    y_pred = best.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Faulty"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
    # Save model + scaler
    joblib.dump({"model":best, "scaler":scaler, "features":feature_names}, save_model_path)
    print("Saved model to", save_model_path)

# -------------------------
# 5) Run pipeline
# -------------------------



if __name__ == "__main__":
    print("Building dataset (this may take a while depending on EMD cost)...")
    df = build_dataset(HEALTHY_DIR, FAULTY_DIR, mat_var=MAT_VAR_NAME, fs=FS, use_combined=True, max_files=None)
    print("Dataset shape:", df.shape)
    print("Feature columns:", list(df.columns[:-1]))
    print("Label distribution:\n", df['label'].value_counts())
    # quick peek
    print(df.head())
    print(df.tail())
    df.to_csv("dataset.csv", index=False)


    healthy_col = df.loc[df['label'] == 0 ]
    faulty_col = df.loc[df['label'] == 1 ]
    # df = pd.concat([healthy_col, faulty_col], axis=0)
    healthy_col.describe()
    faulty_col.describe()
    print(healthy_col.describe())
    print(faulty_col.describe())

   
 
    # df.to_excel("dataset.xlsx", index=False)

    

    # Train
    train_evaluate(df, save_model_path="knn_vibration_model.joblib")




#90.9%