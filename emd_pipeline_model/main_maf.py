import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from PyEMD import EMD

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = "../Mafulda_Dataset/"

SELECT_FAULTS = [
    "normal",
    "horizontal-misalignment/0.5mm",
    "imbalance/6g",
    # "overhang/ball_fault/6g"
]

# Map each class to an integer
label_map = {fault: idx for idx, fault in enumerate(SELECT_FAULTS)}

# Feature names (for reference)
feature_names = [
    "RMS", "P2P", "Var", "Skew", "Kurt", "Crest", "Impulse",
    "TotalEnergy", "Centroid", "Rolloff", "MaxPxx", "SpectralKurtosis", "MedianFreq", "StdFreq"
]

# -----------------------------
# Helper functions
# -----------------------------
def sig_to_imf(sig):
    emd = EMD()
    imfs = emd.emd(sig)

    if imfs is None or imfs.size == 0:
        return np.zeros((1, len(sig)))  # fallback

    imfs = np.atleast_2d(imfs).squeeze()
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    
    return imfs[:7]  # return first 7 IMFs (or fewer)

def time_features(imfs):
    feature_vec = np.zeros((7, len(imfs)))
    for idx, x in enumerate(imfs):
        rms = np.sqrt(np.mean(x**2))
        p2p = np.max(x) - np.min(x)
        var_val = np.var(x)
        skew_val = skew(x)
        kurt_val = kurtosis(x)
        abs_mean = np.mean(np.abs(x))
        crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
        impulse_factor = np.max(np.abs(x)) / (abs_mean + 1e-12)
        feature_vec[:, idx] = [rms, p2p, var_val, skew_val, kurt_val, crest_factor, impulse_factor]
    return feature_vec.flatten()

def spectral_features(imfs):
    feature_vec = np.zeros((7, len(imfs)))
    for idx, item in enumerate(imfs):
        f, Pxx = welch(item, fs=1000, nperseg=min(1024, len(item)))
        total_energy = np.sum(Pxx)
        centroid = np.sum(f * Pxx) / (total_energy + 1e-12)
        cumulative = np.cumsum(Pxx)
        rolloff = f[np.where(cumulative >= 0.85 * total_energy)[0][0]]
        spectral_kurt = kurtosis(Pxx)
        median_freq = np.median(f)
        std_freq = np.std(f)
        feature_vec[:, idx] = [total_energy, centroid, rolloff, np.max(Pxx), spectral_kurt, median_freq, std_freq]
    return feature_vec.flatten()

# -----------------------------
# Load data and extract features
# -----------------------------
X_features = []
y_labels = []

for fault in SELECT_FAULTS:
    folder_path = os.path.join(ROOT_DIR, fault)
    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(folder_path, file), header=None)
        df.columns = ['rot_freq','uhang_x','uhang_y','uhang_z',
                      'ohang_x','ohang_y','ohang_z','microphone']
        
        # Choose which signal to use for EMD
        signal = np.sqrt(df[['uhang_x','uhang_y','uhang_z']].values**2).sum(axis=1)
        imfs = sig_to_imf(signal)
        
        time_f = time_features(imfs)
        spec_f = spectral_features(imfs)
        X_features.append(np.concatenate((time_f, spec_f)))
        y_labels.append(label_map[fault])

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print("Number of samples:", len(X_features))
print("Number of features per sample:", X_features.shape[1])
print("Classes:", label_map)

# -----------------------------
# Cross-validation multiclass
# -----------------------------
def cross_validate_multiclass(model_type="svm", n_splits=5):
    if model_type == "rf":
        base_model = RandomForestClassifier(random_state=42)
    elif model_type == "svm":
        base_model = SVC(probability=True, random_state=42)
    elif model_type == "knn":
        base_model = KNeighborsClassifier()
    else:
        raise ValueError("Invalid model type")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    overall_cm = np.zeros((len(SELECT_FAULTS), len(SELECT_FAULTS)), dtype=int)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = clone(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(SELECT_FAULTS))))
        overall_cm += cm

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SELECT_FAULTS)
        disp.plot()
        plt.title(f"Fold {fold_idx+1} Confusion Matrix")
        plt.show()

    print("Average Accuracy:", np.mean(acc_list))
    print("Average Precision:", np.mean(prec_list))
    print("Average Recall:", np.mean(rec_list))
    print("Average F1:", np.mean(f1_list))

    disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=SELECT_FAULTS)
    disp.plot()
    plt.title("Overall Confusion Matrix")
    plt.show()

# -----------------------------
# Run cross-validation
# -----------------------------
cross_validate_multiclass(model_type="svm")
