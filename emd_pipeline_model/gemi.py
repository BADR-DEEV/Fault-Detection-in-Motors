import os
from lightgbm import LGBMClassifier
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from emd_imfs import sig_to_imf
from load_data import section_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
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

healthy_Dict, faulty_Dict = section_dataset()


x = faulty_Dict["x"][0][:, 0]
y = faulty_Dict["x"][0][:, 1]
z = faulty_Dict["x"][0][:, 2]

S = np.sqrt(x**2 + y**2 + z**2)

imfs = sig_to_imf(S)
MAX_IMFS = len(imfs)

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


def time_features(imfs: list):
    len_imfs = len(imfs)
    feature_vec = np.zeros((8, len_imfs))

    for idx, x in enumerate(imfs):
        rms = np.sqrt(np.mean(x**2))  # not used
        p2p = np.max(x) - np.min(x)  # used
        std_val = np.std(x)  # not used
        skew_val = skew(x)  # used
        kurt_val = kurtosis(x)  # used
        abs_mean = np.mean(np.abs(x))  # used

        crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
        impulse_factor = np.max(np.abs(x)) / (abs_mean + 1e-12)

        feature_vec[:, idx] = [
            rms,
            p2p,
            std_val,
            skew_val,
            kurt_val,
            crest_factor,
            impulse_factor,
            abs_mean,
        ]

    return feature_vec.flatten()


def spectral_features(imfs):
    features = []

    prev_Pxx = None  # For spectral flux

    for item in imfs[:6]:
        # Welch spectrum
        f, Pxx = welch(item, fs=1000, nperseg=min(1024, len(item)))
        Pxx = np.maximum(Pxx, 1e-12)  # avoid log/ratio issues
        total_energy = np.sum(Pxx)

        # ===== CORE FREQUENCY FEATURES =====
        centroid = np.sum(f * Pxx) / total_energy

        std_freq = np.std(f)
        mean_freq = np.mean(f)
        median_freq = np.median(f)
        skew_freq = skew(f)
        kurt_freq = kurtosis(f)

        # roll-off at 85%
        cumulative = np.cumsum(Pxx)
        rolloff = f[np.searchsorted(cumulative, 0.85 * total_energy)]

        band_power = np.sum(Pxx[f > 0.5])

        # ===== SPECTRAL SHAPE FEATURES =====
        spectral_flatness = np.exp(np.mean(np.log(Pxx))) / (np.mean(Pxx) + 1e-12)

        spectral_crest = np.max(Pxx) / (np.mean(Pxx) + 1e-12)

        spectral_spread = np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / total_energy)

        # Slope (linear regression on spectrum)
        slope = (np.max(Pxx) - np.min(Pxx)) / (np.max(f) - np.min(f) + 1e-12)

        # Spectral decrease
        numerator = np.sum((Pxx[1:] - Pxx[0]) / np.arange(1, len(Pxx)))
        denominator = np.sum(Pxx[1:])
        spectral_decrease = numerator / (denominator + 1e-12)

        # ===== SPECTRAL FLUX =====
        if prev_Pxx is None:
            spectral_flux = 0  # no previous frame
        else:
            spectral_flux = np.sqrt(np.sum((Pxx - prev_Pxx) ** 2))
        prev_Pxx = Pxx

        # ===== COLLECT ALL FEATURES =====
        feature_vec = [
            # Requested + standard features
            mean_freq,  # Mean Frequency
            std_freq,  # Frequency Std
            skew_freq,  # Skewness
            kurt_freq,  # Kurtosis
            band_power,  # Band Power
            median_freq,  # Median Freq
            centroid,  # Spectral Centroid
            spectral_flux,  # Spectral Flux
            rolloff,  # Roll-off
            spectral_flatness,  # Flatness
            spectral_crest,  # Crest
            spectral_decrease,  # Decrease
            slope,  # Slope
            spectral_spread,  # Spread
            # Additional useful vibration features
            total_energy,
            np.max(Pxx),
            kurtosis(Pxx),
        ]

        features.append(feature_vec)

    return np.array(features).flatten()


def extract_features():
    features_healthy = []
    features_faulty = []

    for item in healthy_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)
        

        feature1 = time_features(imfs_signal)
        feature2 = spectral_features(imfs_signal)
        # feature3 = time_features([item_mag])  # wrap in list for time_features
        # feature4 = spectral_features([item_mag])

        feature = np.concatenate((feature1, feature2))
        features_healthy.append(feature)

    for item in faulty_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature1 = time_features(imfs_signal)
        feature2 = spectral_features(imfs_signal)
        # feature3 = time_features([item_mag])
        # feature4 = spectral_features([item_mag])

        feature = np.concatenate((feature1, feature2))
        features_faulty.append(feature)

    return np.array(features_healthy), np.array(features_faulty)



from scipy.stats import kurtosis, pearsonr

def get_top_imfs(imfs, original_signal, top_n=2):
    """
    Selects the top N IMFs based on Correlation to the original signal.
    (This is a standard method in MDPI papers).
    """
    correlations = []
    for imf in imfs:
        # Calculate correlation between this IMF and the original signal magnitude
        corr, _ = pearsonr(imf, original_signal)
        correlations.append(abs(corr))
    
    # Get indices of the top N IMFs with highest correlation
    top_indices = np.argsort(correlations)[-top_n:]
    
    # Return the selected IMFs
    return [imfs[i] for i in top_indices]

# MODIFIED EXTRACT FUNCTION
from scipy.stats import kurtosis, pearsonr

def get_top_imfs(imfs, original_signal, top_n=2):
    """
    Selects the top N IMFs based on Correlation to the original signal.
    (This is a standard method in MDPI papers).
    """
    correlations = []
    for imf in imfs:
        # Calculate correlation between this IMF and the original signal magnitude
        corr, _ = pearsonr(imf, original_signal)
        correlations.append(abs(corr))
    
    # Get indices of the top N IMFs with highest correlation
    top_indices = np.argsort(correlations)[-top_n:]
    
    # Return the selected IMFs
    return [imfs[i] for i in top_indices]

# MODIFIED EXTRACT FUNCTION
def extract_features():
    features_healthy = []
    features_faulty = []

    # Process Healthy
    for item in healthy_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)
        
        # --- FIX: SELECT ONLY BEST IMFs ---
        selected_imfs = get_top_imfs(imfs_signal, item_mag, top_n=2) 
        # Now we always have exactly 2 IMFs, so feature vector size is fixed.
        
        # Pass only selected IMFs to feature extraction
        feature1 = time_features(selected_imfs) 
        feature2 = spectral_features(selected_imfs) # Remove the [:6] limit inside the function
        
        feature = np.concatenate((feature1, feature2))
        features_healthy.append(feature)

    # Process Faulty (Repeat same steps)
    for item in faulty_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)
        
        selected_imfs = get_top_imfs(imfs_signal, item_mag, top_n=2)
        
        feature1 = time_features(selected_imfs)
        feature2 = spectral_features(selected_imfs)
        
        feature = np.concatenate((feature1, feature2))
        features_faulty.append(feature)

    return np.array(features_healthy), np.array(features_faulty)


def cross_validate_model(
    model_type="svm", n_splits=5, imfs_len=MAX_IMFS, feature_len=7, faulty_threshold=0.4
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


cross_validate_model()


# def train_evaluate(model:str, save_model_path:str):
#     feature_matrix_healthy, feature_matrix_faulty = extract_features(imfs_len=MAX_IMFS, feature_len=6)
#     concatenated_matrix = np.concatenate((feature_matrix_healthy, feature_matrix_faulty))
#     healthy_labels = np.zeros(feature_matrix_healthy.shape[0])
#     faulty_labels = np.ones(feature_matrix_faulty.shape[0])
#     concatenated_labels = np.concatenate((healthy_labels, faulty_labels))
#     print(concatenated_labels.shape)
#     print(concatenated_matrix.shape)
#     X_train, X_test, y_train, y_test = train_test_split(
#             concatenated_matrix,
#             concatenated_labels,
#             test_size=0.4,
#             random_state=42,
#             shuffle=True
#         )
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     if model == "knn":
#         knn = KNeighborsClassifier()
#         knn.fit(X_train, y_train)
#         y_pred = knn.predict(X_test)
#     elif model == "svm":
#         svm = SVC()
#         svm.fit(X_train, y_train)
#         y_pred = svm.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Faulty"])
#     disp.plot()
#     plt.title("Confusion Matrix")
#     plt.show()

# train_evaluate("svm", "svm_model.joblib")