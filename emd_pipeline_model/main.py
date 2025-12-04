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
from sklearn.preprocessing import StandardScaler
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


def spectral_features(imfs: list):
    len_imfs = len(imfs)
    feature_vec = np.zeros((7, len_imfs))

    for idx, item in enumerate(imfs):
        f, Pxx = welch(item, fs=1000, nperseg=min(1024, len(item)))

        total_energy = np.sum(Pxx)
        centroid = np.sum(f * Pxx) / (total_energy + 1e-12)

        cumulative = np.cumsum(Pxx)
        rolloff = f[np.where(cumulative >= 0.85 * total_energy)[0][0]]

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

def extract_features():
    features_healthy = []
    features_faulty = []

    for item in healthy_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature1 = time_features(imfs_signal)
        feature2 = spectral_features(imfs_signal)
        feature3 = time_features([item_mag])  # wrap in list for time_features

        feature = np.concatenate((feature1, feature2, feature3))
        features_healthy.append(feature)

    for item in faulty_Dict["x"]:
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature1 = time_features(imfs_signal)
        feature2 = spectral_features(imfs_signal)
        feature3 = time_features([item_mag])


        feature = np.concatenate((feature1, feature2, feature3))
        features_faulty.append(feature)

    return np.array(features_healthy), np.array(features_faulty)



def apply_pipeline(model, X, threshold=0.5):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]  # probability of Faulty
        else:
            return model.predict(X)

        preds = (probs >= threshold).astype(int)
        return preds

    except:
        return model.predict(X)


def cross_validate_model(
    model_type="rf", n_splits=5, imfs_len=7, feature_len=7, faulty_threshold=0.4
):

    feature_matrix_healthy, feature_matrix_faulty = extract_features(
    )

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

    results_dir = f"Results/test_enviorment/{model_type}/H1"
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

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
