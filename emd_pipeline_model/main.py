
import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone # To reset the model in each fold
from scipy.signal import welch
from scipy.fft import fft, fftfreq

healthy_Dict, faulty_Dict = section_dataset()


x = faulty_Dict["x"][0][:, 0]
y = faulty_Dict["x"][0][:, 1]
z = faulty_Dict["x"][0][:, 2]

S = np.sqrt(x**2 + y**2 + z**2)


imfs = sig_to_imf(S)
MAX_IMFS = len(imfs)
# feature_dict = {
#     "time_features": time_features(imfs),
# }
  # 6 x 11


# def time_features(imfs: list):
#     len_imfs = len(imfs)
#     feature_Vector = np.zeros((9, len_imfs))
#     for idx, item in enumerate(imfs):

#         rms = np.sqrt(np.mean(item**2))

#         feature_Vector[0][idx] = skew(item) # skewness
#         feature_Vector[1][idx] = kurtosis(item) # kurtosis
#         feature_Vector[2][idx] = np.sum(np.abs(item) ** 2) # energy
#         feature_Vector[3][idx] = np.mean(item) # mean
#         feature_Vector[4][idx] = np.ptp(item) # peak to peak
#         feature_Vector[5][idx] = np.std(item) # std
#         feature_Vector[6][idx] = np.sqrt(np.mean(item**2)) # rms
#         feature_Vector[7][idx] = entropy(np.abs(item)) # entropy
#         feature_Vector[8][idx] =np.abs(np.max(item)/rms) # crest factor


#     return feature_Vector.flatten()




# def time_features(imfs: list):
#     len_imfs = len(imfs)
    
#     # Max possible features = 12 (you will enable the ones you need)
#     feature_vec = np.zeros((12, len_imfs))
    
#     for idx, x in enumerate(imfs):

#         # Preprocessing
#         rms = np.sqrt(np.mean(x ** 2))
#         p2p = np.max(x) - np.min(x)
#         mean_val = np.mean(x)
#         abs_mean = np.mean(np.abs(x))
#         std_val = np.std(x)
#         var_val = np.var(x)
#         skew_val = skew(x)
#         kurt_val = kurtosis(x)

#         # Advanced
#         crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
#         impulse_factor = np.max(np.abs(x)) / (abs_mean + 1e-12)
#         shape_factor = rms / (abs_mean + 1e-12)

#         # Hjorth parameters
#         dx = np.diff(x)
#         hj_mobility = np.sqrt(np.var(dx) / (var_val + 1e-12))
#         hj_complexity = np.sqrt((np.var(np.diff(dx)) + 1e-12) / (np.var(dx) + 1e-12))

#         # -----------------------------
#         #   TIER S — BEST FEATURES
#         # -----------------------------
#         feature_vec[0][idx] = rms
#         feature_vec[1][idx] = p2p
#         feature_vec[2][idx] = var_val
#         feature_vec[3][idx] = skew_val
#         feature_vec[4][idx] = kurt_val
#         feature_vec[5][idx] = crest_factor

#         # # -----------------------------
#         # #   TIER A — STRONG FEATURES
#         # # -----------------------------
#         feature_vec[6][idx] = impulse_factor
#         feature_vec[7][idx] = shape_factor
#         feature_vec[8][idx] = abs_mean

#         # -----------------------------
#         #   TIER B — OPTIONAL
#         # -----------------------------
#         feature_vec[9][idx] = std_val
#         feature_vec[10][idx] = hj_mobility
#         feature_vec[11][idx] = hj_complexity

#     return feature_vec.flatten()

# def spectral_features(imfs: list):
#     len_imfs = len(imfs)
    
#     # Max features = 12 (but best 8 recommended)
#     feature_vec = np.zeros((8, len_imfs))

#     for idx, item in enumerate(imfs):

#         # Welch Spectrum
#         f, Pxx = welch(item, fs=1000, nperseg=min(1024, len(item)))

#         # Basic statistics
#         total_energy = np.sum(Pxx)
#         C = np.sum(f * Pxx) / (total_energy + 1e-12)
#         gm = np.exp(np.mean(np.log(Pxx + 1e-10)))
#         am = np.mean(Pxx)

#         # Rolloff
#         cumulative_sum = np.cumsum(Pxx)
#         threshold = 0.85 * total_energy
#         rolloff_freq = f[np.where(cumulative_sum >= threshold)[0][0]]

#         # Extra stats
#         median_freq = np.median(f)
#         mean_freq = np.mean(f)
#         std_freq = np.std(f)
#         max_freq = np.max(f)

#         # -----------------------------
#         #   TIER S — BEST SPECTRAL FEATURES
#         # -----------------------------
#         feature_vec[0][idx] = total_energy            # Spectral Energy
#         feature_vec[1][idx] = C                       # Centroid
#         # feature_vec[2][idx] = np.sqrt(np.sum(((f - C) ** 2) * Pxx) / (total_energy + 1e-12))  # Bandwidth
#         feature_vec[2][idx] = rolloff_freq            # Rolloff
#         # feature_vec[3][idx] = gm / am                 # Flatness
#         feature_vec[3][idx] = np.max(Pxx)             # Spectral Max
#         feature_vec[4][idx] = kurtosis(Pxx)           # Spectral Kurtosis
#         # feature_vec[5][idx] = median_freq             # Median Frequency

#         # # -----------------------------
#         # #   TIER A — STRONG ADDITIONS
#         # # -----------------------------
#         # feature_vec[6][idx] = max_freq                # Peak frequency
#         feature_vec[7][idx] = mean_freq               # Mean frequency

#         # # -----------------------------
#         # #   TIER B — OPTIONAL
#         # # -----------------------------
#         feature_vec[8][idx] = std_freq               # Standard deviation
#         feature_vec[9][idx] = skew(f)                # Spectral skewness

#     return feature_vec.flatten()


# def hybrid_features(imfs: list):
#     len_imfs = len(imfs)

#     # Allocate enough space for all potential hybrid features.
#     # (You can comment/uncomment freely without changing array shape)
#     feature_Vector = np.zeros((25, len_imfs))  
#     # 25 = max number of hybrid features offered across all tiers

#     for idx, item in enumerate(imfs):

#         # -----------------------------
#         # Time-Domain Pre-Calculations
#         # -----------------------------
#         rms = np.sqrt(np.mean(item**2))
#         peak = np.max(np.abs(item))
#         p2p = np.ptp(item)
#         mean_val = np.mean(item)
#         std_val = np.std(item)
#         kurt = kurtosis(item)
#         cf = peak / (rms + 1e-10)
#         impulse_factor = peak / (mean_val + 1e-10)
#         shape_factor = rms / (mean_val + 1e-10)

#         # Envelope signal
#         analytic = hilbert(item)
#         envelope = np.abs(analytic)
#         env_rms = np.sqrt(np.mean(envelope**2))
#         env_kurt = kurtosis(envelope)
#         env_peak = np.max(envelope)
#         env_mean = np.mean(envelope)

#         # Sample entropy
#         try:
#             sampen = entropy(np.abs(item) + 1e-12, base=2)
#         except:
#             sampen = 0

#         # ------------------------------------
#         # Frequency Domain (Welch)
#         # ------------------------------------
#         f, Pxx = welch(item, fs=1000, nperseg=min(1024, len(item)))
#         total_energy = np.sum(Pxx)

#         C = np.sum(f * Pxx) / (total_energy + 1e-12)
#         gm = np.exp(np.mean(np.log(Pxx + 1e-10)))
#         am = np.mean(Pxx)

#         cumulative_sum = np.cumsum(Pxx)
#         rolloff = f[np.where(cumulative_sum >= 0.85 * total_energy)[0][0]]

#         spectral_kurt = kurtosis(Pxx)
#         spectral_flat = gm / (am + 1e-12)
#         spectral_max = np.max(Pxx)
#         median_freq = np.median(f)

#         # Band energies (example split)
#         low_band = np.sum(Pxx[(f >= 0) & (f < 200)])
#         mid_band = np.sum(Pxx[(f >= 200) & (f < 500)])
#         high_band = np.sum(Pxx[(f >= 500)])

#         # ------------------------------------------------------------
#         # --------- S-TIER HYBRID FEATURES (highest recall) ----------
#         # ------------------------------------------------------------

#         # 0 Envelope RMS
#         # feature_Vector[0][idx] = env_rms

#         # 1 Envelope Kurtosis
#         # feature_Vector[1][idx] = env_kurt

#         # 2 Envelope Peak
#         # feature_Vector[2][idx] = env_peak

#         # 3 High Frequency Band Energy
#         # feature_Vector[3][idx] = high_band

#         # 4 Spectral Kurtosis
#         # feature_Vector[4][idx] = spectral_kurt

#         # 5 Spectral Centroid
#         # feature_Vector[5][idx] = C

#         # 6 Max PSD amplitude
#         # feature_Vector[6][idx] = spectral_max

#         # 7 Time Kurtosis
#         # feature_Vector[7][idx] = kurt

#         # 8 Crest Factor
#         # feature_Vector[8][idx] = cf

#         # 9 Sample Entropy
#         # feature_Vector[9][idx] = sampen


#         # ------------------------------------------------------------
#         # --------- A-TIER HYBRID FEATURES (balanced) ----------------
#         # ------------------------------------------------------------

#         # 10 RMS
#         # feature_Vector[10][idx] = rms

#         # 11 Standard Deviation
#         # feature_Vector[11][idx] = std_val

#         # 12 Peak-to-Peak
#         # feature_Vector[12][idx] = p2p

#         # 13 Spectral Flatness
#         # feature_Vector[13][idx] = spectral_flat

#         # 14 Low Band Energy
#         # feature_Vector[14][idx] = low_band

#         # 15 Mid Band Energy
#         # feature_Vector[15][idx] = mid_band

#         # 16 Median Frequency
#         # feature_Vector[16][idx] = median_freq

#         # 17 Impulse Factor
#         # feature_Vector[17][idx] = impulse_factor

#         # 18 Shape Factor
#         # feature_Vector[18][idx] = shape_factor


#         # ------------------------------------------------------------
#         # --------- B-TIER HYBRID FEATURES (extra optional) ----------
#         # ------------------------------------------------------------

#         # 19 Envelope Mean
#         # feature_Vector[19][idx] = env_mean

#         # 20 Mean of Time Signal
#         # feature_Vector[20][idx] = mean_val

#         # 21 Frequency Rolloff
#         # feature_Vector[21][idx] = rolloff

#         # 22 Normalized High/Total Energy Ratio
#         # feature_Vector[22][idx] = high_band / (total_energy + 1e-12)

#         # 23 Autocorrelation Lag-1
#         # feature_Vector[23][idx] = np.correlate(item, item, mode='full')[len(item)-1] / len(item)

#         # 24 Dominant Frequency
#         # feature_Vector[24][idx] = f[np.argmax(Pxx)]

#     return feature_Vector.flatten()






# labaledColumns = pd.DataFrame(
#     feature_Vector,
#     index=["skew", "kurtosis", "energy", "mean", "p2p", "std"],
#     columns=[
#         "imf1",
#         "imf2",
#         "imf3",
#         "imf4",
#         "imf5",
#         "imf6",
#         "imf7",
#     ],
# )
# labaledColumns.to_csv("feature_fault.csv")


# print(len(healthy_Dict["x"]))







def extract_features(imfs_len: int, feature_len: int):
    feature_matrix_healthy = np.zeros((len(healthy_Dict["x"]), imfs_len * feature_len))
    feature_matrix_faulty = np.zeros((len(faulty_Dict["x"]), imfs_len * feature_len))
   # print(feature_matrix_healthy.shape)

    for idx, item in enumerate(healthy_Dict["x"]):
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature = time_features(imfs_signal).flatten()  # (66,)
        # print(feature)
        # print("----------------------")
    
        feature_matrix_healthy[idx] = feature

    for idx, item in enumerate(faulty_Dict["x"]):
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature = time_features(imfs_signal).flatten()  # (66,)
        feature_matrix_faulty[idx] = feature
    return feature_matrix_healthy, feature_matrix_faulty
    






# fig, axs = plt.subplots(11, figsize=(10, 10))

# axs[0].plot(S)
# axs[0].set_title("Original Vibration Signal S(t) magnitude H1.mat")

# axs[1].plot(healthy_Dict["x"][0])
# axs[1].set_title(healthy_Dict["fileName"][0])

# for idx, item in enumerate(imfs):
#     axs[idx].plot(imfs[idx])
#     axs[idx].set_title(f"IMF {idx+1}")

# plt.tight_layout()
# plt.show()


def apply_pipeline(model, X, threshold=0.5):
    try:
        # Models like SVM need probability=True
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]  # probability of Faulty
        else:
            # If model doesn't support predict_proba, fallback to normal prediction
            return model.predict(X)

        # Threshold decision: classify as Faulty if P(Faulty) >= threshold
        preds = (probs >= threshold).astype(int)
        return preds

    except:
        # Safety fallback
        return model.predict(X)






def cross_validate_model(model_type="lda", n_splits=5,
                                     imfs_len=7, feature_len=12,
                                     feature_names=None,
                                     faulty_threshold=0.4):

    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(feature_len)]

    feature_matrix_healthy, feature_matrix_faulty = extract_features(
        imfs_len=imfs_len, feature_len=feature_len
    )

    X = np.concatenate((feature_matrix_healthy, feature_matrix_faulty))
    Y = np.concatenate((
        np.zeros(feature_matrix_healthy.shape[0]),  # Healthy
        np.ones(feature_matrix_faulty.shape[0])    # Faulty
    ))

    if model_type == "svm":
        # probability=True needed for thresholding
        base_model = SVC(random_state=42, probability=True)
    elif model_type == "knn":
        base_model = KNeighborsClassifier()
    else:
        base_model = LinearDiscriminantAnalysis()

    results_dir = f"Results/{model_type}/thresholded"
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    # Metrics accumulators
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

        # Predict probabilities if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of 'Faulty'
            # Apply threshold
            y_pred = (y_prob >= faulty_threshold).astype(int)
        else:
            # fallback
            y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Accumulate
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        total_TP += tp
        total_TN += tn
        total_FP += fp
        total_FN += fn

        # Save fold CSV
        fold_df = pd.DataFrame({
            "Fold": [fold_idx + 1],
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F1": [f1],
            "TP": [tp],
            "TN": [tn],
            "FP": [fp],
            "FN": [fn]
        })
        fold_df.to_csv(f"{results_dir}/fold_{fold_idx+1}.csv", index=False)

        # Save fold confusion matrix image
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Faulty"])
        disp.plot()
        plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
        plt.savefig(f"{results_dir}/cm_fold_{fold_idx+1}.png")
        plt.close()

    # ---- SUMMARY ----
    summary_df = pd.DataFrame({
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
        "Total Samples": [len(Y)]
    })
    summary_df.to_csv(f"{results_dir}/summary.csv", index=False)

    overall_cm = np.array([[total_TN, total_FP], [total_FN, total_TP]])
    disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=["Healthy", "Faulty"])
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
            
