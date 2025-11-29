
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
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
#         feature_Vector[0][idx] = skew(item)
#         feature_Vector[1][idx] = kurtosis(item)
#         # energy
#         feature_Vector[2][idx] = np.sum(np.abs(item) ** 2)
#         feature_Vector[3][idx] = np.mean(item)
#         feature_Vector[4][idx] = np.ptp(item)
#         feature_Vector[5][idx] = np.std(item)
#         #rms 
#         feature_Vector[6][idx] = np.sqrt(np.mean(item**2))
#         #entropy
#         feature_Vector[7][idx] = entropy(np.abs(item))
#         rms = np.sqrt(np.mean(item**2))
#         feature_Vector[8][idx] =np.abs(np.max(item)/rms)


#     return feature_Vector.flatten()


def spectral_features(imfs: list):
    len_imfs = len(imfs)
    feature_Vector = np.zeros((6, len_imfs))
    for idx, item in enumerate(imfs):
        f, Pxx = welch(item, fs=1000, nperseg=5000)
        feature_Vector[0][idx] = np.sum(Pxx)
        feature_Vector[1][idx] = np.mean(item)
        feature_Vector[2][idx] = np.std(item)
        feature_Vector[3][idx] = skew(item)
        feature_Vector[4][idx] = kurtosis(item)
        feature_Vector[5][idx] = np.median(item)
        #band power 


    return feature_Vector.flatten()

        # feature_Vector[idx][6] = np.max(item)



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
#         "imf8",
#         "imf9",
#         "imf10",
#         "imf11",
#     ],
# )
# labaledColumns.to_csv("feature_fault.csv")


print(len(healthy_Dict["x"]))







def extract_features(imfs_len: int, feature_len: int):
    feature_matrix_healthy = np.zeros((len(healthy_Dict["x"]), imfs_len * feature_len))
    feature_matrix_faulty = np.zeros((len(faulty_Dict["x"]), imfs_len * feature_len))
    print(feature_matrix_healthy.shape)

    for idx, item in enumerate(healthy_Dict["x"]):
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature = spectral_features(imfs_signal).flatten()  # (66,)
        # print(feature)
        # print("----------------------")
    
        feature_matrix_healthy[idx] = feature

    for idx, item in enumerate(faulty_Dict["x"]):
        x, y, z = item[:, 0], item[:, 1], item[:, 2]
        item_mag = np.sqrt(x**2 + y**2 + z**2)
        imfs_signal = sig_to_imf(item_mag)

        feature = spectral_features(imfs_signal).flatten()  # (66,)
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







def cross_validate_model(model_type="lda", n_splits=5):
    feature_matrix_healthy, feature_matrix_faulty = extract_features(imfs_len=7, feature_len=6)
    X = np.concatenate((feature_matrix_healthy, feature_matrix_faulty))
    healthy_labels = np.zeros(feature_matrix_healthy.shape[0])
    faulty_labels = np.ones(feature_matrix_faulty.shape[0])
    Y = np.concatenate((healthy_labels, faulty_labels))

    if model_type == "svm":
        base_model = SVC(random_state=42)
    elif model_type == "knn":
        base_model = KNeighborsClassifier()
    else :
        base_model = LinearDiscriminantAnalysis ()
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    accuracy_scores = []
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = clone(base_model)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)
        print(f"Fold {fold_idx+1} Accuracy: {acc:.4f}")

    print("\n--- Cross-Validation Summary ---")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} +/- {np.std(accuracy_scores):.4f}")
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
            
