import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from scipy import signal, stats

# ================= CONFIG =================
WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 5

SAMPLING_FREQ_RAW = 20000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

VIBRATION_COLS = [1,2,3]

MODEL_PATH = r"C:\DEV\personal\Ai-Driven-Vibrations-motor\src\training_cleaned\i_hope.pkl"
DATA_ROOT = r"C:\DEV\Personal\Ai-Driven-Vibrations-motor\data\CbM-Datasets-main\SampleMotorDataLimerick\SpectraQuest_Rig_Data_Voyager_3\Data_ADXL356C"

# ================= FIXED PARSER =================
def parse_limerick_filename(filename):
    base = os.path.basename(filename).replace('.Wfm.csv','').replace('.csv','')
    parts = base.split('_')

    if len(parts) < 6:
        return None, None

    try:
        rpm = int(parts[0])
    except:
        return None, None

    fault = parts[1]
    shaft = parts[2]
    load = parts[3]
    alignment = parts[4]

    # -----------------------------
    # RELAXED + PHYSICS MAPPING
    # -----------------------------

    # NORMAL
    if fault == 'GoB' and shaft == 'GS':
        return rpm, 'Normal'

    # IMBALANCE (keep all loads except extreme)
    if fault == 'GoB' and shaft == 'GS' and load in ['VLIL','LImL','HlmL']:
        return rpm, 'Imbalance'

    # BALL FAULT (keep ALL loads)
    if fault in ['LBF','HBF'] and shaft == 'GS':
        return rpm, 'Ball_Fault'

    # OUTER RACE (keep ALL loads)
    if fault in ['LOR','HOR'] and shaft == 'GS':
        return rpm, 'Outer_Race'

    return None, None


# ================= FEATURE EXTRACTION =================
def extract_features(vib, fs):
    features = []
    rms_vals = []

    for ax in range(3):
        sig = vib[:, ax]
        sig = np.nan_to_num(sig)

        rms = np.sqrt(np.mean(sig**2)) + 1e-9
        kur = stats.kurtosis(sig, fisher=False)
        crest = np.max(np.abs(sig)) / rms

        rms_vals.append(rms)

        f, Pxx = signal.welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        Pxx = np.maximum(Pxx, 1e-12)

        totalE = np.sum(Pxx)

        if totalE == 0:
            features.extend([rms, kur, crest, 0, 0, 0, 0])
            continue

        FM = np.sum(f * Pxx) / totalE
        FSD = np.sqrt(np.sum((f - FM)**2 * Pxx) / totalE)

        cum = np.cumsum(Pxx)
        FMED = f[np.searchsorted(cum, 0.5 * totalE)]
        SRO = f[np.searchsorted(cum, 0.85 * totalE)]

        features.extend([rms, kur, crest, FM, FSD, FMED, SRO])

    # ratios
    r1, r2, r3 = rms_vals
    features.extend([
        r1 / (r2 + r3 + 1e-12),
        r2 / (r1 + r3 + 1e-12)
    ])

    return features


def process_file(path, rpm):
    try:
        df = pd.read_csv(path, sep=';', header=None, usecols=[0,1,2,3], engine='python')
        df = df.astype(float).dropna()

        if len(df) < WINDOW_SIZE:
            return []

        vib = df.values[:, VIBRATION_COLS]
        vib = signal.decimate(vib, DECIMATION_FACTOR, axis=0, zero_phase=True)

        feats = []

        for i in range(0, len(vib) - WINDOW_SIZE + 1, STRIDE):
            win = vib[i:i+WINDOW_SIZE]
            f = extract_features(win, SAMPLING_FREQ_DECIMATED)

            if f:
                feats.append(f)

        return feats

    except:
        return []


# ================= PCA =================
def plot_pca(X, y_true, y_pred):
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)

    correct = y_true == y_pred

    plt.figure(figsize=(10,7))
    plt.scatter(Xp[correct,0], Xp[correct,1], label='Correct', alpha=0.6)
    plt.scatter(Xp[~correct,0], Xp[~correct,1], marker='x', label='Wrong')

    plt.legend()
    plt.title("PCA Feature Space")
    plt.show()


# ================= MAIN =================
def main():

    data = joblib.load(MODEL_PATH)
    scaler = data['scaler']
    model = data['model']
    le = data['label_encoder']

    files = glob.glob(os.path.join(DATA_ROOT, "*.csv"))

    X_all, y_all = [], []

    for f in files:
        rpm, cls = parse_limerick_filename(f)

        if cls is None or cls not in le.classes_:
            continue

        feats = process_file(f, rpm)

        for ft in feats:
            X_all.append(ft)
            y_all.append(cls)

    X = np.array(X_all)
    y_true = np.array(y_all)

    print(f"Samples: {len(X)}")

    # ✅ IMPORTANT: USE SAME SCALER AS TRAINING
    X_scaled = scaler.transform(X)

    y_pred = le.inverse_transform(model.predict(X_scaled))

    acc = accuracy_score(y_true, y_pred)

    print(f"\n🏆 Accuracy: {acc:.2%}")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.show()

    # PCA
    plot_pca(X_scaled, y_true, y_pred)


if __name__ == "__main__":
    main()