import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
#               CONFIGURATION
# ==========================================
DATA_PATH = '../../data/VBL-VA001/'              # VBL training folder
NEW_FILE_PATH = '../../data/BMP_dataset/New40gNormal.txt'  # Your test file

ORIGINAL_FS = 20000      # VBL original sampling rate
TARGET_FS = 4000         # Target rate after processing
DECIMATION_FACTOR = ORIGINAL_FS // TARGET_FS

CATEGORIES = {
    'normal': 0,
    'bearing': 1,
    'unbalance': 2,
    'misalignment': 3
}
REVERSE_CATEGORIES = {v: k.capitalize() for k, v in CATEGORIES.items()}

# ==========================================
#          FEATURE EXTRACTION (SHARED)
# ==========================================
def extract_custom_features(x, fs):
    """Extracts 12 time/frequency features from a single axis signal."""
    N = len(x)
    if N == 0:
        return np.zeros(12)
    
    eps = 1e-12
    mean_val = np.mean(x)
    std_val = np.std(x, ddof=1) if N > 1 else 0
    rms_val = np.sqrt(np.mean(x**2))
    peak_val = np.max(np.abs(x))
    kurt_val = kurtosis(x, fisher=False)
    skew_val = skew(x)
    
    mean_abs = np.mean(np.abs(x))
    crest_factor = peak_val / (rms_val + eps)
    shape_factor = rms_val / (mean_abs + eps)
    impulse_factor = peak_val / (mean_abs + eps)
    clearance_factor = peak_val / ((np.mean(np.sqrt(np.abs(x))))**2 + eps)
    
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(x))
    fft_vals[0] = 0  # Remove DC
    
    if np.sum(fft_vals) > eps:
        peak_freq = freqs[np.argmax(fft_vals)]
        mean_freq = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    else:
        peak_freq = mean_freq = 0.0
        
    feats = [mean_val, std_val, rms_val, peak_val, kurt_val, skew_val, 
             crest_factor, shape_factor, impulse_factor, clearance_factor,
             peak_freq, mean_freq]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
#          DATA LOADING FUNCTIONS
# ==========================================
def load_vbl_training_data():
    """Loads ONLY VBL dataset files for training."""
    X, y = [], []
    tasks = []
    
    for category, label in CATEGORIES.items():
        folder = os.path.join(DATA_PATH, category)
        if not os.path.exists(folder):
            print(f"⚠️  Skipped missing folder: {folder}")
            continue
        files = glob.glob(os.path.join(folder, '*.csv'))
        print(f"📁 Found {len(files)} files in '{category}'")
        for f in files:
            tasks.append((f, TARGET_FS, DECIMATION_FACTOR))
            y.append(label)
            
    if not tasks:
        raise FileNotFoundError("❌ No VBL files found. Check DATA_PATH.")
        
    print(f"⚙️  Extracting features from {len(tasks)} VBL files...")
    start = time.time()
    for filepath, target_fs, dec_factor in tasks:
        try:
            df = pd.read_csv(filepath, usecols=[1, 2, 3], header=None)
            data = df.values.astype(float)
            feats = []
            for axis in range(3):
                x = data[:, axis]
                if dec_factor > 1 and len(x) > dec_factor * 10:
                    x = decimate(x, dec_factor, zero_phase=True)
                feats.extend(extract_custom_features(x, target_fs))
            if len(feats) == 36:
                X.append(feats)
        except Exception as e:
            print(f"⚠️  Failed to process {filepath}: {e}")
            
    elapsed = time.time() - start
    print(f"✅ VBL feature extraction completed in {elapsed:.1f}s")
    return np.array(X), np.array(y)


def process_new_test_file(filepath, target_fs=TARGET_FS):
    """Processes ONLY the new TXT file for testing. NO training leakage."""
    print(f"📖 Loading new test file: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, sep=',', on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return None
        
    # Identify X, Y, Z columns
    cols = list(df.columns)
    if 'X_g' in cols and 'Y_g' in cols and 'Z_g' in cols:
        data = df[['X_g', 'Y_g', 'Z_g']].values
    elif len(cols) >= 4:
        # Fallback: assume columns 1,2,3 are X,Y,Z
        data = df.iloc[:, 1:4].values
    else:
        print("❌ Could not identify X/Y/Z columns in TXT file.")
        return None
        
    # AC Coupling (remove DC bias)
    data_ac = data - np.mean(data, axis=0)
    
    # Handle sampling rate
    fs_actual = target_fs
    if 'Time_us' in cols and len(df) > 2:
        dt_us = np.median(np.diff(df['Time_us'].values))
        if dt_us > 0:
            fs_actual = 1e6 / dt_us
            print(f"📡 Detected actual sampling rate: {fs_actual:.0f} Hz")
            
    # Resample to exactly TARGET_FS if different
    if abs(fs_actual - target_fs) > 50:
        print(f"🔄 Resampling data from {fs_actual:.0f} Hz to {target_fs} Hz...")
        n_orig = len(data_ac)
        n_target = int(n_orig * target_fs / fs_actual)
        data_resampled = np.zeros((n_target, 3))
        for axis in range(3):
            data_resampled[:, axis] = resample(data_ac[:, axis], n_target)
        data_ac = data_resampled
        
    # Extract features (3 axes × 12 features = 36)
    features = []
    for axis in range(3):
        feats = extract_custom_features(data_ac[:, axis], target_fs)
        features.extend(feats)
        
    if len(features) != 36:
        print(f"❌ Feature extraction failed. Expected 36, got {len(features)}")
        return None
        
    return np.array(features)


# ==========================================
#          VISUALIZATION
# ==========================================
def plot_pca_training_vs_test(X_train_pca, y_train, x_test_pca, pred_class, pca_obj):
    """Plots VBL training clusters + NEW test point with unique marker."""
    plt.figure(figsize=(11, 8))
    
    # Plot VBL TRAINING DATA
    for cls in sorted(np.unique(y_train)):
        mask = y_train == cls
        label = REVERSE_CATEGORIES[cls]
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                   label=f"VBL Train: {label}", alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
                   
    # Plot NEW TEST DATA (Unique shape & size)
    plt.scatter(x_test_pca[0], x_test_pca[1], 
               c='red', s=350, marker='*', edgecolors='darkred', linewidths=2,
               label=f" NEW TEST: {REVERSE_CATEGORIES[pred_class]}", zorder=10)
    
    # Arrow from origin to test point
    plt.arrow(0, 0, x_test_pca[0]*0.85, x_test_pca[1]*0.85, 
             head_width=0.4, head_length=0.6, fc='red', ec='red', 
             alpha=0.5, linestyle='--', length_includes_head=True)
    
    # Annotations
    plt.annotate('NEW MEASUREMENT', 
                xy=(x_test_pca[0], x_test_pca[1]),
                xytext=(x_test_pca[0] + 1.5, x_test_pca[1] + 1.5),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
                
    plt.xlabel(f'Principal Component 1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title('PCA Feature Space: VBL Training Clusters vs New Test Measurement', 
             fontsize=14, fontweight='bold', pad=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Data Source', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
#          MAIN EXECUTION
# ==========================================
def main():
    print("="*65)
    print("🔴 PHASE 1: TRAINING (VBL DATASET ONLY)")
    print("="*65)
    
    # 1️⃣ Load VBL data
    X_train, y_train = load_vbl_training_data()
    print(f" Loaded {len(X_train)} training samples × 36 features\n")
    
    # 2️⃣ FIT Scaler & PCA on TRAINING DATA ONLY
    print("⚙️  Fitting StandardScaler & PCA on VBL training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 🔒 FIT on train
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled) # 🔒 FIT on train
    
    # 3️⃣ Train Classifier on VBL data
    print(" Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Internal validation (still VBL only)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train_scaled, y_train, test_size=0.2, 
                                               random_state=42, stratify=y_train)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"✅ VBL Internal Validation Accuracy: {acc*100:.2f}%\n")
    
    print("="*65)
    print("🟢 PHASE 2: TESTING (NEW FILE ONLY)")
    print("="*65)
    
    # 4️⃣ Process NEW file
    x_new = process_new_test_file(NEW_FILE_PATH)
    if x_new is None:
        return
        
    # 5️⃣ TRANSFORM new data using ALREADY FITTED scaler & PCA
    # ⚠️ CRITICAL: NO .fit() HERE. Prevents data leakage completely.
    x_new_scaled = scaler.transform(x_new.reshape(1, -1))  # 🔒 TRANSFORM only
    x_new_pca = pca.transform(x_new_scaled)                # 🔒 TRANSFORM only
    
    # 6️⃣ Predict
    pred_idx = clf.predict(x_new_scaled)[0]
    pred_proba = clf.predict_proba(x_new_scaled)[0]
    
    print(f"\n🎯 PREDICTION RESULT:")
    print(f"   Predicted Class: {REVERSE_CATEGORIES[pred_idx]}")
    print(f"   PCA Coordinates: ({x_new_pca[0, 0]:.3f}, {x_new_pca[0, 1]:.3f})")
    print(f"   Class Probabilities:")
    for i, prob in enumerate(pred_proba):
        print(f"      {REVERSE_CATEGORIES[i]}: {prob*100:.2f}%")
        
    print("\n" + "="*65)
    print("📊 PHASE 3: VISUALIZATION")
    print("="*65)
    
    # 7️⃣ Plot
    plot_pca_training_vs_test(X_train_pca, y_train, x_new_pca[0], pred_idx, pca)
    
    print("\n💡 INTERPRETATION:")
    print(f"   • The red star shows where your new measurement lands in the VBL feature space.")
    print(f"   • The model was trained ONLY on VBL data. The new file was NEVER used for fitting.")
    print(f"   • If the star lands inside a cluster, the vibration signature matches that fault type.")
    print(f"   • If it lands between clusters, the motor may be in a transitional or mixed state.")


if __name__ == '__main__':
    main()