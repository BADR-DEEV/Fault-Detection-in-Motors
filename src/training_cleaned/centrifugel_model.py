import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
#               CONFIGURATION
# ==========================================
DATA_PATH = '../../data/VBL-VA001/'
NEW_FILE_PATH = '../../data/BMP_dataset/New40gNormal.txt'
MODEL_SAVE_PATH = 'vbl_dimensionless_model.pkl'

TARGET_FS = 4000

CATEGORIES = {'normal': 0, 'bearing': 1, 'unbalance': 2, 'misalignment': 3}
REVERSE_CATEGORIES = {v: k.capitalize() for k, v in CATEGORIES.items()}

# ==========================================
#          SCALE-INVARIANT FEATURES
# ==========================================
def extract_dimensionless_features(x, fs):
    """
    Extracts features that DO NOT care about the absolute volume/amplitude of the signal.
    """
    N = len(x)
    if N == 0: return np.zeros(10)
    
    # 1. TIME-SERIES Z-NORMALIZATION (CRITICAL STEP)
    # This makes the signal volume independent. RMS will always be exactly 1.0 after this.
    std_x = np.std(x)
    if std_x < 1e-8:
        return np.zeros(10)
    x = (x - np.mean(x)) / std_x 

    # 2. Extract Shape-based features (Dimensionless)
    peak_val = np.max(np.abs(x))
    rms_val = 1.0 # By definition of Z-normalization
    mean_abs = np.mean(np.abs(x))
    
    kurt_val = kurtosis(x, fisher=False)
    skew_val = skew(x)
    crest_factor = peak_val / rms_val
    shape_factor = rms_val / (mean_abs + 1e-8)
    impulse_factor = peak_val / (mean_abs + 1e-8)
    
    # 3. Frequency domain (Normalized distribution)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(x))
    fft_vals[0] = 0 # Remove DC component
    
    # Normalize FFT so the sum of energy = 1.0
    fft_norm = fft_vals / (np.sum(fft_vals) + 1e-8)
    
    # Spectral Centroid (Where is the bulk of the frequency energy?)
    spectral_centroid = np.sum(freqs * fft_norm)
    
    # Peak Frequency (In Hz, but relative to motor speed is better)
    peak_freq = freqs[np.argmax(fft_norm)]
        
    feats = [kurt_val, skew_val, crest_factor, shape_factor, impulse_factor, 
             spectral_centroid, peak_freq, peak_val, np.median(np.abs(x)), np.var(x)]
    
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
#          MESSY ADXL FILE LOADER
# ==========================================
def clean_and_load_adxl(filepath):
    """Handles NaNs, empty rows, turn-on transients, and multiple values."""
    print(f"\n📖 Cleaning messy ADXL file: {os.path.basename(filepath)}")
    
    try:
        # Load, forcing bad lines to be ignored
        df = pd.read_csv(filepath, sep=',', on_bad_lines='skip', low_memory=False)
        
        # Force to numeric, dropping NaNs
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Ensure we have enough data
        if len(df) < 2000:
            print("❌ Not enough valid data points after cleaning.")
            return None
            
        # Try to find X, Y, Z columns
        cols = list(df.columns)
        if 'X_g' in cols and 'Y_g' in cols and 'Z_g' in cols:
            data = df[['X_g', 'Y_g', 'Z_g']].values
        elif len(cols) >= 4:
            data = df.iloc[:, 1:4].values # Assuming col 0 is Time
        else:
            data = df.iloc[:, 0:3].values
            
        # CRITICAL: Crop the first 30% of the data to skip "Motor Turning On" phase
        # And crop the last 10% to skip "Motor Turning Off" phase
        start_idx = int(len(data) * 0.3)
        end_idx = int(len(data) * 0.9)
        steady_state_data = data[start_idx:end_idx]
        
        print(f"   ✅ Cleaned! Kept {len(steady_state_data)} steady-state samples.")
        return steady_state_data

    except Exception as e:
        print(f"❌ Failed to parse ADXL file: {e}")
        return None

# ==========================================
#          DATA LOADING (VBL)
# ==========================================
def load_vbl_training_data():
    X, y = [], []
    for category, label in CATEGORIES.items():
        files = glob.glob(os.path.join(DATA_PATH, category, '*.csv'))[:200] # Limit for speed in test
        for f in files:
            try:
                df = pd.read_csv(f, usecols=[1, 2, 3], header=None)
                data = df.values.astype(float)
                
                # Downsample 20kHz -> 4kHz to match ADXL
                n_target = int(len(data) * TARGET_FS / 20000)
                data_resampled = resample(data, n_target)
                
                feats = []
                for axis in range(3):
                    axis_feats = extract_dimensionless_features(data_resampled[:, axis], TARGET_FS)
                    feats.extend(axis_feats)
                X.append(feats)
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

# ==========================================
#          MAIN EXECUTION
# ==========================================
def main():
    print("="*70)
    print("🔴 PHASE 1: DIMENSIONLESS TRAINING (VBL)")
    print("="*70)
    
    # 1. Train Model
    X_train, y_train = load_vbl_training_data()
    print(f"📦 Extracted {X_train.shape[1]} scale-invariant features.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Save objects for fast testing later
    joblib.dump({'scaler': scaler, 'pca': pca, 'clf': clf}, MODEL_SAVE_PATH)
    print(f"💾 Saved pipeline to {MODEL_SAVE_PATH}")
    
    print("\n" + "="*70)
    print("🟢 PHASE 2: TESTING PEDROLLO PKM60 (DRY RUN)")
    print("="*70)
    
    # 2. Process Pedrollo Data
    pkm_data = clean_and_load_adxl(NEW_FILE_PATH)
    if pkm_data is None: return
    
    # Extract features from PKM60
    pkm_feats = []
    for axis in range(3):
        feats = extract_dimensionless_features(pkm_data[:, axis], TARGET_FS)
        pkm_feats.extend(feats)
    
    pkm_feats = np.array(pkm_feats).reshape(1, -1)
    
    # Scale and Transform using VBL's scaler
    pkm_scaled = scaler.transform(pkm_feats)
    pkm_pca = pca.transform(pkm_scaled)
    
    pred_idx = clf.predict(pkm_scaled)[0]
    probs = clf.predict_proba(pkm_scaled)[0]
    
    print(f"\n🎯 PREDICTION FOR PKM60:")
    print(f"   Class: {REVERSE_CATEGORIES[pred_idx]}")
    print(f"   PCA Coordinates: ({pkm_pca[0, 0]:.3f}, {pkm_pca[0, 1]:.3f})")
    
    # 3. Plotting
    plt.figure(figsize=(10, 7))
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for cls in sorted(np.unique(y_train)):
        mask = y_train == cls
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], c=colors[cls], label=f"VBL: {REVERSE_CATEGORIES[cls]}", alpha=0.5, s=30)
                   
    # Plot PKM60
    plt.scatter(pkm_pca[0, 0], pkm_pca[0, 1], c='red', s=400, marker='*', edgecolors='black', label="🔴 PKM60 Dry Run")
    
    plt.title('Dimensionless PCA: VBL vs PKM60')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()