# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.io import loadmat
# from scipy.signal import welch
# from scipy.stats import skew, kurtosis
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.decomposition import PCA
# from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms
# import os

# # ================= PARAMETERS =================
# # Adjust FS based on your recording time. 
# # If 5000 points = 1 second, FS = 5000. Assuming 1000 for now.
# FS = 1000  
# WINDOW_LEN = 800  # 800 points per segment
# WINDOW_STEP = 400 # 50% overlap

# # UPDATE THIS PATH
# BASE_PATH = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\paper_data"

# # Apply professional styling
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_context("paper", font_scale=1.4)

# # =====================================================
# # CORE SIGNAL PROCESSING
# # =====================================================

# def compute_magnitude(data):
#     """
#     Converts 3-axis vibration data (N, 3) into 1D Magnitude (N,).
#     """
#     # Ensure data is numeric
#     data = np.array(data, dtype=np.float64)
#     if data.ndim == 2 and data.shape[1] == 3:
#         return np.sqrt(np.sum(data**2, axis=1))
#     return data.flatten()

# def segment_signal(sig):
#     """Slices signal into overlapping windows."""
#     segments = []
#     n_points = len(sig)
#     if n_points < WINDOW_LEN:
#         # If signal is shorter than window, pad it or take whole
#         return np.array([sig]) 
        
#     for start in range(0, n_points - WINDOW_LEN + 1, WINDOW_STEP):
#         segments.append(sig[start:start+WINDOW_LEN])
#     return np.array(segments)

# # =====================================================
# # FEATURE EXTRACTION
# # =====================================================

# def get_features(sig):
#     """Extracts Time & Frequency features from a signal segment."""
    
#     # --- Time Domain ---
#     mu = np.mean(sig)
#     std = np.std(sig)
#     skw = skew(sig)
#     kurt = kurtosis(sig)
#     rms = np.sqrt(np.mean(sig**2))
#     peak = np.max(np.abs(sig))
#     impulse_factor = peak / (np.mean(np.abs(sig)) + 1e-12)
    
#     # --- Frequency Domain ---
#     f, Pxx = welch(sig, fs=FS, nperseg=len(sig))
#     Pxx = np.maximum(Pxx, 1e-12) # Avoid log(0)
    
#     total_power = np.sum(Pxx)
    
#     # Spectral Centroid
#     centroid = np.sum(f * Pxx) / (total_power + 1e-12)
    
#     # Spectral Spread (Variance)
#     spread = np.sqrt(np.sum(((f - centroid)**2) * Pxx) / (total_power + 1e-12))
    
#     # Peak Frequency
#     peak_freq = f[np.argmax(Pxx)]
    
#     return np.array([mu, std, skw, kurt, rms, peak, impulse_factor, 
#                      total_power, centroid, spread, peak_freq])

# def process_file(file_path):
#     """Loads file, calculates magnitude, averages features over windows."""
#     try:
#         mat = loadmat(file_path)
#         # Find key automatically if 'H' isn't guaranteed, otherwise default to 'H'
#         key = 'H' if 'H' in mat else [k for k in mat.keys() if not k.startswith('_')][0]
#         data = mat[key]
        
#         # 1. Convert (5000,3) -> (5000,)
#         mag = compute_magnitude(data)
        
#         # 2. Extract Raw Indicators for Scatter Plot (Whole signal)
#         raw_rms = np.sqrt(np.mean(mag**2))
#         raw_kurtosis = kurtosis(mag)
#         rolloff = np.max(np.abs(np.fft.rfft(mag)))
#         specSpread = np.sqrt(np.sum(((np.diff(np.arange(len(mag))) - np.diff(mag)) / len(mag)) ** 2))
        
#         # 3. Extract Detailed Features (Windowed & Averaged)
#         segments = segment_signal(mag)
#         feats_list = [get_features(seg) for seg in segments]
#         avg_features = np.mean(feats_list, axis=0)
        

        
#         return raw_rms, raw_kurtosis, avg_features,  rolloff, specSpread
        
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None, None, None

# # =====================================================
# # DATA LOADING LOOP
# # =====================================================

# def load_dataset():
#     classes = {"Healthy": 0, "Faulty": 1}
    
#     raw_metrics = [] # Stores [RMS, Kurtosis]
#     features = []    # Stores high-dim vector
#     labels = []
#     raw_metrics_2 = [] # Stores [RMS, Kurtosis]
#     features_2 = []    # Stores high-dim vector
    
#     for class_name, label in classes.items():
#         folder = os.path.join(BASE_PATH, class_name)
#         if not os.path.exists(folder):
#             print(f"Warning: Folder not found: {folder}")
#             continue
            
#         print(f"Processing {class_name}...")
#         for file in os.listdir(folder):
#             if file.endswith(".mat"):
#                 r_rms, r_kurt, vec, rolloff, specSpread = process_file(os.path.join(folder, file))
#                 if vec is not None:
#                     raw_metrics.append([r_rms, r_kurt])
#                     features.append(vec)
#                     labels.append(label)
#                     raw_metrics_2.append([ rolloff, specSpread])
                    
#     return np.array(raw_metrics), np.array(features), np.array(labels), np.array(raw_metrics_2)

# # =====================================================
# # VISUALIZATION HELPERS
# # =====================================================

# def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
#     """Adds a confidence ellipse to a plot."""
#     if x.size != y.size: return
#     cov = np.cov(x, y)
#     pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
#                       facecolor=facecolor, **kwargs)
#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     mean_x = np.mean(x)
#     scale_y = np.sqrt(cov[1, 1]) * n_std
#     mean_y = np.mean(y)
#     transf = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean_x, mean_y)
#     ellipse.set_transform(transf + ax.transData)
#     return ax.add_patch(ellipse)

# # =====================================================
# # MAIN EXECUTION
# # =====================================================

# if __name__ == "__main__":
    
#     # 1. Load Data
#     raw_X, X, y, raw_metrics_2 = load_dataset()
    
#     if len(y) == 0:
#         print("No data found. Check your BASE_PATH.")
#         exit()

#     # 2. PCA Calculation
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#     var_exp = pca.explained_variance_ratio_

#     # 3. Visualization Setup
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
#     colors = {0: '#2ecc71', 1: '#e74c3c'} # Professional Green & Red
#     class_names = {0: 'Healthy', 1: 'Faulty'}

#     # --- PLOT 1: Raw Feature Scatter (RMS vs Kurtosis) ---
#     ax1 = axes[0]
#     for lbl in [0, 1]:
#         mask = y == lbl
#         ax1.scatter(raw_metrics_2[mask, 0], raw_metrics_2[mask, 1], 
#                     c=colors[lbl], label=class_names[lbl], 
#                     s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
        
#     ax1.set_xlabel("Rolloff (Bandwidth)", fontweight='bold')
#     ax1.set_ylabel("specSpread (Harmonic Complexity)", fontweight='bold')
#     ax1.set_title("Physical Indicators Scatter", fontsize=16)
#     ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
#     ax1.grid(True, linestyle='--', alpha=0.6)

#     # --- PLOT 2: PCA Visualization ---
#     ax2 = axes[1]
#     for lbl in [0, 1]:
#         mask = y == lbl
#         # Points
#         ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
#                     c=colors[lbl], label=class_names[lbl], 
#                     s=120, alpha=0.8, edgecolors='k', linewidth=0.6)
        
#         # Add Confidence Ellipses (2 Std Dev)
#         confidence_ellipse(X_pca[mask, 0], X_pca[mask, 1], ax2, n_std=2.0, 
#                            edgecolor=colors[lbl], linestyle='--', linewidth=2)

#     ax2.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1%} Var)", fontweight='bold')
#     ax2.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1%} Var)", fontweight='bold')
#     ax2.set_title("PCA Feature Space Separation", fontsize=16)
#     ax2.legend(loc='upper right', frameon=True)
#     ax2.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.show()

#     print(f"Dataset Size: {len(y)} samples")
#     print(f"Total Features extracted per file: {X.shape[1]}")
#     print(f"PCA Explained Variance: {var_exp}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import os

# ================= PARAMETERS =================
FS = 1000
WINDOW_LEN = 800
WINDOW_STEP = 400
BASE_PATH = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\paper_data"

# Professional Styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =====================================================
# 1. USER PROVIDED FEATURE FUNCTIONS (EXACT COPY)
# =====================================================

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

def freqB_features(sig, fs=FS):
    # SC, SF, SRO, SFL, SCR, SDEC, SSL, SS
    if sig.size == 0:
        return np.zeros(8)
    f, Pxx = compute_psd(sig, fs)
    totalE = np.sum(Pxx)
    
    # 1. Spectral Centroid
    SC = np.sum(f * Pxx) / (totalE + 1e-12)
    
    # 2. Spectral Flux (Simplified variation)
    Pnorm = Pxx / (np.sum(Pxx) + 1e-12)
    SF = float(np.sum((np.diff(Pnorm, prepend=Pnorm[0])) ** 2))
    
    # 3. Roll-off
    cumulative = np.cumsum(Pxx)
    idx_roll = np.searchsorted(cumulative, 0.85 * totalE)
    SRO = float(f[min(idx_roll, len(f) - 1)])
    
    # 4. Flatness
    geo_mean = np.exp(np.mean(np.log(Pxx + 1e-12)))
    arith_mean = np.mean(Pxx)
    SFL = float(geo_mean / (arith_mean + 1e-12))
    
    # 5. Crest (Freq Domain)
    SCR = float(np.max(Pxx) / (arith_mean + 1e-12))
    
    # 6. Decrease
    if len(Pxx) >= 2:
        n = np.arange(1, len(Pxx) + 1)
        # Fix shape mismatch for safe calc
        limit = min(len(Pxx)-1, len(n)-1)
        diffs = (Pxx[:limit] - Pxx[1:limit+1]) / (n[:limit] + 1e-12)
        SDEC = float(np.sum(diffs) / (np.sum(Pxx) + 1e-12))
    else:
        SDEC = 0.0
        
    # 7. Slope
    try:
        slope, _ = np.polyfit(f, Pxx, 1)
    except Exception:
        slope = 0.0
    SSL = float(slope)
    
    # 8. Spread
    SS = float(np.sqrt(np.sum(((f - SC) ** 2) * Pxx) / (totalE + 1e-12)))
    
    return np.array([SC, SF, SRO, SFL, SCR, SDEC, SSL, SS], dtype=float)

# =====================================================
# 2. DATA PROCESSING
# =====================================================

def compute_magnitude(data):
    """Converts (N, 3) to (N,) Magnitude."""
    data = np.array(data, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] == 3:
        return np.sqrt(np.sum(data**2, axis=1))
    return data.flatten()

def extract_f6_vector(sig):
    """Computes F6 (Temporal + FreqB) -> 15 Features"""
    t = temporal_features(sig) # 7
    b = freqB_features(sig)    # 8
    return np.concatenate((t, b))

def process_folder(folder_path, label_name):
    extracted_data = []
    if not os.path.exists(folder_path):
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            try:
                mat = loadmat(os.path.join(folder_path, file))
                key = 'H' if 'H' in mat else [k for k in mat.keys() if not k.startswith('_')][0]
                
                # 1. Get Magnitude
                mag = compute_magnitude(mat[key])
                
                # 2. Window & Feature Extraction
                feats_temp = []
                for start in range(0, len(mag) - WINDOW_LEN + 1, WINDOW_STEP):
                    segment = mag[start:start+WINDOW_LEN]
                    feats_temp.append(extract_f6_vector(segment))
                
                # 3. Average Windows -> 1 Vector per File
                if feats_temp:
                    avg_feats = np.mean(feats_temp, axis=0)
                    row = list(avg_feats)
                    row.append(label_name)
                    extracted_data.append(row)
                    
            except Exception as e:
                print(f"Error {file}: {e}")
    return extracted_data

# =====================================================
# 3. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    # Define Column Names for F6 (15 Features)
    f6_names = [
        # Temporal (7)
        'Mean', 'StdDev', 'Skewness', 'Kurtosis', 'Peak2Peak', 'RMS', 'Energy',
        # FreqB (8)
        'SpecCentroid', 'SpecFlux', 'RollOff', 'SpecFlatness', 
        'SpecCrest', 'SpecDecrease', 'SpecSlope', 'SpecSpread',
        'Condition'
    ]

    print("Extracting F6 Features...")
    data_h = process_folder(os.path.join(BASE_PATH, "Healthy"), "Healthy")
    data_f = process_folder(os.path.join(BASE_PATH, "Faulty"), "Faulty")
    
    df = pd.DataFrame(data_h + data_f, columns=f6_names)
    
    if df.empty:
        print("No data found.")
        exit()

    # =====================================================
    # 4. STATISTICAL ANALYSIS & RANKING
    # =====================================================
    print("\n" + "="*60)
    print("       F6 FEATURE SEPARATION ANALYSIS (Sorted by Quality)       ")
    print("="*60)
    print(f"{'Rank':<4} | {'Feature':<15} | {'Healthy':<10} | {'Faulty':<10} | {'Score':<6} | {'Status'}")
    print("-" * 65)

    # Separation Score = |Mean1 - Mean2| / (Std1 + Std2)
    # This is better than just difference; it accounts for variance.
    stats = []
    
    for feat in f6_names[:-1]: # Skip 'Condition'
        h_data = df[df['Condition']=='Healthy'][feat]
        f_data = df[df['Condition']=='Faulty'][feat]
        
        mu1, mu2 = h_data.mean(), f_data.mean()
        std1, std2 = h_data.std(), f_data.std()
        
        # Fisher Score Calculation
        score = abs(mu1 - mu2) / (std1 + std2 + 1e-9)
        
        status = "POOR"
        if score > 0.5: status = "OKAY"
        if score > 1.0: status = "GOOD"
        if score > 2.0: status = "GREAT"
        
        stats.append((feat, mu1, mu2, score, status))

    # Sort by Score (High to Low)
    stats.sort(key=lambda x: x[3], reverse=True)

    # Print Table
    for i, (feat, m1, m2, sc, st) in enumerate(stats):
        print(f"{i+1:<4} | {feat:<15} | {m1:<10.2f} | {m2:<10.2f} | {sc:<6.2f} | {st}")

    print("="*60 + "\n")

    # =====================================================
    # 5. SMART PLOTTING (Top 6 Features Only)
    # =====================================================
    
    # Get Top 6 feature names
    top_features = [x[0] for x in stats[:6]]
    top_features.append('Condition') # Add label back
    
    print(f"Plotting Top 6 Features: {top_features[:-1]}")
    
    df_plot = df[top_features]

    g = sns.pairplot(
        df_plot, 
        hue="Condition",
        palette={"Healthy": "#2ecc71", "Faulty": "#e74c3c"},
        corner=True,           # Hides upper triangle (Redundancy)
        kind="scatter",
        diag_kind="kde",
        plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'w'},
        height=2.2,            # Good size
        aspect=1.2
    )
    
    g.fig.suptitle("F6 Feature Analysis (Top 6 Ranked Metrics)", y=1.02, fontsize=22)
    plt.show()