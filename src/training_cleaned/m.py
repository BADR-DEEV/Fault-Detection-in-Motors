import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             roc_curve, auc, make_scorer, f1_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from collections import defaultdict
import scipy.signal
from scipy.signal import welch
import scipy.stats as stats
import logging
import time
import random
import joblib
import plotly.express as px

# ==================== CONFIGURATION ====================
# ADXL355Z sampling optimization
DECIMATION_FACTOR = 13  # 50000 / 13 = 3846.15 Hz 
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

# Large window for high frequency resolution (0.94 Hz)
WINDOW_SIZE = 4096  
STRIDE = 1024  

VIBRATION_COLS = [1, 2, 3]  # Axial, Radial, Tangential
TACH_COL = 0
RANDOM_STATE = 42

RPM_RANGES = {
    'low': (767, 1500),
    'mid': (1501, 2500),
    'high': (2501, 3686)
}

# ==================== 1. PHYSICS-BASED FEATURE EXTRACTION ====================
def extract_features_with_harmonics(vib_signal, tach_signal, sampling_freq):
    """
    The 'Previous' Robust Feature Extraction.
    Focuses heavily on Harmonic Ratios to fix Horizontal Misalignment detection.
    """
    rpm = calculate_rpm_from_tach(tach_signal, sampling_freq)
    if rpm is None or rpm < 500 or rpm > 4000:
        return None, None
    
    fundamental_hz = rpm / 60.0
    features = []
    rms_vals = []
    harmonic_energies = {'ax': [], 'rad': [], 'tan': []}
    
    # Process each axis
    for ax_idx, axis_name in enumerate(['ax', 'rad', 'tan']):
        signal = vib_signal[:, ax_idx]
        
        # --- Time Domain ---
        rms = np.sqrt(np.mean(signal**2))
        kur = stats.kurtosis(signal, fisher=False)
        rms_vals.append(rms)
        
        # --- Frequency Domain (Welch) ---
        # High nperseg for better resolution at low freq
        nperseg = min(2048, len(signal))
        f, Pxx = welch(signal, fs=sampling_freq, nperseg=nperseg, scaling='density')
        Pxx = np.maximum(Pxx, 1e-15)
        
        # Capture Dominant Freq (For Plotting & Diagnosis)
        dom_freq = f[np.argmax(Pxx)]
        
        # --- HARMONIC EXTRACTION (CRITICAL) ---
        # Bandwidth = +/- 5% of target frequency
        bw_1x = 0.05 * fundamental_hz
        bw_2x = 0.05 * (2 * fundamental_hz)
        
        # Energy at 1x RPM
        idx_1x = np.where((f >= fundamental_hz - bw_1x) & (f <= fundamental_hz + bw_1x))[0]
        e_1x = np.sum(Pxx[idx_1x]) if len(idx_1x) > 0 else 1e-15
        
        # Energy at 2x RPM (The Misalignment Fingerprint)
        idx_2x = np.where((f >= 2*fundamental_hz - bw_2x) & (f <= 2*fundamental_hz + bw_2x))[0]
        e_2x = np.sum(Pxx[idx_2x]) if len(idx_2x) > 0 else 1e-15
        
        harmonic_energies[axis_name] = (e_1x, e_2x)
        
        # Spectral Shape Stats
        total_E = np.sum(Pxx)
        if total_E == 0:
            fm, fsd, fmed, sro = 0, 0, 0, 0
        else:
            fm = np.sum(f * Pxx) / total_E
            fsd = np.sqrt(np.sum(((f - fm) ** 2) * Pxx) / total_E)
            cumsum = np.cumsum(Pxx)
            fmed = f[np.searchsorted(cumsum, 0.5 * total_E)]
            sro = f[np.searchsorted(cumsum, 0.85 * total_E)]
            
        features.extend([rms, kur, fm, fsd, fmed, sro, dom_freq])

    # --- CROSS-AXIS & HARMONIC RATIOS (The Logic that fixes Misalignment) ---
    
    # 1. Axial Ratio (Misalignment vibrates axially)
    rms_ax, rms_rad, rms_tan = rms_vals
    axial_ratio = rms_ax / (rms_rad + rms_tan + 1e-12)
    
    # 2. Axial Harmonic Ratio (2x / 1x)
    # Strong 2x in Axial is the definition of Misalignment
    ax_1x, ax_2x = harmonic_energies['ax']
    harmo_ratio_ax = ax_2x / (ax_1x + 1e-12)
    
    # 3. Radial Harmonic Ratio (2x / 1x)
    rad_1x, rad_2x = harmonic_energies['rad']
    harmo_ratio_rad = rad_2x / (rad_1x + 1e-12)
    
    # 4. Axial 2x Dominance (Axial 2x vs Radial 2x)
    ax_2x_dom = ax_2x / (rad_2x + 1e-12)
    
    features.extend([axial_ratio, harmo_ratio_ax, harmo_ratio_rad, ax_2x_dom])
    
    return features, rpm

def calculate_rpm_from_tach(tach_signal, sampling_freq):
    if len(tach_signal) < 10: return None
    threshold = np.mean(tach_signal)
    binary = tach_signal > threshold
    rising = np.where((binary[:-1] == False) & (binary[1:] == True))[0]
    if len(rising) < 2: return None
    diff = (rising[-1] - rising[0]) / sampling_freq
    revs = len(rising) - 1
    if diff <= 0 or revs == 0: return None
    return (revs / diff) * 60

# ==================== 2. DATA LOADING (FILE SPLIT) ====================
def load_data(base_path):
    logger.info("⏳ Loading data with FILE-LEVEL SPLIT...")
    base = Path(base_path)
    
    SOURCES = {
        "Normal": {"root": "normal", "pat": ["*.csv"]},
        "Imbalance": {"root": "imbalance", "subs": ["6g","10g","15g","20g","25g","30g","35g"]},
        "Horiz_Misalign": {"root": "horizontal-misalignment", "subs": ["0.5mm","1.0mm","1.5mm","2.0mm"]},
        "Vert_Misalign": {"root": "vertical-misalignment", "subs": ["0.51mm","0.63mm","1.27mm","1.40mm","1.78mm","1.90mm"]},
        "Ball_Fault": {"root": "underhang/ball_fault", "subs": ["6g","10g","15g","20g","25g","30g","35g"]},
        "Outer_Race": {"root": "underhang/outer_race", "subs": ["6g","10g","15g","20g","25g","30g","35g"]}
    }
    
    # Collect files
    file_map = defaultdict(list)
    for cls, cfg in SOURCES.items():
        p = base
        for part in cfg['root'].split('/'): p = p / part
        
        found = []
        if "subs" in cfg:
            for sub in cfg['subs']:
                matches = [d for d in p.iterdir() if d.is_dir() and sub in d.name]
                if matches: found.extend(sorted(matches[0].glob("*.csv")))
        else:
            found.extend(sorted(p.glob(cfg['pat'][0])))
            
        random.seed(RANDOM_STATE)
        random.shuffle(found)
        file_map[cls] = found[:60] # Balance file count
        
    # Split Files
    train_files, test_files = {}, {}
    for cls, files in file_map.items():
        cut = int(len(files) * 0.3)
        test_files[cls] = files[:cut]
        train_files[cls] = files[cut:]
        
    # Process
    def process(f_map):
        feats, rpms, labs = [], [], []
        for cls, files in f_map.items():
            for f in files:
                try:
                    df = pd.read_csv(f, header=None)
                    vib = scipy.signal.decimate(df.values[:, VIBRATION_COLS], DECIMATION_FACTOR, axis=0)
                    tach = scipy.signal.decimate(df.values[:, TACH_COL], DECIMATION_FACTOR, axis=0)
                    
                    for i in range(0, len(vib)-WINDOW_SIZE, STRIDE):
                        f_vec, rpm = extract_features_with_harmonics(vib[i:i+WINDOW_SIZE], tach[i:i+WINDOW_SIZE], SAMPLING_FREQ_DECIMATED)
                        if f_vec:
                            feats.append(f_vec)
                            rpms.append(rpm)
                            labs.append(cls)
                except: continue
        return feats, rpms, labs

    tr_f, tr_r, tr_l = process(train_files)
    te_f, te_r, te_l = process(test_files)
    
    # Col Names
    axes = ['ax', 'rad', 'tan']
    basics = ['rms','kur','fm','fsd','fmed','sro','dom_freq']
    cols = [f"{a}_{b}" for a in axes for b in basics] + ['ax_ratio','harm_ratio_ax','harm_ratio_rad','ax_2x_dom']
    
    df_tr = pd.DataFrame(tr_f, columns=cols)
    df_tr['label'] = tr_l
    df_tr['rpm'] = tr_r
    
    df_te = pd.DataFrame(te_f, columns=cols)
    df_te['label'] = te_l
    df_te['rpm'] = te_r
    
    # Filter RPM & Balance
    def clean(df):
        df = df[(df['rpm'] >= 500) & (df['rpm'] <= 4000)]
        min_c = min(df['label'].value_counts().min(), 2000)
        return df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), min_c), random_state=RANDOM_STATE))

    return clean(df_tr), clean(df_te), cols

# ==================== 3. PLOTTING FUNCTIONS ====================

def plot_frequency_domain_multiclass(df, class_names):
    logger.info("🔬 Generating Frequency Domain Plot...")
    if 'ax_dom_freq' not in df.columns: return
    
    samples = []
    for cls in class_names:
        tmp = df[df['label']==cls]
        if len(tmp)>0: samples.append(tmp.sample(min(200,len(tmp)), random_state=42))
    
    if not samples: return
    plot_df = pd.concat(samples)
    plot_df['fault_type'] = plot_df['label']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()
    ax_map = {'ax':'Axial','rad':'Radial','tan':'Tangential'}
    
    for i, (code, name) in enumerate(ax_map.items()):
        col = f"{code}_dom_freq"
        
        # Violin
        sns.violinplot(data=plot_df, x='fault_type', y=col, ax=axes[i], hue='fault_type', palette='viridis', legend=False)
        axes[i].set_title(f"{name}: Dominant Freq")
        axes[i].tick_params(axis='x', rotation=45)
        
        # Scatter
        ax_s = axes[i+3]
        base = np.linspace(10, 60, 100)
        ax_s.plot(base, base, 'k--', alpha=0.4, label='1x RPM')
        ax_s.plot(base, 2*base, 'r--', alpha=0.4, label='2x RPM')
        sns.scatterplot(data=plot_df, x=plot_df['rpm']/60, y=col, hue='fault_type', style='fault_type', ax=ax_s, palette='viridis')
        ax_s.set_title(f"{name}: Freq vs Speed")
        ax_s.set_ylim(0, 150)
        
    plt.show()

def plot_confusion_matrix_enhanced(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Absolute Counts')
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names, rotation=0)
    
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', ax=ax2)
    ax2.set_title('Normalized (%)')
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_yticklabels(class_names, rotation=0)
    plt.tight_layout()
    plt.show()

def plot_pca_multiclass(X, y, class_names):
    logger.info("🎨 Generating PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f"PCA (Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def plot_tsne_interactive(X, y, class_names):
    logger.info("🎨 Generating Interactive t-SNE...")
    try:
        if len(X) > 1000:
            idx = np.random.choice(len(X), 1000, replace=False)
            X, y = X[idx], y[idx]
        
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        res = tsne.fit_transform(X)
        df = pd.DataFrame({'x':res[:,0], 'y':res[:,1], 'z':res[:,2], 'Label':[class_names[i] for i in y]})
        
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='Label', title="3D t-SNE Clusters")
        fig.update_traces(marker=dict(size=4))
        fig.show()
    except Exception as e: logger.warning(f"t-SNE skipped: {e}")

def plot_rpm_stratified(df, y_true, y_pred, class_names):
    logger.info("⚙️ RPM Stratification...")
    res = defaultdict(lambda: defaultdict(list))
    for name, (low, high) in RPM_RANGES.items():
        mask = (df['rpm'] >= low) & (df['rpm'] <= high)
        if sum(mask) < 10: continue
        yt, yp = y_true[mask], y_pred[mask]
        for i, cls in enumerate(class_names):
            c_mask = yt == i
            if sum(c_mask) > 5:
                res[name][cls].append(accuracy_score(yt[c_mask], yp[c_mask]))
                
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, cls in enumerate(class_names):
        ax = axes[i]
        vals = [np.mean(res[r][cls]) for r in RPM_RANGES if cls in res[r]]
        lbls = [r.upper() for r in RPM_RANGES if cls in res[r]]
        if vals:
            ax.bar(lbls, vals, color='skyblue')
            ax.set_ylim(0, 1.05)
            ax.set_title(cls)
            ax.axhline(0.85, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X_tr, y_tr, X_te, y_te, feats, class_names):
    logger.info("🧠 Feature Importance...")
    # Global Top 15
    r = permutation_importance(model, X_te, y_te, n_repeats=2, random_state=42, n_jobs=-1)
    top_idx = r.importances_mean.argsort()[::-1][:15]
    top_feats = [feats[i] for i in top_idx]
    
    mat = np.zeros((len(class_names), len(top_idx)))
    for i, cls in enumerate(class_names):
        y_b_tr = (y_tr == i).astype(int)
        y_b_te = (y_te == i).astype(int)
        if sum(y_b_tr) < 10: continue
        
        clf = SVC(kernel='rbf', C=10, class_weight='balanced')
        clf.fit(X_tr[:, top_idx], y_b_tr)
        r_cls = permutation_importance(clf, X_te[:, top_idx], y_b_te, scoring='f1', n_repeats=2, n_jobs=-1)
        mat[i] = np.maximum(r_cls.importances_mean, 0)
        
    plt.figure(figsize=(12, 8))
    sns.heatmap(mat, annot=True, fmt='.2f', xticklabels=top_feats, yticklabels=class_names, cmap='viridis')
    plt.title("Feature Importance (F1-Score Impact)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_roc(y_te, y_score, class_names):
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_te == i, y_score[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.legend()
    plt.title("ROC Curves")
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    logger = logging.getLogger()
    
    # 1. Load
    RAW_PATH = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"
    df_tr, df_te, cols = load_data(RAW_PATH)
    logger.info(f"Loaded: Train {len(df_tr)}, Test {len(df_te)}")
    
    X_tr = df_tr[cols].values
    y_tr = df_tr['label'].values
    X_te = df_te[cols].values
    y_te = df_te['label'].values
    
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)
    classes = le.classes_
    
    # 2. Scale
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    # 3. Train
    logger.info("🧠 Training SVM...")
    model = SVC(kernel='rbf', C=100, gamma=0.1, probability=True, random_state=42)
    model.fit(X_tr_sc, y_tr_enc)
    
    # 4. Save
    joblib.dump(model, 'multi_4khz.pkl')
    joblib.dump(scaler, 'scaler_4khz.pkl')
    logger.info("💾 Model Saved.")
    
    # 5. Eval
    y_pred = model.predict(X_te_sc)
    y_prob = model.decision_function(X_te_sc)
    
    print("\nResults:")
    print(classification_report(y_te_enc, y_pred, target_names=classes))
    
    # 6. Plots
    plot_confusion_matrix_enhanced(y_te_enc, y_pred, classes)
    plot_frequency_domain_multiclass(df_te, classes)
    plot_pca_multiclass(X_te_sc, y_te_enc, classes)
    plot_tsne_interactive(X_te_sc, y_te_enc, classes)
    plot_rpm_stratified(df_te, y_te_enc, y_pred, classes)
    plot_roc(y_te_enc, y_prob, classes)
    plot_feature_importance(model, X_tr_sc, y_tr_enc, X_te_sc, y_te_enc, cols, classes)
    
    logger.info("Done.")