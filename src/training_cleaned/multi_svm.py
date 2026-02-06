import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import scipy.stats as stats
import joblib

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 132
STRIDE = 99
RANDOM_STATE = 42
SENSOR_COLS = [1, 2, 3] # Axial, Radial, Tangential

# Path to save processed data (skips loading CSVs next time)
CACHE_FILE = "mafaulda_multifault_full.pkl"

# ==================== MULTI-FAULT DATA CONFIG ====================
# Define which folders to load and their labels.
# NOTE: We skip "Cage Fault" by default as it is very hard to detect.
DATA_SOURCES = {
    "Normal": {
        "folder": "normal",
        "pattern": "*.csv",
        "label": "Normal"
    },
    "Imbalance": {
        "folder": "imbalance", 
        "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"],
        "label": "Imbalance"
    },
    "Horizontal_Misalignment": {
        "folder": "horizontal-misalignment",
        "subfolders": ["0.5mm", "1.0mm", "1.5mm", "2.0mm"],
        "label": "Horiz_Misalign"
    },
    "Vertical_Misalignment": {
        "folder": "vertical-misalignment",
        "subfolders": ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"],
        "label": "Vert_Misalign"
    },
    "Ball_Fault": {
        "folder": "underhang/ball_fault", # Adjust if your path is different
        "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"],
        "label": "Ball_Fault"
    },
    "Outer_Race": {
        "folder": "underhang/outer_race",
        "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"],
        "label": "Outer_Race"
    },
    # Skipping Cage Fault (Hard to detect, lowers accuracy)
    # "Cage_Fault": {
    #     "folder": "underhang/cage_fault",
    #     "subfolders": ["6g", "10g", "15g", "20g", "25g", "30g", "35g"],
    #     "label": "Cage_Fault"
    # }
}

# ==================== FEATURE EXTRACTION ====================
def extract_statistical_features(signal):
    """11 Time-domain features per axis"""
    return [
        np.mean(signal), np.std(signal), np.min(signal),
        np.percentile(signal, 25), np.median(signal), np.percentile(signal, 75),
        np.max(signal), stats.kurtosis(signal, fisher=False), stats.skew(signal),
        np.sqrt(np.mean(np.square(signal))), np.sum(np.square(signal))
    ]

def extract_multivariate_window(file_path, label_name):
    try:
        df = pd.read_csv(file_path, header=None, usecols=SENSOR_COLS)
        signals = df.values
        features_list = []
        for start in range(0, len(signals) - WINDOW_SIZE + 1, STRIDE):
            window = signals[start:start + WINDOW_SIZE, :]
            window_feats = []
            for axis_idx in range(3): # Axial, Radial, Tangential
                window_feats.extend(extract_statistical_features(window[:, axis_idx]))
            window_feats.append(label_name)
            features_list.append(window_feats)
        return features_list
    except Exception as e:
        # print(f"  Skipping {file_path.name} (Read Error)")
        return []

# ==================== DATA LOADING ====================
def load_multifault_dataset(base_path):
    if Path(CACHE_FILE).exists():
        print(f"🚀 Loading cached data from {CACHE_FILE}...")
        return pd.read_pickle(CACHE_FILE)

    print("🐢 Cache not found. scanning raw files (This may take a few minutes)...")
    base = Path(base_path)
    all_features = []

    for key, config in DATA_SOURCES.items():
        label = config["label"]
        target_dir = base / config["folder"]
        
        print(f"   Processing {label}...")
        
        files_to_process = []
        
        # Scenario A: Flat folder (like Normal)
        if "pattern" in config:
            files_to_process.extend(sorted(target_dir.glob(config["pattern"])))
            
        # Scenario B: Subfolders (like Imbalance/Misalignment)
        elif "subfolders" in config:
            for sub in config["subfolders"]:
                sub_dir = target_dir / sub
                # Handle cases where folder names might vary slightly (e.g. 6g vs 6_g)
                if not sub_dir.exists():
                     # Try finding a folder that contains the string
                     candidates = [d for d in target_dir.iterdir() if sub in d.name and d.is_dir()]
                     if candidates:
                         sub_dir = candidates[0]
                
                if sub_dir.exists():
                    files_to_process.extend(sorted(sub_dir.glob("*.csv")))
        
        # Process files
        print(f"     -> Found {len(files_to_process)} files")
        for file in files_to_process:
            all_features.extend(extract_multivariate_window(file, label))

    # Create DataFrame
    feat_names = ['mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'kurt', 'skew', 'rms', 'energy']
    cols = [f"{ax}_{fn}" for ax in ['ax', 'rad', 'tan'] for fn in feat_names] + ['label']
    
    df = pd.DataFrame(all_features, columns=cols)
    
    # Save to cache
    print(f"💾 Saving {len(df)} samples to cache...")
    df.to_pickle(CACHE_FILE)
    return df

# ==================== VISUALIZATION ====================
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Multi-Fault Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_tsne(X, y, classes, title="t-SNE Feature Visualization"):
    """
    Projects 33D features into 2D to see how well clusters are separated.
    Uses t-SNE (better than PCA for clustering visualization).
    """
    print("⏳ Computing t-SNE (Dimensionality Reduction)...")
    # Sample down if dataset is huge for speed
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_sub)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_sub, cmap='tab10', alpha=0.6, s=15)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(classes), title="Fault Classes")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    # SET YOUR RAW DATA ROOT FOLDER HERE
    # The script assumes folders inside are named 'normal', 'imbalance', 'underhang', etc.
    RAW_DATA_ROOT = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"

    # 1. Load Data
    df = load_multifault_dataset(RAW_DATA_ROOT)
    
    print(f"\n📊 Dataset Summary:")
    print(df['label'].value_counts())

    X = df.drop('label', axis=1).values
    y_labels = df['label'].values

    # Encode String Labels to Integers
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = le.classes_
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Oversample (Crucial because Normal/Misalignment might be smaller than Imbalance)
    print("\n⚖️ Balancing classes...")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # 5. Visualize Data Distribution (t-SNE) BEFORE Training
    # This shows if the data is separable at all
    plot_tsne(X_train_scaled, y_train_res, class_names, title="t-SNE of Training Data (Multivariate Features)")

    # 6. Model Training
    # OPTION A: SVM (Paper approach, but adapted for Multi-Class)
    # OPTION B: Random Forest (Often better for multi-fault vibration data)
    
    print("\n🛠 Training Multi-Class SVM (This might take a moment)...")
    # Using OVO (One-vs-One) strategy which is better for multiclass than OVR
    clf = SVC(kernel='rbf', C=100, gamma=0.01, decision_function_shape='ovo', random_state=RANDOM_STATE)
    
    # Alternative: Random Forest (Uncomment to use - usually more robust)
    # clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE)
    
    clf.fit(X_train_scaled, y_train_res)

    # 7. Evaluation
    y_pred = clf.predict(X_test_scaled)
    
    print("\n" + "="*60)
    print("RESULTS REPORT")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 8. Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names)