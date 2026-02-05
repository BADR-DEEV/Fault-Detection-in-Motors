import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import scipy.stats as stats
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib 

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 132
STRIDE = 99
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
SENSOR_COLS = [1, 2, 3] 

# FILES
CACHE_FILE = "mafaulda_features.pkl"     # <--- Cached Data
MODEL_FILE = "dnn_model.pth"             # <--- Saved Model
SCALER_FILE = "scaler.pkl"               # <--- Saved Scaler

# PATHS (Update if needed)
NORMAL_DIR = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda\normal"
IMBALANCE_DIR = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda\imbalance"

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ==================== FEATURE EXTRACTION ====================
def extract_statistical_features(signal):
    return [
        np.mean(signal), np.std(signal), np.min(signal),
        np.percentile(signal, 25), np.median(signal), np.percentile(signal, 75),
        np.max(signal), stats.kurtosis(signal, fisher=False), stats.skew(signal),
        np.sqrt(np.mean(np.square(signal))), np.sum(np.square(signal))
    ]

def extract_multivariate_window(file_path, label):
    try:
        df = pd.read_csv(file_path, header=None, usecols=SENSOR_COLS)
        signals = df.values
        features_list = []
        for start in range(0, len(signals) - WINDOW_SIZE + 1, STRIDE):
            window = signals[start:start + WINDOW_SIZE, :]
            window_feats = []
            for axis_idx in range(3):
                window_feats.extend(extract_statistical_features(window[:, axis_idx]))
            window_feats.append(label)
            features_list.append(window_feats)
        return features_list
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# ==================== DATA LOADING (CACHED) ====================
def get_dataset(normal_dir, imbalance_dir):
    # 1. Check Cache
    if Path(CACHE_FILE).exists():
        print(f"\n🚀 Cache found at '{CACHE_FILE}'. Loading instantly...")
        return pd.read_pickle(CACHE_FILE)
    
    # 2. Process Raw Files (If cache missing)
    print("\n🐢 Cache not found. Processing raw CSVs (Takes time)...")
    features = []
    
    # Normal
    all_normal = sorted(Path(normal_dir).glob("*.csv"))
    random.shuffle(all_normal)
    for file in all_normal[:49]:
        features.extend(extract_multivariate_window(file, label=0))
        
    # Imbalance
    weights = [6, 10, 15, 20, 25, 30, 35]
    for w in weights:
        folder = None
        for name in [f"{w}g", f"{w}_g", f"imbalance_{w}g", str(w)]:
            if (Path(imbalance_dir) / name).exists():
                folder = Path(imbalance_dir) / name
                break
        if folder:
            print(f"   Processing {w}g...")
            for file in sorted(folder.glob("*.csv")):
                features.extend(extract_multivariate_window(file, label=1))

    # Build DF
    feat_names = ['mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'kurt', 'skew', 'rms', 'energy']
    cols = [f"{ax}_{fn}" for ax in ['ax', 'rad', 'tan'] for fn in feat_names] + ['label']
    df_full = pd.DataFrame(features, columns=cols)
    
    # Balance (Paper Distribution)
    df_norm = df_full[df_full.label==0].sample(n=min(len(df_full[df_full.label==0]), 1393), random_state=RANDOM_STATE)
    df_fault = df_full[df_full.label==1].sample(n=min(len(df_full[df_full.label==1]), 1157), random_state=RANDOM_STATE)
    df_final = pd.concat([df_norm, df_fault]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Save Cache
    print(f"💾 Saving cache to '{CACHE_FILE}'...")
    df_final.to_pickle(CACHE_FILE)
    
    return df_final

# ==================== MODEL & UTILS ====================
class PaperDNN(nn.Module):
    def __init__(self, input_dim):
        super(PaperDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0

# ==================== MAIN ====================
if __name__ == "__main__":
    # 1. LOAD
    df = get_dataset(NORMAL_DIR, IMBALANCE_DIR)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # 2. PREPARE
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=RANDOM_STATE)
    
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_FILE)
    
    # Tensor Conversion
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_res, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # Validation Split
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = torch.utils.data.random_split(train_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE)
    
    # 3. TRAINING
    model = PaperDNN(X_train_t.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=10)
    
    print(f"\n🔥 Starting Training on {device}...")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, train_corr, train_tot = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            train_corr += (pred == labels).sum().item()
            train_tot += labels.size(0)
            
        # Validate
        model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                out = model(inputs)
                val_loss += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                val_corr += (pred == labels).sum().item()
                val_tot += labels.size(0)
        
        # Calculate Metrics (Safe Division)
        avg_t_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        t_acc = train_corr / train_tot if train_tot > 0 else 0
        v_acc = val_corr / val_tot if val_tot > 0 else 0
        
        print(f"Ep {epoch+1:03d} | Train Acc: {t_acc:.3f} | Val Acc: {v_acc:.3f} | Val Loss: {avg_v_loss:.4f}")
        
        scheduler.step(avg_v_loss)
        early_stop(avg_v_loss, model)
        if early_stop.early_stop:
            print("🛑 Early stopping triggered!")
            model.load_state_dict(early_stop.best_model_state)
            break

    # 4. FINAL TEST
    model.eval()
    with torch.no_grad():
        _, y_pred_t = torch.max(model(X_test_t), 1)
        y_pred = y_pred_t.cpu().numpy()
        
    print("\n" + "="*40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("="*40)