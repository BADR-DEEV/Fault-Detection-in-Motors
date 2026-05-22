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
# ==================== CONFIG ====================
WINDOW_SIZE = 132
STRIDE = 99
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3

SENSOR_COLS = [1, 2, 3]

CACHE_FILE = "mafaulda_features_refactored.pkl"
MODEL_FILE = "dnn_model_refactored.pth"
SCALER_FILE = "scaler_refactored.pkl"

NORMAL_DIR = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda\normal"
IMBALANCE_DIR = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda\imbalance"

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ==================== FEATURE EXTRACTION ====================
def extract_statistical_features(x):
    return [
        np.mean(x), np.std(x), np.min(x),
        np.percentile(x, 25), np.median(x), np.percentile(x, 75),
        np.max(x),
        stats.kurtosis(x, fisher=False),
        stats.skew(x),
        np.sqrt(np.mean(x**2)),
        np.sum(x**2)
    ]


def extract_windows(file_path, label, file_id):
    df = pd.read_csv(file_path, header=None, usecols=SENSOR_COLS)
    sig = df.values

    rows = []
    for start in range(0, len(sig) - WINDOW_SIZE + 1, STRIDE):
        feats = []
        window = sig[start:start + WINDOW_SIZE]
        for ax in range(3):
            feats.extend(extract_statistical_features(window[:, ax]))

        rows.append(feats + [label, file_id])

    return rows


# ==================== DATA LOADING (CACHED) ====================
def build_dataset():
    if Path(CACHE_FILE).exists():
        return pd.read_pickle(CACHE_FILE)

    data = []
    file_id = 0

    # Normal
    for f in sorted(Path(NORMAL_DIR).glob("*.csv")):
        data.extend(extract_windows(f, 0, file_id))
        file_id += 1

    # Imbalance
    for folder in Path(IMBALANCE_DIR).glob("*"):
        if folder.is_dir():
            for f in folder.glob("*.csv"):
                data.extend(extract_windows(f, 1, file_id))
                file_id += 1

    feat_names = ['mean','std','min','q1','median','q3','max','kurt','skew','rms','energy']
    cols = [f"{ax}_{f}" for ax in ['ax','rad','tan'] for f in feat_names]
    cols += ['label', 'group']

    df = pd.DataFrame(data, columns=cols)
    df.to_pickle(CACHE_FILE)
    return df


# ==================== MODEL & UTILS ====================
class ImprovedDNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

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
    from sklearn.model_selection import GroupShuffleSplit

    df = build_dataset()

    X = df.drop(['label', 'group'], axis=1).values
    y = df['label'].values
    groups = df['group'].values

    gss = GroupShuffleSplit(train_size=0.7, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_FILE)
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        len(y_train) / (2 * class_counts),
        dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedDNN(X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_loss = np.inf
    patience, counter = 15, 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        scheduler.step(avg_loss)

        print(f"Ep {epoch+1:03d} | Train Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_FILE)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping")
                break



    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), 1).cpu().numpy()

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))




