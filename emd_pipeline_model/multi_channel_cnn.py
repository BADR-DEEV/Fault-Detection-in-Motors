import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT_DIR = "../Mafulda_Dataset/"
SELECT_FAULTS = [
    "normal",
    "horizontal-misalignment/0.5mm",
    "imbalance/6g",
    "overhang/ball_fault",
    "underhang/ball_fault",
    "vertical-misalignment/0.51mm",
]

# We now use ALL accelerometer data
COL_NAMES = ['rot_freq', 'uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z', 'microphone']
SENSORS_TO_USE = ['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z']

WINDOW_SIZE = 1024
STRIDE = 512
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Using {len(SENSORS_TO_USE)} Input Channels: {SENSORS_TO_USE}")

# ==========================================
# 2. MULTI-CHANNEL DATA LOADING
# ==========================================
def load_and_segment_data(root_dir, classes, sensors, window_size, stride):
    X = []
    y = []
    valid_classes = []
    
    print("Loading Multi-Channel Data...")
    
    for idx, fault in enumerate(classes):
        folder_path = os.path.join(root_dir, fault)
        if not os.path.exists(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if len(files) == 0:
            continue
        
        valid_classes.append(fault)
        class_id = len(valid_classes) - 1
        
        for file in files:
            try:
                df = pd.read_csv(os.path.join(folder_path, file), names=COL_NAMES)
                
                # Extract all 6 sensor columns
                # Shape becomes (Total_Time_Points, 6)
                signals = df[sensors].values.astype(float)
                
                # Normalize each channel INDEPENDENTLY
                # (Subtract mean of column X from column X, etc.)
                mean = np.mean(signals, axis=0)
                std = np.std(signals, axis=0)
                signals = (signals - mean) / (std + 1e-9)
                
                # Segment
                num_segments = (len(signals) - window_size) // stride
                for i in range(num_segments + 1):
                    start = i * stride
                    end = start + window_size
                    
                    # Extract segment: (1024, 6)
                    segment = signals[start:end]
                    
                    # Transpose for PyTorch Conv1d: Needs (Channels, Length)
                    # Shape becomes (6, 1024)
                    segment = segment.T 
                    
                    X.append(segment)
                    y.append(class_id)
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
                
    return np.array(X), np.array(y), valid_classes

X_raw, y_raw, valid_classes = load_and_segment_data(ROOT_DIR, SELECT_FAULTS, SENSORS_TO_USE, WINDOW_SIZE, STRIDE)

if len(X_raw) == 0:
    raise ValueError("No data loaded.")

# X_raw is already (N, 6, 1024) thanks to the Transpose in the loader
print(f"Data Shape: {X_raw.shape}") # Should be (Samples, 6, 1024)

# ==========================================
# 3. IMBALANCE HANDLING
# ==========================================
num_classes = len(valid_classes)
class_counts = np.bincount(y_raw, minlength=num_classes)
weights = len(y_raw) / (num_classes * (class_counts + 1e-9))
class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

# ==========================================
# 4. DATA SPLIT & DATASET
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

class VibrationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(VibrationDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VibrationDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(VibrationDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. MULTI-CHANNEL CNN ARCHITECTURE
# ==========================================
class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MultiChannelCNN, self).__init__()
        
        # Layer 1: Conv 5x32
        # NOTICE: in_channels is now dynamic (e.g., 6)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), # Added BatchNorm for stability
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 3: Conv 3x64
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 5: Conv 5x128
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 7: Conv 3x256
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 9: Conv 7x256
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Layer 10: Conv 3x32
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # Flatten size calculation remains 32 channels * 64 length = 2048
        # (The number of input channels only affects the FIRST layer weights, not the output size)
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5) # Added dropout to prevent overfitting
        )
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate with 6 input channels
model = MultiChannelCNN(num_classes=num_classes, input_channels=len(SENSORS_TO_USE)).to(DEVICE)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), 100 * correct / total

def validate():
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), 100 * correct / total

print("\n--- Starting Multi-Channel Training ---")
for epoch in range(EPOCHS):
    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = validate()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} ({t_acc:.1f}%) | Val Loss: {v_loss:.4f} ({v_acc:.1f}%)")

# ==========================================
# 7. FINAL TEST
# ==========================================
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=valid_classes))