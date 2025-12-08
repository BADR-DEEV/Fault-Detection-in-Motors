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
# Ensure this path points to your actual data folders
SELECT_FAULTS = [
    "normal",
    "horizontal-misalignment/0.5mm",
    "imbalance/6g",
    "overhang//ball_fault/6g",
    "underhang/ball_fault/6g",
    "vertical-misalignment/0.51mm",
]

WINDOW_SIZE = 1024
STRIDE = 512
BATCH_SIZE = 64
EPOCHS = 10 # Increased epochs slightly for the deeper model
LEARNING_RATE = 0.001

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA LOADING & SEGMENTATION
# ==========================================
# ==========================================
# 2. DATA LOADING & SEGMENTATION (UPDATED)
# ==========================================
# Define the column names based on your input
COL_NAMES = ['rot_freq', 'uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z', 'microphone']

# Select which sensor to use for training. 
# 'ohang_y' (Overhang Vertical) is often a good general purpose choice.
# You can change this to 'uhang_y' if you want to compare results.
TARGET_SENSOR = 'ohang_y' 

def load_and_segment_data(root_dir, classes, window_size, stride):
    X = []
    y = []
    valid_classes = []
    
    print(f"Loading Data... Using Sensor Column: {TARGET_SENSOR}")
    
    for idx, fault in enumerate(classes):
        folder_path = os.path.join(root_dir, fault)
        if not os.path.exists(folder_path):
            print(f"[Warning] Folder not found: {folder_path}")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if len(files) == 0:
            print(f"[Skipping] Empty class folder: {fault}")
            continue
        
        valid_classes.append(fault)
        class_id = len(valid_classes) - 1
        
        for file in files:
            try:
                # UPDATED: Use the specific column names
                df = pd.read_csv(os.path.join(folder_path, file), names=COL_NAMES)
                
                # Extract the specific sensor column by name
                signal = df[TARGET_SENSOR].values.astype(float)
                
                # Normalize signal (Standard Scaling)
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
                
                num_segments = (len(signal) - window_size) // stride
                for i in range(num_segments + 1):
                    start = i * stride
                    end = start + window_size
                    segment = signal[start:end]
                    X.append(segment)
                    y.append(class_id)
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
                
    return np.array(X), np.array(y), valid_classes

# Load data
X_raw, y_raw, valid_classes = load_and_segment_data(ROOT_DIR, SELECT_FAULTS, WINDOW_SIZE, STRIDE)

# Reshape for PyTorch: (Batch, Channels, Length) -> (N, 1, 1024)
if len(X_raw) > 0:
    X_raw = X_raw.reshape(X_raw.shape[0], 1, WINDOW_SIZE)
    num_classes = len(valid_classes)
    print(f"Total Samples: {X_raw.shape[0]}")
    print(f"Number of Classes: {num_classes}")
else:
    print("No data loaded. Please check your path.")

# ==========================================
# 3. HANDLING IMBALANCE (The Fix)
# ==========================================
# We calculate weights inversely proportional to class frequency.
# Rare classes get higher weights.
class_counts = np.bincount(y_raw, minlength=num_classes)
total_samples = len(y_raw)
class_weights = total_samples / (num_classes * (class_counts + 1e-9)) # +1e-9 avoids div by zero
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print("\n--- Imbalance Check ---")
for i, name in enumerate(valid_classes):
    print(f"Class '{name}': {class_counts[i]} samples (Weight: {class_weights[i]:.2f})")

# ==========================================
# 4. DATA SPLIT & DATASET
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

class VibrationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(VibrationDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VibrationDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(VibrationDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. USER DEFINED ARCHITECTURE
# ==========================================
class UserSpecificCNN(nn.Module):
    def __init__(self, num_classes):
        super(UserSpecificCNN, self).__init__()
        
        # Layer 1: Conv 5x32, Stride 1, ReLU
        # Padding=2 ensures output length remains same before pooling
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # Output Length: 1024 -> 512
        )
        
        # Layer 3: Conv 3x64, Stride 1, ReLU
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2) # Output Length: 512 -> 256
        )
        
        # Layer 5: Conv 5x128, Stride 1, ReLU
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2) # Output Length: 256 -> 128
        )
        
        # Layer 7: Conv 3x256, Stride 1, ReLU
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2) # Output Length: 128 -> 64
        )
        
        # Layer 9: Conv 7x256, Stride 1, ReLU (No Pooling here based on table)
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
            # No pooling
        )
        
        # Layer 10: Conv 3x32, Stride 1, ReLU
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # No pooling
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Calculate Flatten size: 
        # Length is 64 (after 4 poolings of 1024), Channels are 32 (from layer 6)
        # 64 * 32 = 2048
        
        # Layer 12: Dense 128
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 64, 128),
            nn.ReLU()
        )
        
        # Layer 13: Dense Output (Softmax is handled by CrossEntropyLoss)
        # Note: Your table said 2 outputs, but we use 'num_classes' 
        # to match your configuration list.
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

model = UserSpecificCNN(num_classes=num_classes).to(device)
print("\nModel Architecture Created Successfully.")

# ==========================================
# 6. TRAINING
# ==========================================
# Applying the WEIGHTS here handles the imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Zero the gradients (reset)
        optimizer.zero_grad()
        
        # 2. Forward pass (guess)
        outputs = model(inputs)
        
        # 3. Calculate Loss (error)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (learn)
        loss.backward()
        
        # 5. Optimizer step (update weights)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(train_loader), 100 * correct / total

def validate():
    model.eval() # Set to evaluation mode (no learning)
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # Don't calculate gradients to save memory
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), 100 * correct / total

print("\n--- Starting Training ---")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

# ==========================================
# 7. EVALUATION
# ==========================================
print("\n--- Testing Model ---")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Report
print(classification_report(all_labels, all_preds, target_names=valid_classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
# Add labels to plot
tick_marks = np.arange(len(valid_classes))
plt.xticks(tick_marks, valid_classes, rotation=45, ha="right")
plt.yticks(tick_marks, valid_classes)
plt.tight_layout()
plt.show()