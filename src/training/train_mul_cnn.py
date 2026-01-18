import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_loader import load_mafulda_dataset

# ==========================================
# 1. CONFIGURATION (OPTIMIZED FOR CPU)
# ==========================================
SENSORS_TO_USE = [
    'uhang_x', 'uhang_y', 'uhang_z',
    'ohang_x', 'ohang_y', 'ohang_z'
]

# --- FIX FOR FREEZING ---
# We use a larger stride to generate fewer samples.
# Stride 1024 = No overlap between windows.
WINDOW_SIZE = 1024
STRIDE = 1024 

BATCH_SIZE = 32 # Reduced batch size for CPU
EPOCHS = 10     # Reduced epochs for quicker results
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # src/
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Results will be saved to: {RESULTS_DIR}")

# ==========================================
# 2. LOAD DATA
# ==========================================
X_raw, y_raw, valid_classes = load_mafulda_dataset(
    sensors=SENSORS_TO_USE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    original_fs=50000,
    target_fs=4000
)

print(f"Loaded data shape: {X_raw.shape}")
# Expected: Approx (30,000 to 50,000, 6, 1024) - Safe for RAM

if len(X_raw) == 0:
    raise ValueError("No data loaded. Check paths.")

# ==========================================
# 3. SPLIT & DATASET
# ==========================================
# Weights for imbalance
num_classes = len(valid_classes)
class_counts = np.bincount(y_raw, minlength=num_classes)
weights = len(y_raw) / (num_classes * (class_counts + 1e-9))
class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

# Split 70% Train / 15% Val / 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

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
# 4. MODEL ARCHITECTURE
# ==========================================
class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MultiChannelCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256), nn.ReLU() # No Pool
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
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

model = MultiChannelCNN(num_classes=num_classes, input_channels=len(SENSORS_TO_USE)).to(DEVICE)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(EPOCHS):
    # Train
    model.train()
    r_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        r_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    t_loss = r_loss / len(train_loader)
    t_acc = 100 * correct / total
    train_losses.append(t_loss)
    train_accs.append(t_acc)

    # Validate
    model.eval()
    r_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            r_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    v_loss = r_loss / len(val_loader)
    v_acc = 100 * correct / total
    val_losses.append(v_loss)
    val_accs.append(v_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} ({t_acc:.1f}%) | Val Loss: {v_loss:.4f} ({v_acc:.1f}%)")

total_time = time.time() - start_time
print(f"Training finished in {total_time/60:.2f} minutes.")

# ==========================================
# 6. SAVE RESULTS (Model, Matrix, Report)
# ==========================================
timestamp = time.strftime("%Y%m%d-%H%M%S")

# --- A. Generate Predictions ---
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

# --- B. Save Classification Report ---
report_text = classification_report(all_labels, all_preds, target_names=valid_classes)
report_dict = classification_report(all_labels, all_preds, target_names=valid_classes, output_dict=True)

report_path = os.path.join(RESULTS_DIR, f"report_{timestamp}.txt")
with open(report_path, "w") as f:
    f.write(f"Model: MultiChannelCNN\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n")
    f.write("-" * 30 + "\n")
    f.write(report_text)
print(f"Report saved to: {report_path}")

# --- C. Save Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_classes, yticklabels=valid_classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path)
print(f"Confusion Matrix saved to: {cm_path}")
plt.close() # Close plot to free memory

# --- D. Save Training Curves ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss Curves")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title("Accuracy Curves")

curves_path = os.path.join(RESULTS_DIR, f"training_curves_{timestamp}.png")
plt.savefig(curves_path)
plt.close()

# --- E. Save Model ---
final_acc = report_dict['accuracy'] * 100
model_filename = f"cnn_acc{final_acc:.1f}_{timestamp}.pth"
save_path = os.path.join(MODELS_DIR, model_filename)

save_dict = {
    'model_state_dict': model.state_dict(),
    'class_names': valid_classes,
    'input_channels': len(SENSORS_TO_USE),
    'window_size': WINDOW_SIZE,
    'metrics': report_dict
}
torch.save(save_dict, save_path)
print(f"Model saved to: {save_path}")