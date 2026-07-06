import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Path Setup to allow importing from utils ---
# This allows running the script from the root directory or src/training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.data_loader import load_mafulda_dataset
except ImportError:
    print("Error: Could not import 'load_mafulda_dataset'. Ensure you are running from the project root or src folder.")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION & SENSOR SELECTION
# ==========================================

# Define available sensor groups
SENSOR_GROUPS = {
    "overhang":   ['ohang_x', 'ohang_y', 'ohang_z'],           # <--- USE THIS FOR YOUR ADXL357Z PROJECT
    "underhang":  ['uhang_x', 'uhang_y', 'uhang_z'],
    "microphone": ['microphone'],
    "all_accel":  ['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z'],
    "all_sensors":['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z', 'microphone']
}

# --- USER SETTING: CHOOSE YOUR CONFIG HERE ---
CURRENT_CONFIG = "underhang"  # Options: "overhang", "underhang", "microphone", etc.
SENSORS_TO_USE = SENSOR_GROUPS[CURRENT_CONFIG]

# Hyperparameters
WINDOW_SIZE = 1024
STRIDE = 512
BATCH_SIZE = 64
EPOCHS = 20  # Increased slightly for better convergence
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- CONFIGURATION ---")
print(f"Target Device: {DEVICE}")
print(f"Selected Mode: {CURRENT_CONFIG}")
print(f"Active Sensors ({len(SENSORS_TO_USE)}): {SENSORS_TO_USE}")

# ==========================================
# 2. DATA LOADING
# ==========================================
# Assuming data_loader.py handles the column extraction based on the list we pass
X_raw, y_raw, valid_classes = load_mafulda_dataset(
    sensors=SENSORS_TO_USE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    target_fs=4000 # Downsampling to 4kHz to match ADXL357Z
)

if len(X_raw) == 0:
    raise ValueError("No data loaded. Check paths in data_loader.py")

print(f"Dataset Shape: {X_raw.shape}") # (Samples, Channels, Window)

# ==========================================
# 3. IMBALANCE HANDLING & SPLITTING
# ==========================================
num_classes = len(valid_classes)
class_counts = np.bincount(y_raw, minlength=num_classes)
weights = len(y_raw) / (num_classes * (class_counts + 1e-9))
class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

# Split: 70% Train, 15% Val, 15% Test
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
# 4. DYNAMIC CNN MODEL
# ==========================================
class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MultiChannelCNN, self).__init__()
        
        # Layer 1: Dynamic Input Channels
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 5 (No Pool)
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Layer 6
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # Linear Layers
        # Calculation: Input 1024 -> Pool(2)x4 times = 1024 / 16 = 64 length
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

# Initialize Model with Dynamic Input Channels
model = MultiChannelCNN(num_classes=num_classes, input_channels=len(SENSORS_TO_USE)).to(DEVICE)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch():
    model.train()
    correct = 0; total = 0; r_loss = 0
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
    return r_loss / len(train_loader), 100 * correct / total

def validate():
    model.eval()
    correct = 0; total = 0; r_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            r_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return r_loss / len(val_loader), 100 * correct / total

print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = validate()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} ({t_acc:.1f}%) | Val Loss: {v_loss:.4f} ({v_acc:.1f}%)")

# ==========================================
# 6. EVALUATION & REPORT
# ==========================================
model.eval()
all_preds = []; all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

correct_predictions = np.sum(np.array(all_preds) == np.array(all_labels))
final_acc = (correct_predictions / len(all_labels)) * 100

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=valid_classes))
# ==========================================
# 7. XAI: SALIENCY MAPS (FIXED)
# ==========================================
def compute_saliency_maps(model, X, y, class_names, device):
    model.eval()
    found_classes = set()
    samples_to_plot = []

    # Iterate through the data
    for i in range(len(X)):
        if len(found_classes) == len(class_names): break
        
        # FIX: Directly access the tensor data
        # We assume X is a Tensor [N, Channels, Window]
        img = X[i] 
        label = y[i]
        label_int = label.item()
        
        if label_int in found_classes: continue

        # Prepare input for Gradient Calculation
        input_tensor = img.unsqueeze(0).to(device) # Shape: (1, 3, 1024)
        input_tensor.requires_grad_() # CRITICAL for Saliency
        
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

        # Only map correctly classified examples
        if predicted.item() == label_int:
            # Backward pass to get the gradient
            score = output[0, label_int]
            score.backward()
            
            # Get gradient (Saliency)
            saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
            original_signal = img.numpy()
            
            samples_to_plot.append((original_signal, saliency, label_int))
            found_classes.add(label_int)

    # Plotting Logic
    fig, axes = plt.subplots(len(samples_to_plot), 2, figsize=(15, 3 * len(samples_to_plot)))
    if len(samples_to_plot) == 1: axes = np.expand_dims(axes, axis=0)

    for idx, (signal, saliency, label_idx) in enumerate(samples_to_plot):
        class_name = class_names[label_idx]
        
        # Combine saliency across X, Y, Z (Max importance)
        if saliency.ndim > 1:
            combined_saliency = np.max(saliency, axis=0)
        else:
            combined_saliency = saliency

        # Plot Channel 0 (Overhang X)
        signal_data = signal[0] 
        sensor_name = SENSORS_TO_USE[0]

        # Left Plot: Raw Signal
        axes[idx][0].plot(signal_data, color='blue', alpha=0.6)
        axes[idx][0].set_title(f"Class: {class_name} | Sensor: {sensor_name}")
        axes[idx][0].grid(True, alpha=0.3)
        
        # Right Plot: Saliency Map
        axes[idx][1].plot(signal_data, color='lightgray', alpha=0.5)
        # Scatter plot colored by importance
        im = axes[idx][1].scatter(
            range(len(signal_data)), 
            signal_data, 
            c=combined_saliency, 
            cmap='hot', 
            s=5
        )
        axes[idx][1].set_title("AI Attention Map")
        plt.colorbar(im, ax=axes[idx][1])
    
    plt.tight_layout()
    plt.show()

print("\n--- Generating Saliency Maps ---")

# FIX: Convert numpy arrays to Torch Tensors explicitly before passing
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Pass the raw tensors, NOT the Dataset object
compute_saliency_maps(model, X_test_tensor, y_test_tensor, valid_classes, DEVICE)


# ==========================================
# 8. SAVE MODEL
# ==========================================
def save_model(model, class_names, accuracy):
    # Save to /models folder relative to this script
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"cnn_{CURRENT_CONFIG}_acc{accuracy:.1f}_{timestamp}.pth"
    save_path = os.path.join(save_dir, filename)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config_mode': CURRENT_CONFIG,
        'sensor_names': SENSORS_TO_USE,
        'window_size': WINDOW_SIZE,
        'metrics': classification_report(all_labels, all_preds, target_names=valid_classes, output_dict=True)
    }

    torch.save(save_dict, save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}")

save_model(model, valid_classes, final_acc)