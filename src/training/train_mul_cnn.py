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

from utils.data_loader import load_mafulda_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
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


ROOT_MAFULDA = "../../data/raw_mafulda/"

SENSORS_TO_USE = [
    'uhang_x', 'uhang_y', 'uhang_z',
    'ohang_x', 'ohang_y', 'ohang_z'
]

X_raw, y_raw, valid_classes = load_mafulda_dataset(
    sensors=SENSORS_TO_USE,
    window_size=1024,
    stride=64, 
    target_fs=4000 
)

print(f"Loaded data: {X_raw.shape}")  # (N, 6, 1024)
num_classes = len(valid_classes)


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




# ... (After the Classification Report print statement) ...

# ==========================================
# 8. XAI: SALIENCY MAP VISUALIZATION
# ==========================================
def compute_saliency_maps(model, X, y, class_names, device):
    """
    Computes and plots Saliency Maps for one example of each class.
    """
    model.eval()
    
    # We want to find one correct example for each class to visualize
    found_classes = set()
    samples_to_plot = []

    # Iterate through test set to find one example per class
    for i in range(len(X)):
        if len(found_classes) == len(class_names):
            break
            
        img, label = X[i], y[i]
        label_int = label.item()
        
        # Skip if we already found an example for this class
        if label_int in found_classes:
            continue

        # Prepare input
        input_tensor = img.unsqueeze(0).to(device) # Add batch dim: (1, 6, 1024)
        input_tensor.requires_grad_() # CRITICAL: Tell PyTorch to track gradients for the INPUT
        
        # Forward pass
        output = model(input_tensor)
        score, predicted = torch.max(output, 1)

        # Only use correctly classified examples for explanation
        if predicted.item() == label_int:
            # Backward pass to get gradients
            score.backward()
            
            # Get the gradient (Saliency)
            # Shape: (1, 6, 1024) -> squeeze to (6, 1024)
            saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
            original_signal = img.numpy()
            
            samples_to_plot.append((original_signal, saliency, label_int))
            found_classes.add(label_int)

    # Plotting
    fig, axes = plt.subplots(len(samples_to_plot), 2, figsize=(15, 4 * len(samples_to_plot)))
    if len(samples_to_plot) == 1: axes = [axes] # Handle case of single plot

    for idx, (signal, saliency, label_idx) in enumerate(samples_to_plot):
        class_name = class_names[label_idx]
        
        # 1. We combine the saliency across all 6 sensors to see "Global Importance" per time step
        # You can also pick just one sensor (e.g., saliency[0])
        combined_saliency = np.max(saliency, axis=0) # Take max influence across 6 sensors
        
        # Just plot the first sensor (uhang_x) for clarity, but color it with importance
        sensor_idx = 0 
        sensor_name = SENSORS_TO_USE[sensor_idx]
        signal_data = signal[sensor_idx]

        # Plot A: The Original Signal
        ax1 = axes[idx][0]
        ax1.plot(signal_data, color='blue', alpha=0.6, label='Raw Signal')
        ax1.set_title(f"Class: {class_name} | Sensor: {sensor_name}")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # Plot B: The Saliency Map (Where the AI looked)
        ax2 = axes[idx][1]
        ax2.plot(signal_data, color='lightgray', alpha=0.5) # Faint background signal
        
        # Overlay the heatmap
        # We scatter plot points, coloring them by how important they are
        im = ax2.scatter(
            range(len(signal_data)), 
            signal_data, 
            c=combined_saliency, 
            cmap='hot', 
            s=10,  # Dot size
            label='Importance'
        )
        ax2.set_title(f"Saliency Map (Red = High Importance)")
        plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()

# Run the visualization using the Test Dataset
print("\n--- Generating Saliency Maps (XAI) ---")
# Get the raw dataset from the loader wrapper
test_dataset_obj = VibrationDataset(X_test, y_test)
compute_saliency_maps(model, test_dataset_obj, test_dataset_obj.y, valid_classes, DEVICE)


# ==========================================
# 9. SAVE MODEL
# ==========================================
import time

def save_model(model, class_names, accuracy):
    # 1. Define the save directory
    # This creates: src/saved_models/
    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Create a unique filename with accuracy and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mafulda_cnn_acc{accuracy:.1f}_{timestamp}.pth"
    save_path = os.path.join(save_dir, filename)

    # 3. Create the save dictionary
    # We save the weights AND the class names so we can load them easily later
    save_dict = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_channels': len(SENSORS_TO_USE),
        'window_size': WINDOW_SIZE,
        'metrics': classification_report(all_labels, all_preds, target_names=valid_classes, output_dict=True)
    }

    # 4. Save to disk
    torch.save(save_dict, save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}")

# Calculate final accuracy from the test run
# (all_preds and all_labels come from the 'Final Test' section above)
correct_predictions = np.sum(np.array(all_preds) == np.array(all_labels))
final_acc = (correct_predictions / len(all_labels)) * 100

# Execute Save
save_model(model, valid_classes, final_acc)