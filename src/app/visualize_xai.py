import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_mafulda_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THIS PATH TO YOUR SAVED MODEL FILE
# Check src/saved_models/ for the exact name
MODEL_FILENAME = "cnn_acc99.8_20260118-205008.pth"

# Go up THREE levels: training → src → project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME)

print(f"Model Path: {MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENSORS = ['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z']

# ==========================================
# 2. DEFINE MODEL ARCHITECTURE 
# (Must match the saved model exactly)
# ==========================================
class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MultiChannelCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(input_channels, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 5, 1, 2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
        self.layer5 = nn.Sequential(nn.Conv1d(256, 256, 7, 1, 3), nn.BatchNorm1d(256), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv1d(256, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(32 * 64, 128), nn.ReLU(), nn.Dropout(0.5))
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

# ==========================================
# 3. LOAD DATA (Just enough to find samples)
# ==========================================
print("--- Loading Data for Visualization ---")
X_raw, y_raw, class_names = load_mafulda_dataset(
    sensors=SENSORS,
    window_size=1024,
    stride=2048, # Big stride = load fast, we only need a few samples
    original_fs=50000,
    target_fs=4000
)

# ==========================================
# 4. LOAD MODEL
# ==========================================
print(f"--- Loading Model: {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}\nPlease check the filename.")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = MultiChannelCNN(num_classes=len(class_names), input_channels=6).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 5. GENERATE SALIENCY MAPS
# ==========================================
def plot_saliency(model, X, y, class_names):
    found_classes = set()
    samples = []
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    print("Searching for one example per class...")
    indices = np.random.permutation(len(X)) # Shuffle search
    
    for i in indices:
        if len(found_classes) == len(class_names): break
        
        label_idx = y_tensor[i].item()
        if label_idx in found_classes: continue
            
        input_data = X_tensor[i].unsqueeze(0).to(DEVICE) # (1, 6, 1024)
        input_data.requires_grad_()
        
        output = model(input_data)
        pred_score, pred_label = torch.max(output, 1)
        
        if pred_label.item() == label_idx: # Only plot if prediction is correct
            pred_score.backward()
            saliency = input_data.grad.data.abs().squeeze().cpu().numpy()
            samples.append((X[i], saliency, label_idx))
            found_classes.add(label_idx)

    # Plot
    print("Generating Plots...")
    fig, axes = plt.subplots(len(samples), 2, figsize=(15, 3 * len(samples)))
    
    for idx, (signal, saliency, label_idx) in enumerate(sorted(samples, key=lambda x: x[2])):
        cls_name = class_names[label_idx]
        
        # We plot the first sensor (uhang_x)
        sensor_data = signal[0] 
        # We take max saliency across all 6 sensors to see global importance
        importance = np.max(saliency, axis=0) 
        
        # Normalize importance for better coloring (0 to 1)
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)

        # Plot 1: Raw Signal
        ax1 = axes[idx][0]
        ax1.plot(sensor_data, 'b')
        ax1.set_title(f"Class: {cls_name} (Raw Signal - U_Hang X)")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: XAI Heatmap
        ax2 = axes[idx][1]
        ax2.plot(sensor_data, 'k', alpha=0.3) # Faint background
        im = ax2.scatter(
            np.arange(1024), 
            sensor_data, 
            c=importance, 
            cmap='jet', 
            s=10
        )
        ax2.set_title(f"Saliency Map (Red = AI Focus)")
        plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()

plot_saliency(model, X_raw, y_raw, class_names)