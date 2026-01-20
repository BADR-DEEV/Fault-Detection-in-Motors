import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_mafulda_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = "cnn_acc99.8_20260118-205008.pth" # <--- CHECK THIS NAME

# Define Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME)
XAI_SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "xai_plots")

os.makedirs(XAI_SAVE_DIR, exist_ok=True)
print(f"Images will be saved to: {XAI_SAVE_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENSORS = ['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z']

# ==========================================
# 2. MODEL ARCHITECTURE (EXACT COPY FROM TRAINING)
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
        # LAYER 6: This contains the FINAL Convolution
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

# ==========================================
# 3. XAI FUNCTIONS
# ==========================================

# --- A. Saliency Map ---
def compute_saliency(model, input_tensor, target_class):
    model.eval()
    input_tensor.requires_grad_()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    return np.max(saliency, axis=0)

# --- B. Grad-CAM (FIXED FOR ORIGINAL ARCHITECTURE) ---
def compute_gradcam(model, input_tensor, target_class):
    model.eval()
    
    # 1. FIND THE LAST CONV LAYER
    # In your code, the last conv is the first item inside self.layer6
    # layer6 = [Conv1d, BatchNorm, ReLU]
    # Index 0 = Conv1d
    target_layer = model.layer6[0] 

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
    def forward_hook(module, input, output): activations.append(output)

    # Attach hooks to the Conv1d layer inside layer6
    h1 = target_layer.register_backward_hook(backward_hook)
    h2 = target_layer.register_forward_hook(forward_hook)

    # Forward
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward
    score = output[0, target_class]
    score.backward()

    # Get hooked data
    grads = gradients[0].cpu().data.numpy()[0]
    fmaps = activations[0].cpu().data.numpy()[0]
    h1.remove(); h2.remove()

    # Calculate CAM
    weights = np.mean(grads, axis=1)
    cam = np.zeros(fmaps.shape[1], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * fmaps[i]
    
    # ReLU and Normalize
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
    
    # Resize to 1024
    # Note: Layer 6 output size is 64 (because of previous poolings)
    x = np.linspace(0, 1024, num=64) 
    x_new = np.linspace(0, 1024, num=1024)
    cam_resized = np.interp(x_new, x, cam)
    return cam_resized

# --- C. Occlusion Sensitivity ---
def compute_occlusion(model, input_tensor, target_class, window_size=50, stride=25):
    model.eval()
    width = input_tensor.shape[2]
    heatmap = np.zeros(width)
    counts = np.zeros(width)
    
    with torch.no_grad():
        baseline_score = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()

    for start in range(0, width - window_size, stride):
        end = start + window_size
        masked_input = input_tensor.clone()
        masked_input[:, :, start:end] = 0.0
        
        with torch.no_grad():
            score = torch.softmax(model(masked_input), dim=1)[0, target_class].item()
        
        importance = baseline_score - score
        heatmap[start:end] += importance
        counts[start:end] += 1

    heatmap /= (counts + 1e-9)
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    return heatmap

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
print("Loading Data...")
X_raw, y_raw, class_names = load_mafulda_dataset(
    sensors=SENSORS, window_size=1024, stride=4096, original_fs=50000, target_fs=4000
)

print(f"Loading Model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found. Check path.")
    exit()
    
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = MultiChannelCNN(num_classes=len(class_names), input_channels=6).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

found_classes = set()
indices = np.random.permutation(len(X_raw))

for idx in indices:
    if len(found_classes) == len(class_names): break
    
    label = y_raw[idx]
    if label in found_classes: continue
    
    class_name_str = class_names[label]
    print(f"Generating XAI for Class: {class_name_str}...")
    
    input_data = torch.FloatTensor(X_raw[idx]).unsqueeze(0).to(DEVICE)
    input_data.requires_grad = True
    
    out = model(input_data)
    pred = torch.argmax(out, 1).item()
    if pred != label: continue
    
    found_classes.add(label)
    
    # Calculate XAI
    saliency = compute_saliency(model, input_data, label)
    gradcam = compute_gradcam(model, input_data, label)
    occlusion = compute_occlusion(model, input_data, label)
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    signal = X_raw[idx][0] # uhang_x
    
    # 1. Raw
    axes[0].plot(signal, 'b')
    axes[0].set_title(f"Class: {class_name_str} (Raw Signal)")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Saliency
    axes[1].plot(signal, 'k', alpha=0.3)
    im1 = axes[1].scatter(np.arange(1024), signal, c=saliency, cmap='hot', s=5)
    axes[1].set_title("Saliency Map (Pixel Sensitivity)")
    plt.colorbar(im1, ax=axes[1])
    
    # 3. Grad-CAM
    axes[2].plot(signal, 'k', alpha=0.3)
    axes[2].imshow(gradcam.reshape(1, -1), aspect='auto', cmap='jet', alpha=0.7, 
                   extent=[0, 1024, min(signal), max(signal)])
    axes[2].set_title("Grad-CAM (Feature Activation)")
    
    # 4. Occlusion
    axes[3].plot(signal, 'k', alpha=0.3)
    axes[3].imshow(occlusion.reshape(1, -1), aspect='auto', cmap='autumn', alpha=0.7,
                   extent=[0, 1024, min(signal), max(signal)])
    axes[3].set_title("Occlusion Sensitivity (Masking Impact)")
    
    plt.tight_layout()
    
    # Save
    clean_name = class_name_str.replace("/", "_").replace("\\", "_")
    filename = f"xai_{clean_name}.png"
    save_path = os.path.join(XAI_SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    
    plt.show()