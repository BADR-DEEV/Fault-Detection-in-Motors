import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = "cnn_underhang_acc100.0_20260125-115627.pth" # <--- CHECK THIS NAME

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME)
DATA_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "vis_cache_3axis.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actual Sampling Rate (50000 / 12)
FS = 4000 

# ==========================================
# 2. MODEL ARCHITECTURE
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
# 3. HELPER FUNCTIONS
# ==========================================
def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(signal)) / n 
    return freqs, mag

def compute_statistics(signal):
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    peak = np.max(np.abs(signal))
    crest = peak / rms if rms > 0 else 0
    return rms, kurt, crest

def compute_gradcam(model, input_tensor, target_class):
    target_layer = model.layer6[0] 
    gradients, activations = [], []
    def bh(m, gi, go): gradients.append(go[0])
    def fh(m, i, o): activations.append(o)
    h1 = target_layer.register_backward_hook(bh)
    h2 = target_layer.register_forward_hook(fh)
    
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()
    
    grads = gradients[0].cpu().data.numpy()[0]
    fmaps = activations[0].cpu().data.numpy()[0]
    h1.remove(); h2.remove()
    
    weights = np.mean(grads, axis=1)
    cam = np.zeros(fmaps.shape[1], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * fmaps[i]
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
    return np.interp(np.linspace(0, 1024, 1024), np.linspace(0, 1024, 64), cam)

# ==========================================
# 4. LOAD RESOURCES
# ==========================================
print("1. Loading Cached Data...")
if not os.path.exists(DATA_CACHE_PATH):
    print("Error: Run 'src/utils/cache_data_2.py' first!")
    exit()
    
cache = torch.load(DATA_CACHE_PATH)
X_data = cache['X'] # Now shape is (N, 7, 1024)
y_data = cache['y']
class_names = cache['class_names']

print("2. Loading Model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = MultiChannelCNN(num_classes=len(class_names), input_channels=3).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# ==========================================
# 5. MAIN PLOTTING LOOP (CORRECTED)
# ==========================================
found_classes = set()

# Defined by the MAFAULDA Paper
MIN_HZ = 10.0  # ~600 RPM
MAX_HZ = 65.0  # ~3900 RPM (Paper max is 3686, we give a little buffer)

for i in range(len(X_data)):
    label = y_data[i].item()
    if label in found_classes: continue
    
    class_name = class_names[label]
    print(f"Analyzing {class_name}...")
    
    # --- DATA PREP ---
    raw_sample = X_data[i]
    tach_signal = raw_sample[0].numpy()
    vib_tensor = raw_sample[1:].unsqueeze(0).to(DEVICE)
    vib_tensor.requires_grad = True
    
    # --- 1. CALCULATE TRUE RPM (WITH PHYSICS LIMITS) ---
    tach_freqs, tach_mags = compute_fft(tach_signal, FS)
    
    # CRITICAL FIX: Only look for peaks within the valid motor range (10-65 Hz)
    # This prevents picking up the 2nd or 3rd harmonic of the tachometer pulse
    valid_indices = np.where((tach_freqs >= MIN_HZ) & (tach_freqs <= MAX_HZ))[0]
    
    if len(valid_indices) > 0:
        # Find index of the highest peak WITHIN the valid range
        local_peak = np.argmax(tach_mags[valid_indices])
        global_peak = valid_indices[local_peak]
        true_hz = tach_freqs[global_peak]
    else:
        true_hz = 0 # Should not happen
        
    true_rpm = true_hz * 60
    
    # --- 2. VIBRATION ANALYSIS ---
    cam = compute_gradcam(model, vib_tensor, label)
    
    # Use Sensor 1 (uhang_x)
    vib_signal = raw_sample[1].numpy() 
    vib_freqs, vib_mags = compute_fft(vib_signal, FS)
    rms, kurt, crest = compute_statistics(vib_signal)
    
    # --- 3. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Diagnosis: {class_name} | True Speed: {true_rpm:.0f} RPM ({true_hz:.1f} Hz)", 
                 fontsize=14, fontweight='bold')
    
    # Plot Time Domain
    ax1.plot(vib_signal, 'k', alpha=0.6, label="Vibration (U-Hang X)")
    im = ax1.imshow(cam.reshape(1, -1), aspect='auto', cmap='jet', alpha=0.6, 
                   extent=[0, 1024, min(vib_signal), max(vib_signal)])
    ax1.set_title("Vibration Signal & AI Focus Areas")
    ax1.legend(loc='upper right')
    
    # Plot Frequency Domain
    ax2.plot(vib_freqs, vib_mags, 'b', label="Vibration Spectrum")
    
    # Plot Ground Truth Lines (1x, 2x, 3x)
    ax2.axvline(x=true_hz, color='r', linestyle='--', linewidth=2, label=f"True 1x ({true_hz:.1f}Hz)")
    ax2.axvline(x=true_hz*2, color='g', linestyle='--', alpha=0.5, label=f"True 2x")
    ax2.axvline(x=true_hz*3, color='g', linestyle='--', alpha=0.5, label=f"True 3x")

    ax2.set_title("Frequency Spectrum vs Ground Truth RPM")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, 500) # Zoom in to see the harmonics clearly
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Stats Box
    stats_text = (
        f"RMS: {rms:.2f}\n"
        f"Kurtosis: {kurt:.2f}\n"
        f"Crest: {crest:.2f}\n"
        f"True RPM: {true_rpm:.0f}"
    )
    fig.text(0.80, 0.90, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    found_classes.add(label)


























# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import kurtosis

# # ==========================================
# # 1. CONFIGURATION
# # ==========================================

# MODEL_FILENAME = "cnn_acc99.8_20260118-205008.pth" 

# # Fix: Use __file__ for script path
# try:
#     PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# except NameError:
#     # Fallback if running in Jupyter/Colab
#     PROJECT_ROOT = os.getcwd()

# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME)
# DATA_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "vis_cache_2.pt")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # FIX: Precision Sampling Rate based on text (50kHz / 12)
# # If FS is wrong, calculated RPM will be wrong.
# FS = 50000 / 12  # approx 4166.67 Hz

# # ==========================================
# # 2. MODEL ARCHITECTURE
# # ==========================================

# class MultiChannelCNN(nn.Module):
#     # Fix: __init__ syntax
#     def __init__(self, num_classes, input_channels):
#         super(MultiChannelCNN, self).__init__()
        
#         # Architecture matching the logic provided
#         self.layer1 = nn.Sequential(nn.Conv1d(input_channels, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
#         self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
#         self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 5, 1, 2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
#         self.layer4 = nn.Sequential(nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
#         self.layer5 = nn.Sequential(nn.Conv1d(256, 256, 7, 1, 3), nn.BatchNorm1d(256), nn.ReLU())
#         self.layer6 = nn.Sequential(nn.Conv1d(256, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU())
        
#         self.flatten = nn.Flatten()
        
#         # Calculation: Input 1024 -> /2 (L1) -> /2 (L2) -> /2 (L3) -> /2 (L4) = 64 width
#         # 32 channels * 64 width = 2048
#         self.fc1 = nn.Sequential(nn.Linear(32 * 64, 128), nn.ReLU(), nn.Dropout(0.5))
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# # ==========================================
# # 3. HELPER FUNCTIONS
# # ==========================================

# def compute_fft(signal, fs, padding_multiplier=4):
#     """
#     Computes FFT. 
#     Padding added to improve visual peak interpolation for RPM estimation
#     since 1024 points @ 4kHz has poor resolution (~4Hz).
#     """
#     n = len(signal)
#     n_padded = n * padding_multiplier 
#     freqs = np.fft.rfftfreq(n_padded, d=1/fs)
#     mag = np.abs(np.fft.rfft(signal, n=n_padded)) / n
#     return freqs, mag

# def compute_statistics(signal):
#     rms = np.sqrt(np.mean(signal**2))
#     kurt = kurtosis(signal)
#     peak = np.max(np.abs(signal))
#     crest = peak / rms if rms > 0 else 0
#     return rms, kurt, crest

# def compute_gradcam(model, input_tensor, target_class):
#     target_layer = model.layer6[0]
#     gradients = []
#     activations = []
    
#     # Hooks
#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])
        
#     def forward_hook(module, input, output):
#         activations.append(output)
        
#     # Fix: Use register_full_backward_hook for compatibility
#     h1 = target_layer.register_full_backward_hook(backward_hook)
#     h2 = target_layer.register_forward_hook(forward_hook)

#     model.zero_grad()
#     output = model(input_tensor)
    
#     # Backward pass for specific class
#     score = output[0, target_class]
#     score.backward()

#     # Get data
#     grads = gradients[0].cpu().data.numpy()[0] # [Channels, Length]
#     fmaps = activations[0].cpu().data.numpy()[0] # [Channels, Length]
    
#     h1.remove()
#     h2.remove()

#     # GAP (Global Average Pooling) on gradients
#     weights = np.mean(grads, axis=1)
    
#     # Weighted combination
#     cam = np.zeros(fmaps.shape[1], dtype=np.float32)
#     for i, w in enumerate(weights):
#         cam += w * fmaps[i]
        
#     # ReLU
#     cam = np.maximum(cam, 0)
    
#     # Normalize
#     if np.max(cam) > 0:
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
        
#     # Resize CAM to match input signal length (1024)
#     # Layer 6 output width is 64, Input is 1024
#     return np.interp(np.linspace(0, 1024, 1024), np.linspace(0, 1024, 64), cam)

# # ==========================================
# # 4. LOAD RESOURCES
# # ==========================================

# print("1. Loading Cached Data...")
# if not os.path.exists(DATA_CACHE_PATH):
#     print(f"Error: File not found at {DATA_CACHE_PATH}")
#     # exit() # Commented out to prevent notebook crash

# # Assuming cache exists for this context
# if os.path.exists(DATA_CACHE_PATH):
#     cache = torch.load(DATA_CACHE_PATH, map_location=DEVICE)
#     X_data = cache['X'] 
#     y_data = cache['y']
#     class_names = cache['class_names']

#     print(f"2. Loading Model from {MODEL_FILENAME}...")
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
#     # Input channels = 6 (7 total - 1 tachometer)
#     model = MultiChannelCNN(num_classes=len(class_names), input_channels=6).to(DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # ==========================================
#     # 5. MAIN PLOTTING LOOP
#     # ==========================================

#     found_classes = set()

#     # Defined by the MAFAULDA Paper
#     MIN_HZ = 10.0  # ~600 RPM
#     MAX_HZ = 65.0  # ~3900 RPM (Paper max is 3686)

#     print("3. Starting Analysis...")

#     for i in range(len(X_data)):
#         label = y_data[i].item()
        
#         # Only process one example per class
#         if label in found_classes: 
#             continue

#         class_name = class_names[label]
#         print(f"Analyzing Class: {class_name}...")

#         # --- DATA PREP ---
#         raw_sample = X_data[i]
        
#         # Assumes Channel 0 is Tachometer, Channels 1-6 are Vibration
#         tach_signal = raw_sample[0].numpy()
#         vib_tensor = raw_sample[1:].unsqueeze(0).to(DEVICE) # Shape [1, 6, 1024]
#         vib_tensor.requires_grad = True

#         # --- 1. CALCULATE TRUE RPM ---
#         # Using padding_multiplier to get smoother peaks
#         tach_freqs, tach_mags = compute_fft(tach_signal, FS, padding_multiplier=8)

#         # Filter for valid motor range (10-65 Hz)
#         valid_indices = np.where((tach_freqs >= MIN_HZ) & (tach_freqs <= MAX_HZ))[0]

#         if len(valid_indices) > 0:
#             local_peak = np.argmax(tach_mags[valid_indices])
#             global_peak = valid_indices[local_peak]
#             true_hz = tach_freqs[global_peak]
#             true_rpm = true_hz * 60
#         else:
#             true_hz = 0
#             true_rpm = 0
            
#         # --- 2. VIBRATION ANALYSIS ---
#         cam = compute_gradcam(model, vib_tensor, label)

#         # Use Sensor 1 (e.g., Underhang X) for visualization
#         vib_signal = raw_sample[1].numpy() 
#         vib_freqs, vib_mags = compute_fft(vib_signal, FS) # Standard FFT for viz
#         rms, kurt, crest = compute_statistics(vib_signal)

#         # --- 3. PLOTTING ---
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#         fig.suptitle(f"Diagnosis: {class_name} | Est. Speed: {true_rpm:.0f} RPM ({true_hz:.2f} Hz)", 
#                      fontsize=14, fontweight='bold')

#         # Plot Time Domain with GradCAM overlay
#         ax1.plot(vib_signal, 'k', alpha=0.6, label="Vibration (Sensor 1)")
        
#         # Extent matches x-axis (0 to 1024) and y-axis (min/max of signal)
#         im = ax1.imshow(cam.reshape(1, -1), aspect='auto', cmap='jet', alpha=0.6, 
#                        extent=[0, 1024, np.min(vib_signal), np.max(vib_signal)])
        
#         ax1.set_title("Time Domain Signal & CNN Attention (GradCAM)")
#         ax1.set_xlabel("Time (Samples)")
#         ax1.set_ylabel("Amplitude")
#         ax1.legend(loc='upper right')

#         # Plot Frequency Domain
#         ax2.plot(vib_freqs, vib_mags, 'b', label="Vibration Spectrum")

#         # Plot Ground Truth Lines (1x, 2x, 3x Harmonics)
#         if true_hz > 0:
#             ax2.axvline(x=true_hz, color='r', linestyle='--', linewidth=2, label=f"1x ({true_hz:.1f}Hz)")
#             ax2.axvline(x=true_hz*2, color='g', linestyle='--', alpha=0.5, label=f"2x")
#             ax2.axvline(x=true_hz*3, color='g', linestyle='--', alpha=0.5, label=f"3x")

#         ax2.set_title("Frequency Spectrum")
#         ax2.set_xlabel("Frequency (Hz)")
#         ax2.set_ylabel("Magnitude")
#         ax2.set_xlim(0, 500) # Zoom to see lower harmonics
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)

#         # Stats Box
#         stats_text = (
#             f"RMS: {rms:.2f}\n"
#             f"Kurtosis: {kurt:.2f}\n"
#             f"Crest Factor: {crest:.2f}\n"
#         )
#         fig.text(0.80, 0.85, stats_text, fontsize=10, 
#                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

#         found_classes.add(label)