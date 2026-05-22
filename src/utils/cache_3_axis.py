import sys
import os
import torch
import numpy as np

# --- Path Setup to allow importing from utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.data_loader import load_mafulda_dataset
except ImportError:
    print("Error: Could not import 'load_mafulda_dataset'. Run this from src/notebooks/")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We select ONLY the sensors relevant to your ADXL357Z Project
# Index 0: rot_freq (Tachometer) - Useful for "Ground Truth" reference
# Index 1-3: ohang (Overhang) - This matches your ADXL357Z placement
SENSORS = ['rot_freq', 'ohang_x', 'ohang_y', 'ohang_z']
SENSORS_UNCHANGED = ['rot_freq', 'uhang_x', 'uhang_y', 'uhang_z']

# Output Path
# Saves to Graduation_Project_Root/data/vis_cache_3axis.pt
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_PATH = os.path.join(DATA_DIR, "vis_cache_ungh_3axis.pt")

# Create data dir if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("--- 1. Loading Raw Data (Downsampling to 4kHz) ---")
# original_fs=50000, target_fs=4000
# This simulates the exact data format your ESP32/FT232H will provide
X_raw, y_raw, class_names = load_mafulda_dataset(
    sensors=SENSORS_UNCHANGED, 
    window_size=1024,   # 1 second snapshot logic
    stride=1024,        # No overlap needed for cache
    original_fs=50000, 
    target_fs=4000 
)

print(f"Loaded Shape: {X_raw.shape}") # Expect (N, 4, 1024) -> 1 Tach + 3 Accel

print("--- 2. Selecting Representative Samples ---")
# We save 5 random examples per class for the Dashboard "Gallery"
samples_X = []
samples_y = []

found_counts = {i: 0 for i in range(len(class_names))}
SAMPLES_PER_CLASS = 5

# Shuffle indices to get random variety
indices = np.random.permutation(len(X_raw))

for i in indices:
    label = y_raw[i]
    if found_counts[label] < SAMPLES_PER_CLASS:
        samples_X.append(X_raw[i])
        samples_y.append(label)
        found_counts[label] += 1
        
    # Stop if we have enough of every class
    if all(count >= SAMPLES_PER_CLASS for count in found_counts.values()):
        break

# Convert to Tensors
data_dict = {
    "X": torch.FloatTensor(np.array(samples_X)),
    "y": torch.LongTensor(np.array(samples_y)),
    "class_names": class_names,
    "sensor_names": SENSORS_UNCHANGED,
    "fs": 4000
}

print(f"--- 3. Saving Cache ---")
torch.save(data_dict, SAVE_PATH)
print(f"[SUCCESS] Saved {len(samples_X)} samples to: {SAVE_PATH}")
print(f"Channels included: {SENSORS_UNCHANGED}")