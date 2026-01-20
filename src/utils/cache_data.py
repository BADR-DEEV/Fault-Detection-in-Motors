import os
import torch
import numpy as np
from utils.data_loader import load_mafulda_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
# WE ADD 'rot_freq' HERE TO GRAB THE TACHOMETER SIGNAL
SENSORS = ['rot_freq', 'uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z']

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "vis_cache_2.pt")

print("--- 1. Loading Raw Data (Including Tachometer) ---")
# Note: target_fs is 4000. 50k/4k = 12.5. Actual FS = 4166.6 Hz
X_raw, y_raw, class_names = load_mafulda_dataset(
    sensors=SENSORS, 
    window_size=1024, 
    stride=1024, 
    original_fs=50000, 
    target_fs=4000 
)

print("--- 2. Selecting Samples ---")
# We will save 5 examples per class
samples_X = []
samples_y = []

found_counts = {i: 0 for i in range(len(class_names))}
needed = 5

indices = np.random.permutation(len(X_raw))

for i in indices:
    label = y_raw[i]
    if found_counts[label] < needed:
        samples_X.append(X_raw[i])
        samples_y.append(label)
        found_counts[label] += 1
        
    if all(count >= needed for count in found_counts.values()):
        break

# Convert to Tensors
data_dict = {
    "X": torch.FloatTensor(np.array(samples_X)),
    "y": torch.LongTensor(np.array(samples_y)),
    "class_names": class_names
}

print(f"--- 3. Saving to {SAVE_PATH} ---")
torch.save(data_dict, SAVE_PATH)
print("Done! Cache updated with Tachometer data.")