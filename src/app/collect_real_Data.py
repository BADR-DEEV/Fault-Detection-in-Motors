import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import decimate

# CONFIG
BASE_DIR = Path(__file__).resolve().parents[2] 
DATA_DIR = BASE_DIR / "data" / "raw_mafulda"
OUTPUT_FILE = "dashboard_data.json"

# We want one sample file for each class to show in the dashboard
SAMPLES = {
    "normal": "normal/normal_1797.csv",  # Adjust filename if needed
    "imbalance": "imbalance/20g/imbalance_20g_1797.csv",
    "horizontal_misalignment": "horizontal-misalignment/2.0mm/horiz_2.0mm_1797.csv",
    "vertical_misalignment": "vertical-misalignment/1.90mm/vert_1.90mm_1797.csv"
}

export_data = {}

print("Extracting real 0.25 HP motor data...")

for label, subpath in SAMPLES.items():
    # Find the first file that matches the pattern if exact name varies
    folder = DATA_DIR / Path(subpath).parent
    if not folder.exists():
        print(f"Skipping {label} (Folder not found: {folder})")
        continue
        
    # Get first CSV in folder
    file_path = list(folder.glob("*.csv"))[0]
    
    # Read Data
    df = pd.read_csv(file_path, header=None)
    # Columns: 1=X, 2=Y, 3=Z
    raw = df.iloc[:, [1, 2, 3]].values.astype(np.float32)

    # 1. Downsample (50k -> ~4166 Hz) to match your training
    sig = decimate(raw, q=12, axis=0, ftype='iir')

    # 2. Normalize (CRITICAL: Use same logic as training)
    # (x - mean) / std
    sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-6)

    # 3. Take 1024 points (Window size)
    segment = sig[:1024]

    export_data[label] = {
        "ax": segment[:, 0].tolist(),
        "ay": segment[:, 1].tolist(),
        "az": segment[:, 2].tolist()
    }
    print(f" -> Processed {label}")

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(export_data, f)

print(f"\n[SUCCESS] Data saved to {OUTPUT_FILE}")
print("Copy the content of this file into your React App.")