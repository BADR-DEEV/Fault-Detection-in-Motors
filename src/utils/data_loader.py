import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import decimate

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_MAFULDA_DIR = BASE_DIR / "data" / "raw_mafulda"

# ==========================================
# 2. SELECTIVE FOLDER MAPPING (Curated List)
# ==========================================
# We map the Output Class Label -> The Specific Folder Path we want
SELECTED_PATHS = {
    "normal": "normal",
    "imbalance": "imbalance/6g",                        # Smallest weight = hardest to detect
    "horizontal_misalignment": "horizontal-misalignment/0.5mm", # Smallest misalignment
    "vertical_misalignment": "vertical-misalignment/0.51mm",    # Smallest misalignment
    # Optional: Add bearing fault if you want 5 classes
    # "bearing_fault": "underhang/outer_race/6g" 
}

# Map Class Strings to Integers
CLASS_TO_INT = {k: i for i, k in enumerate(SELECTED_PATHS.keys())}

# ==========================================
# 3. SELECTIVE LOADER FUNCTION
# ==========================================
def load_mafulda_dataset(
    sensors=['uhang_x', 'uhang_y', 'uhang_z'], # DEFAULTING TO UNDERHANG (Drive End)
    window_size=1024,
    stride=1024, # No overlap to reduce data leakage
    original_fs=50000,
    target_fs=4000
):
    X, y = [], []
    downsample_factor = int(original_fs / target_fs)
    
    print(f"\n[LOADER] Loading Balanced Dataset...")
    print(f"[INFO] Targeting Sensors: {sensors}")

    for class_name, sub_folder in SELECTED_PATHS.items():
        label_int = CLASS_TO_INT[class_name]
        
        # Construct full path to the specific subfolder
        # e.g. .../data/raw_mafulda/imbalance/6g
        folder_path = RAW_MAFULDA_DIR / sub_folder
        
        if not folder_path.exists():
            print(f"[ERROR] Path not found: {folder_path}")
            continue

        # Get all CSVs in that SPECIFIC folder only
        files = list(folder_path.glob("*.csv"))
        print(f"  -> Class '{class_name}': Found {len(files)} files in '{sub_folder}'")

        for file_path in files:
            try:
                # MAFAULDA usually has no headers, just 8 columns
                df = pd.read_csv(file_path, header=None)
                
                # Column mapping based on MAFAULDA documentation
                # 0: tach, 1-3: underhang accel, 4-6: overhang accel, 7: mic
                # uhang_x=1, uhang_y=2, uhang_z=3
                # ohang_x=4, ohang_y=5, ohang_z=6
                
                # Extract only the 3 columns we want
                # We need to map string names to column indices
                col_indices = []
                if 'uhang_x' in sensors: col_indices.append(1)
                if 'uhang_y' in sensors: col_indices.append(2)
                if 'uhang_z' in sensors: col_indices.append(3)
                if 'ohang_x' in sensors: col_indices.append(4)
                
                raw_signals = df.iloc[:, col_indices].values.astype(np.float32)

                # DOWNSAMPLE (50kHz -> 4kHz)
                signals = decimate(raw_signals, q=downsample_factor, axis=0, ftype='iir')

                # NORMALIZE (Z-Score)
                mean = signals.mean(axis=0)
                std = signals.std(axis=0) + 1e-9
                signals = (signals - mean) / std

                # SLICE INTO WINDOWS
                num_segments = (len(signals) - window_size) // stride
                for i in range(num_segments):
                    start = i * stride
                    end = start + window_size
                    # Transpose to (Channels, Time) -> (3, 1024)
                    segment = signals[start:end].T 
                    X.append(segment)
                    y.append(label_int)

            except Exception as e:
                print(f"[SKIP] Error reading {file_path.name}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    print(f"[RESULT] Final Dataset Shape: {X.shape}")
    print(f"[RESULT] Class mapping: {CLASS_TO_INT}")
    
    return X, y, list(CLASS_TO_INT.keys())


# import os
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.signal import decimate
# from scipy.io import loadmat

# # ==========================================
# # 1. DYNAMIC PATH CONFIGURATION
# # ==========================================
# # File location: .../Ai-Driven-Vibrations-motor/src/utils/data_loader.py

# # .resolve() = absolute path to this file
# # .parents[0] = utils
# # .parents[1] = src
# # .parents[2] = Ai-Driven-Vibrations-motor (PROJECT ROOT)
# BASE_DIR = Path(__file__).resolve().parents[2]

# DATA_ROOT = BASE_DIR / "data"
# RAW_MAFULDA_DIR = DATA_ROOT / "raw_mafulda"
# PAPER_DATA_DIR = DATA_ROOT / "paper_data"

# print(f"[DEBUG] Project Root: {BASE_DIR}")
# print(f"[DEBUG] Looking for Mafulda at: {RAW_MAFULDA_DIR}")

# # Check if path exists immediately
# if not RAW_MAFULDA_DIR.exists():
#     print(f"[ERROR] The path {RAW_MAFULDA_DIR} does not exist!")
#     print("Please check your folder structure.")

# # ==========================================
# # 2. CONSTANTS & CLASSES
# # ==========================================
# COL_NAMES = [
#     'rot_freq',
#     'uhang_x', 'uhang_y', 'uhang_z',
#     'ohang_x', 'ohang_y', 'ohang_z',
#     'microphone'
# ]

# DATA_CLASSES = {
#     "normal": 0,
#     "horizontal-misalignment": 1,
#     "vertical-misalignment": 2,
#     "overhang": 3,
#     "underhang": 4,
#     "imbalance": 5,
# }

# # ==========================================
# # 3. MAFAULDA LOADER
# # ==========================================
# def load_mafulda_dataset(
#     root_dir=None, # Set to None so it uses default
#     sensors=None,
#     window_size=1024,
#     stride=512,
#     original_fs=50000,
#     target_fs=4000
# ):
#     # Use the calculated default if user doesn't provide one
#     if root_dir is None:
#         root_dir = RAW_MAFULDA_DIR
#     else:
#         root_dir = Path(root_dir)

#     # if sensors is None:
#     #     sensors = ['uhang_x', 'uhang_y', 'uhang_z', 'ohang_x', 'ohang_y', 'ohang_z']

#     X, y = [], []
#     class_names = []
    
#     downsample_factor = int(original_fs / target_fs)
#     print(f"--- Loading Mafulda ---")
#     print(f"Path: {root_dir}")
#     print(f"Downsampling Factor: {downsample_factor}")

#     for class_name, label in DATA_CLASSES.items():
#         # Look for the class folder
#         class_path = root_dir / class_name
        
#         # NOTE: Your folder names in Windows might be "normal", "imbalance", etc.
#         # But sometimes extracted data has subfolders. We check existence first.
#         if not class_path.exists():
#             print(f"[Warning] Folder not found: {class_path}")
#             continue

#         class_names.append(class_name)
        
#         # RECURSIVE SEARCH (Finds CSVs even inside sub-folders like '6g', '10g')
#         file_paths = []
#         for root, _, files in os.walk(class_path):
#             for file in files:
#                 if file.endswith(".csv"):
#                     file_paths.append(os.path.join(root, file))

#         if not file_paths:
#             print(f"[Warning] No CSVs found in {class_name}")
#             continue

#         print(f"Processing '{class_name}': {len(file_paths)} files.")

#         for file_path in file_paths:
#             try:
#                 df = pd.read_csv(file_path, names=COL_NAMES)
#                 if df.empty: continue
                
#                 raw_signals = df[sensors].values.astype(np.float32)

#                 # DOWNSAMPLE
#                 signals = decimate(raw_signals, q=downsample_factor, axis=0, ftype='iir')

#                 # NORMALIZE
#                 mean = signals.mean(axis=0)
#                 std = signals.std(axis=0) + 1e-9
#                 signals = (signals - mean) / std

#                 # SLICE
#                 num_segments = (len(signals) - window_size) // stride
                
#                 if num_segments < 0: continue

#                 for i in range(num_segments + 1):
#                     start = i * stride
#                     end = start + window_size
#                     segment = signals[start:end].T 
#                     X.append(segment)
#                     y.append(label)

#             except Exception as e:
#                 print(f"[Error] {file_path}: {e}")

#     return np.array(X), np.array(y), list(DATA_CLASSES.keys())