import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly, welch
from tqdm import tqdm

from training.RES_CNN import ResNet1D, compute_order_spectrum, safe_zscore

# --- 1. MODEL ARCHITECTURE ---
class ExplainableCNN(nn.Module):
    def __init__(self, input_bins=128, num_classes=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        
        final_width = input_bins // 8 
        linear_input_size = 64 * final_width
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.classifier(x)
        return x

# --- 2. CONFIGURATION ---
class InferenceConfig:
    def __init__(self):
        self.max_order = 6.0      
        self.order_bins = 128     
        self.dataset_fs = 4096    
        self.model_fs = 4000      
        self.window_sec = 1.0     
        self.classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]
        
        # 0D/0E = Normal, 1D-4E = Imbalance
        self.fault_map = {
            "0": "Normal",
            "1": "Imbalance",
            "2": "Imbalance",
            "3": "Imbalance",
            "4": "Imbalance"
        }

# def safe_zscore(x):
#     return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

# def compute_order_spectrum(sig, fs, rpm, cfg):
#     if rpm < 10: rpm = 1500.0 
#     shaft_hz = rpm / 60.0
#     nperseg = min(len(sig), 1024)
#     f, Pxx = welch(sig, fs=fs, nperseg=nperseg, axis=0)
#     orders = f / shaft_hz
#     target_orders = np.linspace(0, cfg.max_order, cfg.order_bins)
#     specs = []
#     for ch in range(sig.shape[1]):
#         s = np.interp(target_orders, orders, Pxx[:, ch], left=0, right=0)
#         s[:int(cfg.order_bins * 0.05)] = 0.0 
#         s_log = np.log(s + 1e-12)
#         s_norm = (s_log - s_log.mean()) / (s_log.std() + 1e-6)
#         specs.append(s_norm)
#     return np.stack(specs).astype(np.float32)

# --- 3. STREAMING LOGIC ---
def process_large_file(filepath, model, device, cfg, writer):
    filename = filepath.name
    file_id = filename[0] 
    ground_truth = cfg.fault_map.get(file_id, "Unknown")
    
    print(f"\nProcessing {filename} | Ground Truth: {ground_truth}")
    
    chunk_size = int(cfg.dataset_fs * 5) 
    stats = {k: 0 for k in cfg.classes}
    total_windows = 0
    
    with pd.read_csv(filepath, chunksize=chunk_size) as reader:
        for i, df_chunk in enumerate(tqdm(reader, desc=f"Scanning {filename}", unit="chunk")):
            
            rpm_raw = df_chunk.iloc[:, 1].values
            vib_raw = df_chunk.iloc[:, 2:5].values
            
            if len(vib_raw) < cfg.dataset_fs: continue

            # --- RPM FILTER FIX ---
            avg_rpm = np.mean(rpm_raw)
            if avg_rpm <= 0 or avg_rpm > 10000:
                continue # Skip garbage chunks
            if avg_rpm<768 or avg_rpm>3686:
                continue
            # ----------------------

            if cfg.dataset_fs != cfg.model_fs:
                gcd = np.gcd(cfg.dataset_fs, cfg.model_fs)
                vib_4k = resample_poly(vib_raw, cfg.model_fs//gcd, cfg.dataset_fs//gcd, axis=0)
            else:
                vib_4k = vib_raw

            win_pts = int(cfg.window_sec * cfg.model_fs)
            step = win_pts 
            batch_tensors = []
            
            for start in range(0, len(vib_4k) - win_pts, step):
                segment = vib_4k[start : start + win_pts]
                segment = safe_zscore(segment)
                x_map = compute_order_spectrum(segment, cfg.model_fs, avg_rpm, cfg)
                batch_tensors.append(torch.from_numpy(x_map))
            
            if not batch_tensors: continue

            batch_input = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                logits = model(batch_input)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confs = torch.max(probs, dim=1).values.cpu().numpy()
            
            for pred_idx, conf in zip(preds, confs):
                pred_label = cfg.classes[pred_idx]
                stats[pred_label] += 1
                total_windows += 1
                writer.writerow([filename, ground_truth, pred_label, f"{conf:.4f}", f"{avg_rpm:.1f}"])

    dominant_class = max(stats, key=stats.get)
    acc = (stats[ground_truth] / total_windows * 100) if ground_truth in stats and total_windows > 0 else 0.0
    
    print(f"    Summary: {dominant_class} ({stats[dominant_class]} wins) | Acc: {acc:.2f}%")
    
    return {"File": filename, "GroundTruth": ground_truth, "DominantPred": dominant_class, "Accuracy": acc}

# --- 4. MAIN ---
def main():
    # --- UPDATE PATHS HERE ---
    # script_dir = Path(__file__).resolve().parent
    # # Assuming this script is in src/inference, go up to root
    # ROOT_DIR = script_dir.parent.parent 
    # MODEL_PATH = ROOT_DIR / "src/models_transfer_learning/best_model_transfer.pth"
    # DATA_DIR = ROOT_DIR / "data/TFDT_Data"
    # OUTPUT_CSV = ROOT_DIR / "inference_results_fixed.csv"


    script_dir = Path(__file__).resolve().parent
    script_dir2 = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = script_dir2 / "data" / "TFDT_Data"
    print(f"Data Dir: {DATA_DIR}")
    print(f"Base Dir: {script_dir}")
    # Your NEW trained model
    MODEL_PATH = Path(f"{script_dir2}/src/models_sota_resnet/best_model_resnet.pth") 
    
    OUTPUT_CSV = "inference_results_unbalance.csv"
    
    cfg = InferenceConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Model: {MODEL_PATH}")
    model = ResNet1D().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        return

    all_files = sorted(list(DATA_DIR.glob("*.csv")))
    if not all_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(all_files)} files.")
    
    file_summaries = []
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Ground_Truth", "Prediction", "Confidence", "Avg_RPM"])
        for csv_file in all_files:
            summary = process_large_file(csv_file, model, device, cfg, writer)
            file_summaries.append(summary)

    print("\nFINAL REPORT")
    print(pd.DataFrame(file_summaries)[["File", "GroundTruth", "DominantPred", "Accuracy"]])

if __name__ == "__main__":
    main()