import sys
import csv
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly
from tqdm import tqdm

# --------------------------------------------------
# IMPORT NEW ARCHITECTURE & UTILITIES
# --------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "training"))
from train_explain_cnn import GeneralizingCNN, compute_robust_spectrum, get_rpm

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
class InferenceConfig:
    def __init__(self):
        self.max_order = 15.0
        self.order_bins = 256
        self.dataset_fs = 4096     # CSV sampling rate
        self.model_fs = 4000       # Model trained at 4kHz
        self.window_sec = 1.0
        self.classes = ["Normal", "Imbalance", "Misalignment", "Bearing"]

# --------------------------------------------------
# INFERENCE FUNCTIONS
# --------------------------------------------------
def sliding_window_chunks(vib_raw, rpm, cfg):
    """Compute spectra for sliding windows"""
    win_pts = int(cfg.window_sec * cfg.model_fs)
    step_pts = win_pts  # non-overlapping

    batch_tensors = []
    for start in range(0, len(vib_raw) - win_pts + 1, step_pts):
        segment = vib_raw[start:start+win_pts]
        spec = compute_robust_spectrum(segment, cfg.model_fs, rpm, cfg)
        batch_tensors.append(torch.from_numpy(spec))

    if not batch_tensors:
        return None
    return torch.stack(batch_tensors)

def process_csv_file(filepath, model, device, cfg, writer):
    filename = filepath.name
    ground_truth = filename[0]  # Optional mapping based on filename

    # Example mapping (update if needed)
    fault_map = {"0": "Normal", "1": "Imbalance", "2": "Imbalance",
                 "3": "Imbalance", "4": "Imbalance"}
    ground_truth = fault_map.get(ground_truth, "Unknown")

    print(f"\nProcessing {filename} | Ground Truth: {ground_truth}")
    
    chunk_size = cfg.dataset_fs * 5  # 5-second chunks
    stats = {k: 0 for k in cfg.classes}
    total_windows = 0

    with pd.read_csv(filepath, chunksize=chunk_size) as reader:
        for df_chunk in tqdm(reader, desc=f"Scanning {filename}", unit="chunk"):
            if df_chunk.shape[0] < cfg.dataset_fs:
                continue

            # --- COLUMN MAPPING ---
            tach = df_chunk.iloc[:, 1].values        # Measured_RPM
            vib_raw = df_chunk.iloc[:, 2:5].values   # Vibration_1-3

            avg_rpm = np.mean(tach)
            if avg_rpm < 500 or avg_rpm > 2500:
                continue  # Skip chunks outside dataset RPM

            # Resample to model_fs if necessary
            if cfg.dataset_fs != cfg.model_fs:
                gcd = np.gcd(cfg.dataset_fs, cfg.model_fs)
                vib_4k = resample_poly(vib_raw, cfg.model_fs//gcd, cfg.dataset_fs//gcd, axis=0)
            else:
                vib_4k = vib_raw

            batch_input = sliding_window_chunks(vib_4k, avg_rpm, cfg)
            if batch_input is None:
                continue
            batch_input = batch_input.to(device)

            # --- MODEL PREDICTION ---
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
    print(f"    Summary: {dominant_class} ({stats[dominant_class]} windows) | Acc: {acc:.2f}%")

    return {"File": filename, "GroundTruth": ground_truth, "DominantPred": dominant_class, "Accuracy": acc}

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "TFDT_Data"
    MODEL_PATH = BASE_DIR / "src" / "models_gen_robust" / "best_model.pth"
    OUTPUT_CSV = BASE_DIR / "inference_results.csv"

    cfg = InferenceConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = GeneralizingCNN(input_bins=cfg.order_bins).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"ERROR: Model not found at {MODEL_PATH}")
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
            summary = process_csv_file(csv_file, model, device, cfg, writer)
            file_summaries.append(summary)

    # --- FINAL REPORT ---
    print("\nFINAL REPORT")
    print(pd.DataFrame(file_summaries)[["File", "GroundTruth", "DominantPred", "Accuracy"]])

if __name__ == "__main__":
    main()