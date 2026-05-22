import os
import random
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import welch, decimate, hilbert, butter, sosfilt
from scipy import interpolate
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

# ==================== CONFIGURATION ====================
# 👉 Update this path to where your Kaggle output best_model.pt is located
MODEL_WEIGHTS_PATH = r"./best_model.pt"
RAW_DATA_ROOT = r"C:\DEV\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLING_FREQ_RAW = 50000
VIBRATION_COLS = [1, 2, 3]
TACH_COL = 0
AXIS_PREFIXES = ['ax', 'rad', 'tan']

# Label Encoder alphabetical order (used during sklearn's LabelEncoder fit)
CLASS_NAMES = ['Ball_Fault', 'Horiz_Misalign', 'Imbalance', 'Normal', 'Outer_Race', 'Vert_Misalign']

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("Deep_MaFaulDa_Server")

app = FastAPI(title="VibraGuard Deep AI - Spectral CNN", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==================== PYTORCH MODEL DEFINITION ====================
class SpectralCNN(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
            nn.Dropout1d(0.2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # Convert to (batch, channels, sequence)
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

deep_model = None

# ==================== SPECTRAL PREPROCESSOR ====================
class SpectralOrderPreprocessor:
    def __init__(self, orders_per_rev: int = 64, revolutions: int = 8):
        self.orders_per_rev = orders_per_rev
        self.revolutions = revolutions
        self.window_size = orders_per_rev * revolutions
    
    def detect_tach_pulses(self, tach_signal: np.ndarray, sampling_freq: float) -> np.ndarray:
        # If tach is flat (missing), return empty
        if np.max(tach_signal) - np.min(tach_signal) < 0.1:
            return np.array([])
            
        threshold = np.mean(tach_signal) + 0.5 * np.std(tach_signal)
        binary = tach_signal > threshold
        min_samples = max(1, int(sampling_freq / 5000))
        rising_edges, last_edge = [], -min_samples
        for i in range(1, len(binary) - 1):
            if not binary[i-1] and binary[i] and binary[i+1]:
                if i - last_edge > min_samples:
                    rising_edges.append(int(i)) 
                    last_edge = i
        return np.array(rising_edges, dtype=np.int32)
    
    def resample_to_orders(self, vib_signal: np.ndarray, tach_pulses: np.ndarray) -> Optional[np.ndarray]:
        if len(tach_pulses) < self.revolutions + 1: return None
        start_idx, end_idx = int(tach_pulses[0]), int(tach_pulses[self.revolutions])
        if end_idx <= start_idx or (end_idx - start_idx) < (self.window_size * 0.3): return None
        
        original_indices = np.arange(start_idx, end_idx, dtype=np.float32)
        target_indices = np.linspace(start_idx, end_idx, self.window_size, dtype=np.float32)
        resampled = np.zeros((self.window_size, vib_signal.shape[1]), dtype=np.float32)
        
        for ax in range(vib_signal.shape[1]):
            f = interpolate.interp1d(original_indices, vib_signal[start_idx:end_idx, ax], kind='cubic', fill_value="extrapolate")
            resampled[:, ax] = f(target_indices)
        return resampled

# ==================== LIFECYCLE ====================
@app.on_event("startup")
async def load_model():
    global deep_model
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logger.error(f"❌ PyTorch model not found at {MODEL_WEIGHTS_PATH}")
        return
        
    try:
        logger.info("⏳ Loading PyTorch Deep Learning model...")
        deep_model = SpectralCNN(num_classes=6).to(DEVICE)
        deep_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        deep_model.eval()
        logger.info("✅ PyTorch Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load PyTorch model: {e}")

# ==================== API MODELS & UTILS ====================
class SignalData(BaseModel):
    tach: List[float]
    ax: List[float]
    ay: List[float]
    az: List[float]
    fs: float = 50000.0

def compute_spectrum(signal: np.ndarray, fs: float) -> tuple:
    f, pxx = welch(signal, fs=fs, nperseg=1024, window='hann', scaling='density')
    return f.astype(np.float32), pxx.astype(np.float32)

def compute_envelope_spectrum(signal: np.ndarray, fs: float) -> tuple:
    nyq = fs / 2
    sos = butter(4, [max(500/nyq, 0.01), min(2000/nyq, 0.99)], btype='band', output='sos')
    filtered = sosfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    f_env, pxx_env = welch(envelope - np.mean(envelope), fs=fs, nperseg=1024, window='hann')
    return f_env.astype(np.float32), pxx_env.astype(np.float32)

def calculate_rpm(tach_signal, fs):
    if len(tach_signal) < 10 or np.max(tach_signal) - np.min(tach_signal) < 0.1:
        return 1750.0
    thresh = (np.max(tach_signal) + np.min(tach_signal)) / 2
    binary = tach_signal > thresh
    edges = np.where((binary[:-1] == False) & (binary[1:] == True))[0]
    if len(edges) < 2: return 1750.0
    return ((len(edges) - 1) / ((edges[-1] - edges[0]) / fs)) * 60

# ==================== ENDPOINTS ====================
@app.get("/health")
async def health_check():
    return {
        "status": "ready" if deep_model else "error",
        "model_loaded": deep_model is not None,
        "classes": CLASS_NAMES,
        "engine": "PyTorch SpectralCNN"
    }

@app.get("/get_sample")
async def get_sample(fault_type: str = "Normal"):
    # (Same file fetching logic as your previous server)
    name_mapping = {
        "Normal": "Normal", "Imbalance": "Imbalance", 
        "Horiz_Misalign": "Horiz_Misalign", "Vert_Misalign": "Vert_Misalign",
        "Ball_Fault": "Ball_Fault", "Outer_Race": "Outer_Race"
    }
    actual_fault = name_mapping.get(fault_type, "Normal")
    folder_map = {
        "Normal": ("normal", []),
        "Imbalance": ("imbalance", ["6g", "10g", "20g", "30g", "35g"]),
        "Horiz_Misalign": ("horizontal-misalignment", ["0.5mm", "1.0mm", "2.0mm"]),
        "Vert_Misalign": ("vertical-misalignment", ["0.51mm", "1.27mm", "1.90mm"]),
        "Ball_Fault": ("underhang/ball_fault", ["6g", "20g", "35g"]),
        "Outer_Race": ("underhang/outer_race", ["6g", "20g", "35g"])
    }
    
    base_path, subfolders = folder_map[actual_fault]
    search_path = Path(RAW_DATA_ROOT) / base_path
    
    candidates = []
    if subfolders:
        for sub in subfolders: candidates.extend(search_path.glob(f"**/{sub}/*.csv"))
    else:
        candidates.extend(search_path.glob("**/*.csv"))
        
    if not candidates: raise HTTPException(404, "Files not found")
    
    target = random.choice([c for c in candidates if c.is_file()])
    df = pd.read_csv(target, header=None, nrows=50000)
    
    return {
        "filename": target.name,
        "fault_type": actual_fault,
        "tach": df.iloc[:, 0].values.tolist(),
        "ax": df.iloc[:, 1].values.tolist(),
        "ay": df.iloc[:, 2].values.tolist(),
        "az": df.iloc[:, 3].values.tolist()
    }

@app.post("/predict")
async def predict(data: SignalData):
    if deep_model is None: raise HTTPException(503, "Model offline")
    
    raw_vib = np.stack([data.ax, data.ay, data.az], axis=1)
    raw_tach = np.array(data.tach, dtype=np.float32)
    fs = data.fs
    
    rpm = calculate_rpm(raw_tach, fs)
    preprocessor = SpectralOrderPreprocessor()
    
    # ⚠️ FALLBACK: If user uploads ESP32 data with no tach, generate fake tach pulses
    pulses = preprocessor.detect_tach_pulses(raw_tach, fs)
    if len(pulses) < preprocessor.revolutions + 1:
        logger.warning("No Tach pulses found! Synthesizing pulses assuming 1750 RPM.")
        samples_per_rev = fs / (1750 / 60)
        pulses = np.arange(0, len(raw_vib), samples_per_rev).astype(np.int32)
        
    # Apply DL Preprocessing Pipeline
    window = preprocessor.resample_to_orders(raw_vib, pulses[:preprocessor.revolutions + 1])
    
    if window is not None:
        hanning_win = np.hanning(preprocessor.window_size)[:, None]
        windowed_signal = window * hanning_win
        spectrum = np.abs(np.fft.rfft(windowed_signal, axis=0)) / preprocessor.window_size
        
        X_log = np.log1p(spectrum * 1000)
        local_max = np.max(X_log) + 1e-8 # Local max normalization fallback
        X_scaled = X_log / local_max
        
        tensor_x = torch.from_numpy(X_scaled).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = deep_model(tensor_x)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            
        pred_idx = np.argmax(probs)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
    else:
        pred_label = "Insufficient Data"
        confidence = 0.0

    # Extract UI baseline features so the React gauges don't break
    radial_signal = raw_vib[-4096:, 1]
    rms_ay = np.sqrt(np.mean(radial_signal**2))
    spec_f, spec_val = compute_spectrum(radial_signal, fs)
    env_f, env_val = compute_envelope_spectrum(radial_signal, fs)
    
    return {
        "prediction": pred_label,
        "confidence": confidence,
        "rpm": rpm,
        "explanation": "Deep Learning Spectral Analysis",
        "features": {
            "ax_rms": np.sqrt(np.mean(raw_vib[-4096:,0]**2)),
            "tan_rms": np.sqrt(np.mean(raw_vib[-4096:,2]**2)),
            "rad_kurt": stats.kurtosis(radial_signal, fisher=False),
            "rad_crest": np.max(np.abs(radial_signal)) / (rms_ay + 1e-12),
            "ax_spec_spread": 80, # Stubbed out for PyTorch DL speed
            "tan_spec_spread": 80
        },
        "spectrum_f": spec_f.tolist(),
        "spectrum_val": spec_val.tolist(),
        "envelope_f": env_f.tolist(),
        "envelope_val": env_val.tolist(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")