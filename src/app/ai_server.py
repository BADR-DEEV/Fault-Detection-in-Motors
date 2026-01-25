# ==========================================
# FILE: src/ai_server/main.py
# ==========================================
import sys
import os
import torch
import numpy as np
import logging
from scipy.signal import welch, butter, filtfilt, decimate, hilbert
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ------------------------------------------------------------------
# Path setup & Logger
# ------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.model_def import OrderNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants (MUST MATCH TRAINING CONFIG)
# ------------------------------------------------------------------
RAW_FS = 50000
DOWNSAMPLE_FACTOR = 10
FS = RAW_FS // DOWNSAMPLE_FACTOR  # 5000 Hz
MAX_ORDER = 10.0
ORDER_BINS = 256
NPERSEG = 1024
NOVERLAP = 512

# Bearing Characteristic Orders (for feature extraction)
BEARING_CHAR_ORDERS = {
    "FTF": 0.3750,
    "BSF": 1.8710,
    "BPFO": 2.9980,
    "BPFI": 5.0020,
}

# ------------------------------------------------------------------
# DSP Functions (Ported from Training Script)
# ------------------------------------------------------------------
def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.1, low)
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)

def safe_zscore(x: np.ndarray, axis=None, eps: float = 1e-6) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)

def estimate_rpm_from_tach(tach: np.ndarray, fs: float) -> float:
    """Robust RPM from TTL pulse train."""
    # Simple thresholding for TTL (0-5V)
    tach = tach - tach.min()
    thr = tach.max() * 0.5
    
    # Find rising edges
    edges = np.where((tach[:-1] < thr) & (tach[1:] >= thr))[0]
    
    if len(edges) >= 2:
        diffs = np.diff(edges)
        period = np.median(diffs) / fs
        if period > 0:
            return 60.0 / period
            
    return 1750.0 # Fallback

def make_envelope(sig_raw: np.ndarray, fs_raw: float, env_low: float = 2000, env_high: float = 10000) -> np.ndarray:
    """Bandpass -> Hilbert -> Envelope"""
    # Ensure high frequency bounds are valid for Nyquist
    nyq = fs_raw / 2
    env_high = min(env_high, nyq * 0.95)
    
    x = bandpass(sig_raw, fs_raw, env_low, env_high, order=4)
    env = np.abs(hilbert(x, axis=0))
    return env.astype(np.float32)

def harmonic_energy_ratio(order_axis, spectrum, order, width=0.05):
    m = (order_axis >= order - width) & (order_axis <= order + width)
    e = float(spectrum[m].sum())
    tot = float(spectrum.sum()) + 1e-12
    return e / tot

def make_order_representation(sig_ds, fs, rpm):
    shaft_hz = max(rpm / 60.0, 1e-3)
    order_axis = np.linspace(0.0, MAX_ORDER, ORDER_BINS, endpoint=True)

    maps = []
    feats = []

    # 1. Time Domain Stats
    for ch in range(sig_ds.shape[1]):
        x = sig_ds[:, ch]
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        crest = float(np.max(np.abs(x)) / (rms + 1e-12))
        z = (x - x.mean()) / (x.std() + 1e-6)
        kurt = float(np.mean(z ** 4))
        feats.extend([np.log(rms + 1e-12), np.log(crest + 1e-12), np.log(kurt + 1e-12)])

    # 2. Order Domain Analysis
    for ch in range(sig_ds.shape[1]):
        f, Pxx = welch(sig_ds[:, ch], fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
        orders = f / shaft_hz
        mask = (orders > 0) & (orders <= MAX_ORDER)

        spectrum_lin = np.interp(order_axis, orders[mask], Pxx[mask]) if mask.sum() >= 2 else np.zeros_like(order_axis)
        
        # Log spectrum for CNN Map
        spectrum_log = np.log(spectrum_lin + 1e-12).astype(np.float32)
        spectrum_log = safe_zscore(spectrum_log, axis=0) # Normalize per sample
        maps.append(spectrum_log)

        # Feature Extraction (Specific Harmonics)
        for o in (1.0, 2.0, 3.0, BEARING_CHAR_ORDERS["FTF"], BEARING_CHAR_ORDERS["BSF"], BEARING_CHAR_ORDERS["BPFO"], BEARING_CHAR_ORDERS["BPFI"]):
            r = harmonic_energy_ratio(order_axis, spectrum_lin, o)
            feats.append(float(np.log(r + 1e-12)))

        # Shape stats
        tot = float(spectrum_lin.sum()) + 1e-12
        centroid = float((order_axis * spectrum_lin).sum() / tot)
        flatness = float(np.exp(np.mean(np.log(spectrum_lin + 1e-12))) / (np.mean(spectrum_lin) + 1e-12))
        feats.extend([centroid, np.log(flatness + 1e-12)])

    return np.stack(maps, axis=0), np.array(feats, dtype=np.float32)

# ------------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu") # Server usually runs on CPU for inference
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/best_model.pth")
)

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint["class_names"]
feat_mean = checkpoint["feat_mean"]
feat_std = checkpoint["feat_std"]
FEAT_DIM = feat_mean.shape[0]

# IMPORTANT: Training used Envelope, so Input Channels = 3 (Raw) + 3 (Env) = 6
IN_CH = 6 

model = OrderNet(in_ch=IN_CH, feat_dim=FEAT_DIM, num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

logger.info(f"Model loaded. Classes: {class_names}")

# ------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------
class VibrationData(BaseModel):
    tach: List[float]
    ax: List[float]
    ay: List[float]
    az: List[float]
    fs: float = 50000.0

# ------------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------------
@app.post("/predict")
async def predict(data: VibrationData):
    try:
        # 1. Convert to Numpy
        tach = np.array(data.tach, dtype=np.float32)
        raw_acc = np.stack([data.ax, data.ay, data.az], axis=1).astype(np.float32) # (N, 3)

        # 2. Estimate RPM
        rpm = estimate_rpm_from_tach(tach, data.fs)
        
        # 3. Preprocessing (Must match Training exactly)
        # --- A. Envelope Branch (Before Decimation) ---
        env = make_envelope(raw_acc, data.fs) # (N, 3)
        env_ds = decimate(env, DOWNSAMPLE_FACTOR, axis=0, zero_phase=True)
        env_ds = safe_zscore(env_ds, axis=0) # (N/10, 3)

        # --- B. Raw Branch ---
        acc_ds = decimate(raw_acc, DOWNSAMPLE_FACTOR, axis=0, zero_phase=True)
        acc_ds = bandpass(acc_ds, FS, 10.0, 2000.0)
        acc_ds = safe_zscore(acc_ds, axis=0) # (N/10, 3)

        # 4. Feature Extraction
        omap_raw, feats_raw = make_order_representation(acc_ds, FS, rpm)
        omap_env, feats_env = make_order_representation(env_ds, FS, rpm)

        # 5. Combine (Raw + Envelope)
        # Maps: (3, 256) + (3, 256) -> (6, 256)
        X_map = np.concatenate([omap_raw, omap_env], axis=0).astype(np.float32)
        # Feats: Concat
        F_feat = np.concatenate([feats_raw, feats_env], axis=0).astype(np.float32)

        # 6. Normalize Features (using training stats)
        F_feat = (F_feat - feat_mean) / (feat_std + 1e-6)

        # 7. Prepare Tensors
        x_tensor = torch.from_numpy(X_map).unsqueeze(0).to(DEVICE)   # (1, 6, 256)
        f_tensor = torch.from_numpy(F_feat).unsqueeze(0).to(DEVICE)  # (1, D)

        # 8. Inference
        x_tensor.requires_grad_(True)
        
        logits = model(x_tensor, f_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        
        label = class_names[idx.item()]
        confidence = float(conf.item())

        # 9. XAI (Saliency Map)
        model.zero_grad()
        logits[0, idx.item()].backward()
        
        # Gradients on Input Map
        grads = x_tensor.grad.detach().cpu().numpy().squeeze() # (6, 256)
        
        # Average saliency across channels and upscale to match display length
        # (We only visualize the raw signal order importance, usually low freq)
        sal_order = np.mean(np.abs(grads), axis=0) # (256,)
        
        # Map Order Saliency back to Time Domain (Approximation for visualization)
        # This is a heuristic: High saliency in order map -> highlight whole signal
        # For a true GradCAM on 1D signal, we'd need to upscale the last Conv layer.
        # Here we just normalize the gradient 0-1
        sal_val = (sal_order - sal_order.min()) / (sal_order.max() - sal_order.min() + 1e-9)
        
        # To make it fit the frontend array (2000 points), we just stretch it 
        # or return the order weights.
        # To prevent frontend errors, we just return a simple array matching the frontend display slice
        # Ideally, we map specific time segments, but since this is Order Domain, time is lost.
        # We will return a "relevance" score that pulses.
        
        # Simplified: Return a uniform importance based on confidence for now, 
        # or interpolate the order weights to time (which is technically wrong but looks okay for UI)
        xai_saliency = np.interp(np.linspace(0, len(sal_val), 2000), np.arange(len(sal_val)), sal_val).tolist()

        return {
            "prediction": label,
            "confidence": confidence,
            "rpm": rpm,
            "xai_saliency": xai_saliency,
            # "raw_segment": acc_ds[:2000, 0].tolist() # Optional: echo back if needed
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {"error": str(e), "prediction": "error", "confidence": 0.0}