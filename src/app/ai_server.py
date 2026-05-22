import sys
import os
import random
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import welch, decimate, hilbert, butter, sosfilt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path

# ==================== CONFIGURATION ====================
MODEL_PIPELINE_PATH = r"C:\DEV\personal\Ai-Driven-Vibrations-motor\src\training_cleaned\svm_pipeline_physics_validated.pkl"
RAW_DATA_ROOT = r"C:\DEV\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"

WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 13
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR
VIBRATION_COLS = [1, 2, 3]
TACH_COL = 0
AXIS_PREFIXES = ['ax', 'rad', 'tan']
PER_AXIS_FEATURES = ['rms', 'kurt', 'crest', 'spec_centroid', 'spec_spread', 'freq_median', 'freq_rolloff85']

# 🔊 NOISE CONFIGURATION
# Adds realistic sensor noise to test model generalization
NOISE_RATIO = 0.4  # % of signal RMS as noise standard deviation
ENABLE_NOISE = True  # Set False to disable noise injection

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("MaFaulDa_Server")

model_pipeline = None
app = FastAPI(title="VibraGuard AI Server - MaFaulDa Physics-Aligned", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_jitter(signal: np.ndarray, jitter_ratio: float = 0.005) -> np.ndarray:
    """
    Adds small random jitter (temporal instability simulation).
    This simulates:
    - ADC timing jitter
    - Micro mechanical vibration disturbances
    """
    if not ENABLE_NOISE:
        return signal

    jitter = np.random.normal(0, jitter_ratio, size=signal.shape)
    return signal + jitter


# ==================== NOISE INJECTION ====================
def add_sensor_noise(signal: np.ndarray, noise_ratio: float = NOISE_RATIO) -> np.ndarray:
    """
    Adds realistic sensor noise to vibration signal.
    Noise is Gaussian with std = noise_ratio * signal_rms.
    This simulates:
    - Sensor electronic noise
    - Calibration drift
    - Environmental interference
    """
    if not ENABLE_NOISE:
        return signal
    
    rms = np.sqrt(np.mean(signal**2))
    if rms < 1e-12:
        return signal  # No noise for zero signal
    
    noise_std = rms * noise_ratio
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise

# ==================== SPECTRUM COMPUTATION ====================
def compute_spectrum(signal: np.ndarray, fs: float) -> tuple:
    nperseg = min(1024, len(signal))
    f, pxx = welch(signal, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    return f.astype(np.float32), pxx.astype(np.float32)

def compute_envelope_spectrum(signal: np.ndarray, fs: float) -> tuple:
    nyq = fs / 2
    low = max(500 / nyq, 0.01)
    high = min(2000 / nyq, 0.99)
    
    try:
        sos = butter(4, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, signal)
    except:
        filtered = signal
    
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    envelope_ac = envelope - np.mean(envelope)
    
    nperseg = min(1024, len(envelope_ac))
    f_env, pxx_env = welch(envelope_ac, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    
    return f_env.astype(np.float32), pxx_env.astype(np.float32)

# ==================== PHYSICS ENGINE ====================
def calculate_rpm_from_tach(tach_signal, sampling_freq):
    if len(tach_signal) < 10:
        return None
        
    # FIX: Prevent processing if the tachometer is a dummy array (all zeros)
    if np.max(tach_signal) == np.min(tach_signal):
        return None
        
    threshold = (np.max(tach_signal) + np.min(tach_signal)) / 2
    binary = tach_signal > threshold
    rising_edges = np.where((binary[:-1] == False) & (binary[1:] == True))[0]
    if len(rising_edges) < 2:
        return None
    time_between = (rising_edges[-1] - rising_edges[0]) / sampling_freq
    revolutions = len(rising_edges) - 1
    if time_between <= 0 or revolutions == 0:
        return None
    return (revolutions / time_between) * 60

def extract_features_with_rpm(vib_signal, tach_signal, sampling_freq):
    rpm = calculate_rpm_from_tach(tach_signal, sampling_freq)
    features = []
    rms_vals = []
    
    for ax_idx, ax_prefix in enumerate(AXIS_PREFIXES):
        signal = vib_signal[:, ax_idx]
        rms = np.sqrt(np.mean(signal**2))
        kur = stats.kurtosis(signal, fisher=False)
        crest = np.max(np.abs(signal)) / (rms + 1e-12)
        rms_vals.append(rms)
        
        nperseg = min(1024, len(signal))
        f, Pxx = welch(signal, fs=sampling_freq, nperseg=nperseg)
        Pxx = np.maximum(Pxx, 1e-12)
        totalE = np.sum(Pxx)
        
        if totalE == 0:
            FM = FSD = FMED = SRO = 0.0
        else:
            FM = np.sum(f * Pxx) / totalE
            FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / totalE)
            cumulative = np.cumsum(Pxx)
            FMED = f[np.searchsorted(cumulative, 0.5 * totalE)]
            SRO = f[np.searchsorted(cumulative, 0.85 * totalE)]
        
        features.extend([rms, kur, crest, FM, FSD, FMED, SRO])
    
    rms_axial, rms_radial, rms_tangential = rms_vals
    axial_ratio = rms_axial / (rms_radial + rms_tangential + 1e-12)
    radial_ratio = rms_radial / (rms_axial + rms_tangential + 1e-12)
    features.extend([axial_ratio, radial_ratio])
    
    return np.array(features, dtype=np.float32), rpm

# ==================== LIFECYCLE ====================
@app.on_event("startup")
async def load_model_pipeline():
    global model_pipeline
    
    if not os.path.exists(MODEL_PIPELINE_PATH):
        logger.error(f"❌ Model pipeline not found: {MODEL_PIPELINE_PATH}")
        return
    
    try:
        logger.info("⏳ Loading MaFaulDa physics-validated SVM pipeline...")
        model_pipeline = joblib.load(MODEL_PIPELINE_PATH)
        
        required_keys = ['scaler', 'model', 'label_encoder', 'feature_names', 'physics_validation']
        if not all(k in model_pipeline for k in required_keys):
            raise ValueError(f"Pipeline missing required keys: {required_keys}")
        
        if len(model_pipeline['feature_names']) != 23:
            raise ValueError(f"Feature count mismatch! Expected 23, got {len(model_pipeline['feature_names'])}")
        
        n_classes = len(model_pipeline['label_encoder'].classes_)
        if n_classes != 6:
            raise ValueError(f"Class count mismatch! Expected 6, got {n_classes}")
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Classes: {model_pipeline['label_encoder'].classes_}")
        logger.info(f"   Features: {len(model_pipeline['feature_names'])}")
        logger.info(f"   Noise injection: {'ENABLED' if ENABLE_NOISE else 'DISABLED'} (ratio={NOISE_RATIO})")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model pipeline: {e}", exc_info=True)
        model_pipeline = None

# ==================== API MODELS ====================
class SignalData(BaseModel):
    tach: List[float]
    ax: List[float]
    ay: List[float]
    az: List[float]
    fs: float = 50000.0

# ==================== ENDPOINTS ====================
@app.get("/health")
async def health_check():
    if model_pipeline is None:
        return {
            "status": "error",
            "model_loaded": False,
            "error": "Model pipeline not loaded",
            "required_file": MODEL_PIPELINE_PATH
        }
    
    return {
        "status": "ready",
        "model_loaded": True,
        "classes": model_pipeline['label_encoder'].classes_.tolist(),
        "feature_count": len(model_pipeline['feature_names']),
        "physics_validated": model_pipeline['physics_validation'],
        "noise_enabled": ENABLE_NOISE,
        "noise_ratio": NOISE_RATIO,
        "configuration": {
            "window_size_decimated": WINDOW_SIZE,
            "decimation_factor": DECIMATION_FACTOR,
            "sampling_freq_decimated": SAMPLING_FREQ_DECIMATED
        }
    }

@app.get("/get_sample")
async def get_sample(fault_type: str = "Normal"):
    name_mapping = {
        "normal": "Normal", "Normal": "Normal",
        "imbalance": "Imbalance", "Imbalance": "Imbalance",
        "misalignment": "Horiz_Misalign",
        "Horiz_Misalign": "Horiz_Misalign", "Vert_Misalign": "Vert_Misalign",
        "bearing": "Ball_Fault",
        "Ball_Fault": "Ball_Fault", "Outer_Race": "Outer_Race"
    }
    
    actual_fault = name_mapping.get(fault_type, "Normal")
    
    folder_map = {
        "Normal": ("normal", []),
        "Imbalance": ("imbalance", ["6g", "10g", "15g", "20g", "25g", "30g", "35g"]),
        "Horiz_Misalign": ("horizontal-misalignment", ["0.5mm", "1.0mm", "1.5mm", "2.0mm"]),
        "Vert_Misalign": ("vertical-misalignment", ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"]),
        "Ball_Fault": ("underhang/ball_fault", ["6g", "20g", "35g"]),
        "Outer_Race": ("underhang/outer_race", ["6g", "20g", "35g"])
    }
    
    if actual_fault not in folder_map:
        raise HTTPException(400, f"Invalid fault type. Valid: {list(folder_map.keys())}")
    
    base_path, subfolders = folder_map[actual_fault]
    search_path = Path(RAW_DATA_ROOT) / base_path
    
    candidates = []
    if subfolders:
        for sub in subfolders:
            candidates.extend(search_path.glob(f"**/{sub}/*.csv"))
            candidates.extend(search_path.glob(f"{sub}/*.csv"))
    else:
        candidates.extend(search_path.glob("*.csv"))
        candidates.extend(search_path.glob("**/*.csv"))
    
    candidates = [c for c in candidates if c.is_file()]
    
    if not candidates:
        raise HTTPException(404, f"No files found for '{actual_fault}' in {search_path}")
    
    target = random.choice(candidates)
    try:
        df = pd.read_csv(target, header=None, nrows=50000)
        if df.shape[1] < 4:
            raise ValueError(f"CSV has {df.shape[1]} columns, expected ≥4")
        
        # Extract signals
        tach = df.iloc[:, TACH_COL].astype(float).values
        ax = df.iloc[:, VIBRATION_COLS[0]].astype(float).values
        ay = df.iloc[:, VIBRATION_COLS[1]].astype(float).values
        az = df.iloc[:, VIBRATION_COLS[2]].astype(float).values
        
        # 🔊 ADD SENSOR NOISE
        if ENABLE_NOISE:
            ax = add_sensor_noise(ax, NOISE_RATIO)
            ay = add_sensor_noise(ay, NOISE_RATIO)
            az = add_sensor_noise(az, NOISE_RATIO)
            # Tach is kept clean for reliable RPM calculation
            logger.info(f"🔊 Added noise (ratio={NOISE_RATIO}) to {target.name}")
        
        def extract_severity(p):
            import re
            ps = str(p).lower()
            if m := re.search(r'(\d+\.?\d*)\s*(g|mm)', ps):
                return f"{m.group(1)}{m.group(2)}"
            return "healthy" if 'normal' in ps else "unknown"
        
        def estimate_rpm(fp):
            try:
                d = pd.read_csv(fp, header=None, nrows=10000)
                t = d.iloc[:, 0].values
                r = calculate_rpm_from_tach(t, SAMPLING_FREQ_RAW)
                return r if r and 400 <= r <= 5000 else 1750.0
            except:
                return 1750.0
        
        return {
            "filename": target.name,
            "fault_type": actual_fault,
            "severity": extract_severity(target),
            "rpm_estimate": estimate_rpm(target),
            "tach": tach.tolist(),
            "ax": ax.tolist(),
            "ay": ay.tolist(),
            "az": az.tolist()
        }
    except Exception as e:
        logger.error(f"Error reading {target}: {e}")
        raise HTTPException(500, f"Failed to read sample: {str(e)}")

@app.post("/predict")
async def predict(data: SignalData):
    if model_pipeline is None:
        raise HTTPException(503, "Model pipeline not loaded")
    
    try:
        raw_ax = np.array(data.ax, dtype=np.float32)
        raw_ay = np.array(data.ay, dtype=np.float32)
        raw_az = np.array(data.az, dtype=np.float32)

        if ENABLE_NOISE:
            raw_ax = add_sensor_noise(raw_ax, NOISE_RATIO)
            raw_ay = add_sensor_noise(raw_ay, NOISE_RATIO)
            raw_az = add_sensor_noise(raw_az, NOISE_RATIO)

        raw_vib = np.stack([raw_ax, raw_ay, raw_az], axis=1)
        raw_tach = np.array(data.tach, dtype=np.float32)
        
        fs_raw = data.fs
        
        # FIX: DYNAMIC DECIMATION
        # If it is MaFaulDa (50kHz), we decimate. If it's your Motor (4kHz), WE SKIP IT to prevent explosions!
        if fs_raw > 10000:
            sig_vib = decimate(raw_vib, DECIMATION_FACTOR, axis=0, zero_phase=True)
            # Use simple slicing for tachometer to prevent IIR filter ripples
            sig_tach = raw_tach[::DECIMATION_FACTOR] 
            fs_actual = fs_raw / DECIMATION_FACTOR
        else:
            sig_vib = raw_vib
            sig_tach = raw_tach
            fs_actual = fs_raw
        
        # FIX: Safe padding with zeros, not repeating arrays
        if len(sig_vib) < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - len(sig_vib)
            sig_vib = np.pad(sig_vib, ((0, pad_len), (0, 0)), mode='constant')
            sig_tach = np.pad(sig_tach, (0, pad_len), mode='constant')
        
        window_vib = sig_vib[-WINDOW_SIZE:, :]
        window_tach = sig_tach[-WINDOW_SIZE:]
        
        feats, rpm = extract_features_with_rpm(window_vib, window_tach, fs_actual)
        
        if rpm is None or not (400 <= rpm <= 5000):
            rpm = 1750.0
            
        # FIX: Clean any NaNs before they hit the model
        feats = np.nan_to_num(feats)
        
        feats_scaled = model_pipeline['scaler'].transform(feats.reshape(1, -1))
        probs = model_pipeline['model'].predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = model_pipeline['label_encoder'].inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        
        radial_signal = window_vib[:, 1]
        spec_f, spec_val = compute_spectrum(radial_signal, fs_actual)
        env_f, env_val = compute_envelope_spectrum(radial_signal, fs_actual)
        
        # ... Keep your explanation logic here ...
        explanation = "Analysis Complete"

        return {
            "prediction": str(pred_label),
            "confidence": confidence,
            "rpm": float(rpm),
            "explanation": explanation,
            "features": {name: float(val) for name, val in zip(model_pipeline['feature_names'], feats)},
            "spectrum_f": spec_f.tolist(),
            "spectrum_val": spec_val.tolist(),
            "envelope_f": env_f.tolist(),
            "envelope_val": env_val.tolist(),
            "physics_validation": model_pipeline.get('physics_validation', True)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")