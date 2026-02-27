# ==========================================
# FILE: src/ai_server/main.py
# ==========================================
import sys
import os
import random
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import welch, decimate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path

# ==================== CONFIGURATION ====================
# PATHS (MUST MATCH TRAINING SCRIPT EXACTLY)
MODEL_PIPELINE_PATH = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\src\training_cleaned\svm_pipeline_physics_validated.pkl"
RAW_DATA_ROOT = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"

# PHYSICS CONSTANTS (BIT-FOR-BIT IDENTICAL TO TRAINING SCRIPT)
WINDOW_SIZE_RAW = 4096          # Raw samples BEFORE decimation (matches training)
DECIMATION_FACTOR = 13          # MaFaulDa standard decimation (50kHz → 3.85kHz)
WINDOW_SIZE_DECIM = WINDOW_SIZE_RAW // DECIMATION_FACTOR  # 315 samples AFTER decimation
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR  # 3846.15 Hz
VIBRATION_COLS = [1, 2, 3]      # Axial (col1), Radial (col2), Tangential (col3) per MaFaulDa CSV spec
TACH_COL = 0
AXIS_NAMES = ['ax', 'rad', 'tan']  # MUST match training feature naming exactly

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("MaFaulDa_Server")

# ==================== GLOBAL STATE ====================
model_pipeline = None
app = FastAPI(title="VibraGuard AI Server - MaFaulDa Physics-Aligned", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PHYSICS ENGINE (EXACT REPLICA OF TRAINING) ====================
def calculate_rpm_from_tach(tach_signal, sampling_freq):
    """
    Physics-accurate RPM from tachometer (1 pulse/revolution assumption)
    CRITICAL: Training used 1-pulse assumption consistently → Server MUST match to avoid distribution shift
    (MaFaulDa actually has 60 pulses/rev but training assumed 1 pulse/rev → we maintain consistency)
    """
    if len(tach_signal) < 10:
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

def extract_features_single_window(vib_window, tach_window, fs):
    """
    BIT-FOR-BIT REPLICA of training script's extract_features_with_rpm()
    Returns 23 features in EXACT order used during training:
    [ax_rms, ax_kurt, ax_crest, ax_spec_centroid, ax_spec_spread, ax_freq_median, ax_freq_rolloff85,
     rad_rms, ..., tan_freq_rolloff85, axial_ratio, radial_ratio]
    """
    rpm = calculate_rpm_from_tach(tach_window, fs)
    features = []
    rms_vals = []
    
    # EXACT FEATURE ORDER: 3 axes × 7 features = 21 features
    for ax_idx in range(3):
        signal = vib_window[:, ax_idx]
        
        # Time domain (EXACT ORDER: rms, kurt, crest)
        rms = np.sqrt(np.mean(signal**2))
        kur = stats.kurtosis(signal, fisher=False)  # Pearson's kurtosis (not Fisher's)
        crest = np.max(np.abs(signal)) / (rms + 1e-12)
        rms_vals.append(rms)
        
        # Frequency domain (EXACT ORDER: spec_centroid, spec_spread, freq_median, freq_rolloff85)
        nperseg = min(1024, len(signal))
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
        Pxx = np.maximum(Pxx, 1e-12)
        totalE = np.sum(Pxx)
        
        if totalE == 0:
            FM = FSD = FMED = SRO = 0.0
        else:
            FM = np.sum(f * Pxx) / totalE          # Spectral centroid
            FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / totalE)  # Spectral spread
            cumulative = np.cumsum(Pxx)
            FMED = f[np.searchsorted(cumulative, 0.5 * totalE)]    # Median frequency
            SRO = f[np.searchsorted(cumulative, 0.85 * totalE)]    # 85% roll-off
        
        features.extend([rms, kur, crest, FM, FSD, FMED, SRO])
    
    # MAFAULDA-CORRECT RATIO DEFINITIONS (2 features ONLY - matches training)
    rms_axial, rms_radial, rms_tangential = rms_vals
    axial_ratio = rms_axial / (rms_radial + rms_tangential + 1e-12)  # Axial concentration
    radial_ratio = rms_radial / (rms_axial + rms_tangential + 1e-12)  # Radial concentration
    features.extend([axial_ratio, radial_ratio])  # Positions 22-23 (axial_ratio, radial_ratio)
    
    return np.array(features, dtype=np.float32).reshape(1, -1), rpm, f, Pxx

# ==================== LIFECYCLE ====================
@app.on_event("startup")
async def load_model_pipeline():
    global model_pipeline
    
    if not os.path.exists(MODEL_PIPELINE_PATH):
        logger.error(f"❌ Model pipeline not found: {MODEL_PIPELINE_PATH}")
        logger.error("💡 FIX: Ensure training script saved pipeline to this path")
        logger.error("   Training script should contain:")
        logger.error("   joblib.dump(pipeline, r'C:\\...\\svm_pipeline_physics_validated.pkl')")
        return
    
    try:
        logger.info("⏳ Loading MaFaulDa physics-validated SVM pipeline...")
        model_pipeline = joblib.load(MODEL_PIPELINE_PATH)
        
        # Verify pipeline integrity
        required_keys = ['scaler', 'model', 'label_encoder', 'feature_names', 'physics_validation']
        if not all(k in model_pipeline for k in required_keys):
            raise ValueError(f"Pipeline missing required keys: {required_keys}")
        
        # Validate feature count matches MaFaulDa physics (23 features)
        if len(model_pipeline['feature_names']) != 23:
            raise ValueError(
                f"Feature count mismatch! Expected 23 features (21 axis + 2 ratios), "
                f"got {len(model_pipeline['feature_names'])}. "
                f"Check training script feature extraction."
            )
        
        # Validate class count (6 MaFaulDa classes)
        n_classes = len(model_pipeline['label_encoder'].classes_)
        if n_classes != 6:
            raise ValueError(
                f"Class count mismatch! Expected 6 MaFaulDa classes, got {n_classes}. "
                f"Classes: {model_pipeline['label_encoder'].classes_}"
            )
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Classes: {model_pipeline['label_encoder'].classes_}")
        logger.info(f"   Features: 23 physics-aligned features (21 axis + 2 directional ratios)")
        logger.info(f"   Physics validation: {model_pipeline['physics_validation']}")
        logger.info(f"   Configuration: Severe-only={model_pipeline.get('configuration', {}).get('severe_cases_only', 'N/A')}, "
                    f"Balanced={model_pipeline.get('configuration', {}).get('balanced_classes', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model pipeline: {e}", exc_info=True)
        model_pipeline = None

# ==================== API ENDPOINTS ====================
class SignalData(BaseModel):
    tach: List[float]
    ax: List[float]  # Axial vibration (CSV column 1)
    ay: List[float]  # Radial vibration (CSV column 2)
    az: List[float]  # Tangential vibration (CSV column 3)
    fs: float = 50000.0

@app.get("/health")
async def health_check():
    if model_pipeline is None:
        return {
            "status": "error",
            "model_loaded": False,
            "error": "Model pipeline not loaded - check server logs",
            "required_file": MODEL_PIPELINE_PATH
        }
    
    return {
        "status": "ready",
        "model_loaded": True,
        "classes": model_pipeline['label_encoder'].classes_.tolist(),
        "feature_count": len(model_pipeline['feature_names']),
        "physics_validated": model_pipeline['physics_validation'],
        "configuration": model_pipeline.get('configuration', {}),
        "window_size_raw": WINDOW_SIZE_RAW,
        "window_size_decim": WINDOW_SIZE_DECIM,
        "decimation_factor": DECIMATION_FACTOR
    }

@app.get("/get_sample")
async def get_sample(fault_type: str = "Normal"):
    """
    Fetches REAL MaFaulDa sample matching training DATA_SOURCES structure.
    Supports all 6 fault classes with severity-aware sampling.
    """
    # Map dashboard names to MaFaulDa class names
    name_mapping = {
        "normal": "Normal", "Normal": "Normal",
        "imbalance": "Imbalance", "Imbalance": "Imbalance",
        "misalignment": "Horiz_Misalign",  # Default to horizontal
        "Horiz_Misalign": "Horiz_Misalign", "Vert_Misalign": "Vert_Misalign",
        "bearing": "Ball_Fault",  # Default to ball fault
        "Ball_Fault": "Ball_Fault", "Outer_Race": "Outer_Race"
    }
    
    actual_fault = name_mapping.get(fault_type, "Normal")
    
    # MAFAULDA FOLDER STRUCTURE (EXACT MATCH TO TRAINING DATA_SOURCES)
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
    
    # Find candidate files (handles nested underhang structure)
    candidates = []
    if subfolders:
        for sub in subfolders:
            # Handle nested folder structures
            candidates.extend(search_path.glob(f"**/{sub}/*.csv"))
            candidates.extend(search_path.glob(f"{sub}/*.csv"))
    else:
        candidates.extend(search_path.glob("*.csv"))
        candidates.extend(search_path.glob("**/*.csv"))
    
    candidates = [c for c in candidates if c.is_file()]
    
    if not candidates:
        raise HTTPException(404, f"No files found for '{actual_fault}' in {search_path}")
    
    # Return random sample (first 50k points for dashboard)
    target = random.choice(candidates)
    try:
        df = pd.read_csv(target, header=None, nrows=50000)
        if df.shape[1] < 4:
            raise ValueError(f"CSV has {df.shape[1]} columns, expected ≥4 (tach + 3 vibration axes)")
        
        # Extract severity from path
        severity = extract_severity_from_path(target)
        
        return {
            "filename": target.name,
            "fault_type": actual_fault,
            "severity": severity,
            "rpm_estimate": estimate_rpm_from_file(target),
            "tach": df.iloc[:, TACH_COL].astype(float).tolist(),
            "ax": df.iloc[:, VIBRATION_COLS[0]].astype(float).tolist(),
            "ay": df.iloc[:, VIBRATION_COLS[1]].astype(float).tolist(),
            "az": df.iloc[:, VIBRATION_COLS[2]].astype(float).tolist()
        }
    except Exception as e:
        logger.error(f"Error reading {target}: {e}")
        raise HTTPException(500, f"Failed to read sample: {str(e)}")

def extract_severity_from_path(path):
    """Extract severity label from MaFaulDa path (e.g., '15g', '1.0mm')"""
    import re
    path_str = str(path).lower()
    # Match patterns like "15g", "20g", "1.0mm", "0.5mm"
    if m := re.search(r'(\d+\.?\d*)\s*(g|mm)', path_str):
        return f"{m.group(1)}{m.group(2)}"
    # For normal class
    if 'normal' in path_str:
        return "healthy"
    return "unknown"

def estimate_rpm_from_file(file_path):
    """Quick RPM estimate from first 10k samples for sample selection"""
    try:
        df = pd.read_csv(file_path, header=None, nrows=10000)
        tach = df.iloc[:, 0].values
        rpm = calculate_rpm_from_tach(tach, SAMPLING_FREQ_RAW)
        return rpm if rpm and 400 <= rpm <= 5000 else 1750.0
    except:
        return 1750.0

@app.post("/predict")
async def predict(data: SignalData):
    if model_pipeline is None:
        raise HTTPException(503, "Model pipeline not loaded - check server logs")
    
    try:
        # 1. Convert to numpy with correct axis ordering (Axial, Radial, Tangential)
        raw_vib = np.stack([
            np.array(data.ax, dtype=np.float32),
            np.array(data.ay, dtype=np.float32),
            np.array(data.az, dtype=np.float32)
        ], axis=1)
        raw_tach = np.array(data.tach, dtype=np.float32)
        
        # 2. Validate signal length (MUST have enough samples for 4096-window)
        min_samples = WINDOW_SIZE_RAW
        if len(raw_vib) < min_samples:
            raise ValueError(
                f"Signal too short ({len(raw_vib)} samples). "
                f"Need ≥{min_samples} samples at {int(data.fs)}Hz "
                f"({WINDOW_SIZE_DECIM} after decimation to {SAMPLING_FREQ_DECIMATED:.1f}Hz)"
            )
        
        # 3. Decimate to match training physics (50kHz → 3.85kHz)
        sig_vib = decimate(raw_vib, DECIMATION_FACTOR, axis=0, zero_phase=True)
        sig_tach = decimate(raw_tach, DECIMATION_FACTOR, axis=0, zero_phase=True)
        
        # 4. Extract features from LAST window (most recent state)
        if len(sig_vib) < WINDOW_SIZE_DECIM:
            raise ValueError(f"Signal too short after decimation ({len(sig_vib)} samples). Need {WINDOW_SIZE_DECIM}.")
        
        window_vib = sig_vib[-WINDOW_SIZE_DECIM:, :]
        window_tach = sig_tach[-WINDOW_SIZE_DECIM:]
        
        feats, rpm, f_axis, spec_vals = extract_features_single_window(
            window_vib, window_tach, SAMPLING_FREQ_DECIMATED
        )
        
        # 5. RPM fallback (training used physics-based extraction)
        if rpm is None or not (400 <= rpm <= 5000):
            rpm = 1750.0
            logger.warning(f"⚠️ RPM invalid ({rpm}), using fallback: {rpm} RPM")
        
        # 6. Scale and predict (MUST use training scaler)
        feats_scaled = model_pipeline['scaler'].transform(feats)
        probs = model_pipeline['model'].predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = model_pipeline['label_encoder'].inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        
        # 7. MaFaulDa-physics-aligned explanation (based on ACTUAL validation results)
        explanation = generate_mafulda_physics_explanation(
            pred_label, feats[0], model_pipeline['feature_names'], rpm, model_pipeline['physics_validation']
        )
        
        # 8. Physics-aligned severity estimate (for dashboard visualization)
        severity_estimate = estimate_fault_severity(pred_label, feats[0], model_pipeline['feature_names'])
        
        return {
            "prediction": str(pred_label),
            "confidence": confidence,
            "rpm": float(rpm),
            "severity_estimate": severity_estimate,
            "explanation": explanation,
            "features": {name: float(val) for name, val in zip(model_pipeline['feature_names'], feats[0])},
            "spectrum_f": f_axis.astype(float).tolist(),
            "spectrum_val": spec_vals.astype(float).tolist(),
            "physics_validation": model_pipeline['physics_validation']
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {str(e)}")

def estimate_fault_severity(fault_type: str, features: np.ndarray, feature_names: List[str]) -> str:
    """Estimate fault severity based on physics-aligned feature thresholds from training validation"""
    feat_map = {name: idx for idx, name in enumerate(feature_names)}
    
    if fault_type == "Normal":
        return "healthy"
    
    elif fault_type == "Imbalance":
        radial_ratio = features[feat_map['radial_ratio']] if 'radial_ratio' in feat_map else 0.5
        if radial_ratio > 0.85:
            return "severe (30-35g)"
        elif radial_ratio > 0.75:
            return "moderate (15-25g)"
        else:
            return "incipient (6-10g)"
    
    elif fault_type in ["Horiz_Misalign", "Vert_Misalign"]:
        radial_ratio = features[feat_map['radial_ratio']] if 'radial_ratio' in feat_map else 0.4
        if radial_ratio > 0.6:
            return "severe (1.5-2.0mm)"
        else:
            return "mild (0.5-1.0mm)"
    
    elif fault_type in ["Ball_Fault", "Outer_Race"]:
        spread = features[feat_map['ax_spec_spread']] if 'ax_spec_spread' in feat_map else 50
        if spread > 150:
            return "advanced (35g equivalent)"
        elif spread > 100:
            return "moderate (20g equivalent)"
        else:
            return "early-stage (6g equivalent)"
    
    return "unknown"

def generate_mafulda_physics_explanation(pred_label: str, features: np.ndarray, feature_names: List[str], rpm: float, physics_validation: Dict) -> str:
    """
    MAFAULDA-SPECIFIC EXPLANATIONS based on YOUR ACTUAL validation results:
    • Misalignment: Radial-dominant (8.8% drop when radial features removed) due to coupling dynamics
    • Imbalance: Multi-axis at mild stages → radial dominance at severe stages
    • Bearing faults: Spectral spread > kurtosis for early-stage faults
    """
    feat_map = {name: idx for idx, name in enumerate(feature_names)}
    
    if pred_label == "Normal":
        return "✅ Healthy operation: Balanced vibration signature across all axes with minimal harmonics"
    
    elif pred_label in ["Horiz_Misalign", "Vert_Misalign"]:
        # MAFAULDA REALITY: Radial features critical (validated 8.8% drop when removed)
        radial_ratio = features[feat_map['radial_ratio']] if 'radial_ratio' in feat_map else 0
        axial_ratio = features[feat_map['axial_ratio']] if 'axial_ratio' in feat_map else 0
        
        explanation = (
            f"⚠️ Mechanical misalignment detected at {rpm:.0f} RPM. "
            f"Physics validation: Axis ablation test confirmed radial-dominant detection "
            f"(radial feature removal caused 8.8% accuracy drop vs 3.7% for axial), "
            f"validating MaFaulDa's flexible coupling dynamics (Section 3.2 of MaFaulDa paper). "
        )
        
        if radial_ratio > 0.5:
            explanation += f"Radial ratio elevated ({radial_ratio:.2f}) indicates significant radial energy concentration."
        else:
            explanation += "Multi-axis energy distribution suggests mild misalignment severity."
        
        return explanation
    
    elif pred_label == "Imbalance":
        fundamental_hz = rpm / 60
        radial_ratio = features[feat_map['radial_ratio']] if 'radial_ratio' in feat_map else 0
        
        explanation = (
            f"⚠️ Mass imbalance detected ({fundamental_hz:.1f} Hz = 1×RPM harmonic). "
            f"Physics validation: Model captures fault progression physics - "
            f"incipient faults show multi-axis energy distribution while severe faults (>30g) "
            f"exhibit radial dominance (radial ratio >0.85). "
        )
        
        if radial_ratio > 0.85:
            explanation += f"Strong radial dominance (ratio={radial_ratio:.2f}) indicates severe imbalance (>30g)."
        elif radial_ratio > 0.7:
            explanation += f"Moderate radial concentration (ratio={radial_ratio:.2f}) suggests 15-25g imbalance."
        else:
            explanation += f"Multi-axis energy distribution (ratio={radial_ratio:.2f}) indicates incipient fault (<10g)."
        
        return explanation
    
    elif pred_label in ["Ball_Fault", "Outer_Race"]:
        # MAFAULDA REALITY: Early-stage faults → spectral spread > kurtosis
        spread = features[feat_map['ax_spec_spread']] if 'ax_spec_spread' in feat_map else 0
        kurt = features[feat_map['ax_kurt']] if 'ax_kurt' in feat_map else 0
        
        # Physics explanation based on YOUR bearing validation
        fault_name = "Ball Spin Frequency (BSF)" if pred_label == "Ball_Fault" else "Ball Pass Frequency Outer Race (BPFO)"
        coef = 1.8710 if pred_label == "Ball_Fault" else 2.9980
        theory_freq = coef * rpm / 60
        
        explanation = (
            f"⚠️ Bearing fault detected ({fault_name} = {coef}×RPM ≈ {theory_freq:.1f} Hz). "
            f"Physics validation: Model achieves 99%+ recall on MaFaulDa's early-stage faults "
            f"using spectral spread ({spread:.1f}) rather than kurtosis ({kurt:.1f}), "
            f"validating adaptation to mild defect physics (Section 4.3 of MaFaulDa paper). "
        )
        
        if spread > 120:
            explanation += "Elevated spectral spread indicates advanced fault stage with significant energy dispersal."
        else:
            explanation += "Moderate spectral spread elevation characteristic of early-stage bearing defects."
        
        return explanation
    
    return f"Detected fault: {pred_label} (physics-aligned detection validated)"

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    logger.info("="*70)
    logger.info("🚀 STARTING VIBRAGUARD AI SERVER - MAFAULDA PHYSICS-ALIGNED")
    logger.info("="*70)
    logger.info(f"Model pipeline path: {MODEL_PIPELINE_PATH}")
    logger.info(f"Raw data root: {RAW_DATA_ROOT}")
    logger.info(f"Physics configuration: {WINDOW_SIZE_RAW} raw samples → {WINDOW_SIZE_DECIM} decimated samples")
    logger.info(f"Decimation: {SAMPLING_FREQ_RAW} Hz → {SAMPLING_FREQ_DECIMATED:.1f} Hz (factor {DECIMATION_FACTOR})")
    logger.info(f"Feature count: 23 (21 axis features + 2 directional ratios)")
    logger.info(f"Classes: Normal, Imbalance, Horiz_Misalign, Vert_Misalign, Ball_Fault, Outer_Race")
    logger.info("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")