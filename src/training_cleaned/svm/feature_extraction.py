# utils/feature_extraction.py
"""
Physics-aligned feature extraction for MaFaulDa dataset.
Extracts exactly 23 handcrafted features that align with mechanical fault physics:
- 7 time/frequency features × 3 axes = 21 features
- 2 directional energy ratios = 2 features
Total = 23 features

All functions are pure, importable, and free of side effects.
"""

import numpy as np
import scipy.signal
import scipy.stats as stats
from scipy.signal import welch

# === PHYSICS CONSTANTS (MaFaulDa-specific) ===
WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 13
VIBRATION_COLS = [4,5,6]  # Axial, Radial, Tangential underhang
TACH_COL = 0
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

# === FEATURE COLUMN DEFINITION ===
# per_axis_features = [
#     'rms',                 # Root Mean Square - overall energy
#     'kurt',               # Kurtosis - impulsiveness (bearing faults)
#     'crest',              # Crest factor - peak-to-RMS ratio (shock severity)
#     'spec_centroid',       # Spectral centroid - energy center frequency
#     'spec_spread',         # Spectral spread - bandwidth of vibration energy
#     'freq_median',         # Median frequency - 50% power point
#     'freq_rolloff85'      # 85% roll-off frequency - high-frequency content
# ]
axes = ['ax', 'rad', 'tan']


per_axis_features = [

    'rms', 'kurt', 'crest', 'spec_centroid', 'spec_spread', 
    '1x_rpm', '2x_rpm', '3x_rpm', # Separates Imbalance and Misalignments
    'bpfo_amp', 'bsf_amp', 'ftf_amp' # Separates Bearing Faults
]
# 11 features * 3 axes + 2 ratios = 35 total features
FEATURE_COLUMNS = [f"{axis}_{feat}" for axis in axes for feat in per_axis_features] + ['axial_ratio', 'radial_ratio']

# Validate feature count
assert len(FEATURE_COLUMNS) == 35, f"Expected 35 features, got {len(FEATURE_COLUMNS)}"


# utils/feature_extraction.py
import numpy as np
import scipy.signal
import scipy.stats as stats
from scipy.signal import welch

# === PHYSICS CONSTANTS (MaFaulDa-specific) ===
WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 13
VIBRATION_COLS = [4,5,6]  # Axial, Radial, Tangential underhang
TACH_COL = 0
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

axes = ['ax', 'rad', 'tan']

# 7 features per axis
per_axis_features = [
    'rms', 'kurt', 'crest', 'spec_centroid', 'spec_spread', 'freq_median', 'freq_rolloff85'
]

# 7 features * 3 axes + 2 ratios = 23 total features
FEATURE_COLUMNS = [f"{axis}_{feat}" for axis in axes for feat in per_axis_features] + ['axial_ratio', 'radial_ratio']

# Validate feature count
assert len(FEATURE_COLUMNS) == 23, f"Expected 23 features, got {len(FEATURE_COLUMNS)}"

def calculate_rpm_from_tach(tach_signal, sampling_freq):
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

def extract_features_with_rpm(vib_signal, tach_signal, sampling_freq):
    """Physics-aligned feature extraction with bearing fault awareness"""
    rpm = calculate_rpm_from_tach(tach_signal, sampling_freq)
    features = []
    rms_vals = []

    for ax in range(3):
        signal = vib_signal[:, ax]
        
        # Time domain features
        rms = np.sqrt(np.mean(signal**2))
        kur = stats.kurtosis(signal, fisher=False)
        crest = np.max(np.abs(signal)) / (rms + 1e-12)
        rms_vals.append(rms)
        
        # Frequency domain features (PSD-based)
        nperseg = min(1024, len(signal))
        f, Pxx = welch(signal, fs=sampling_freq, nperseg=nperseg)
        Pxx = np.maximum(Pxx, 1e-12)
        totalE = np.sum(Pxx)
        
        if totalE == 0:
            features.extend([rms, kur, crest, 0, 0, 0, 0])
            continue

        # Spectral centroid (FM)
        FM = np.sum(f * Pxx) / totalE
        # Spectral spread (FSD)
        FSD = np.sqrt(np.sum(((f - FM) ** 2) * Pxx) / totalE)
        # Median frequency
        cumulative = np.cumsum(Pxx)
        FMED = f[np.searchsorted(cumulative, 0.5 * totalE)]
        # 85% roll-off frequency
        SRO = f[np.searchsorted(cumulative, 0.85 * totalE)]
        
        features.extend([rms, kur, crest, FM, FSD, FMED, SRO])
    
    # Standard MaFaulDa ratio definitions (2 features)
    rms_axial = rms_vals[0]
    rms_radial = rms_vals[1]
    rms_tangential = rms_vals[2]
    
    # Axial concentration ratio (key for misalignment physics)
    axial_ratio = rms_axial / (rms_radial + rms_tangential + 1e-12)
    # Radial concentration ratio (key for imbalance physics)
    radial_ratio = rms_radial / (rms_axial + rms_tangential + 1e-12)
    
    features.extend([axial_ratio, radial_ratio])  # ONLY 2 ratios
    
    return features, rpm