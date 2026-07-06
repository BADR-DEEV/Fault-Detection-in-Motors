"""
Physics-aligned feature extraction for MaFaulDa.

The feature vector has exactly 23 values:
- 7 time/frequency features per axis across axial, radial, and tangential axes.
- 2 directional energy ratios.
"""

import numpy as np
import scipy.stats as stats
from scipy.signal import welch


WINDOW_SIZE = 4096
STRIDE = 2048
DECIMATION_FACTOR = 13
VIBRATION_COLS = [4, 5, 6]
TACH_COL = 0
SAMPLING_FREQ_RAW = 50000
SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

AXES = ["ax", "rad", "tan"]
PER_AXIS_FEATURES = [
    "rms",
    "kurt",
    "crest",
    "spec_centroid",
    "spec_spread",
    "freq_median",
    "freq_rolloff85",
]
FEATURE_COLUMNS = [f"{axis}_{feature}" for axis in AXES for feature in PER_AXIS_FEATURES] + [
    "axial_ratio",
    "radial_ratio",
]

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
    """Extract 23 physics-aligned features and the estimated RPM."""
    rpm = calculate_rpm_from_tach(tach_signal, sampling_freq)
    features = []
    rms_vals = []

    for axis in range(3):
        signal = vib_signal[:, axis]

        rms = np.sqrt(np.mean(signal**2))
        kurtosis = stats.kurtosis(signal, fisher=False)
        crest = np.max(np.abs(signal)) / (rms + 1e-12)
        rms_vals.append(rms)

        nperseg = min(1024, len(signal))
        freqs, pxx = welch(signal, fs=sampling_freq, nperseg=nperseg)
        pxx = np.maximum(pxx, 1e-12)
        total_energy = np.sum(pxx)

        if total_energy == 0:
            features.extend([rms, kurtosis, crest, 0, 0, 0, 0])
            continue

        spectral_centroid = np.sum(freqs * pxx) / total_energy
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * pxx) / total_energy)
        cumulative = np.cumsum(pxx)
        median_frequency = freqs[np.searchsorted(cumulative, 0.5 * total_energy)]
        rolloff85 = freqs[np.searchsorted(cumulative, 0.85 * total_energy)]

        features.extend(
            [
                rms,
                kurtosis,
                crest,
                spectral_centroid,
                spectral_spread,
                median_frequency,
                rolloff85,
            ]
        )

    rms_axial = rms_vals[0]
    rms_radial = rms_vals[1]
    rms_tangential = rms_vals[2]

    axial_ratio = rms_axial / (rms_radial + rms_tangential + 1e-12)
    radial_ratio = rms_radial / (rms_axial + rms_tangential + 1e-12)
    features.extend([axial_ratio, radial_ratio])

    return features, rpm
