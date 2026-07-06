import sys

import numpy as np
from PyEMD import EMD
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .config import FS, MAX_IMFS, WINDOW_LEN, WINDOW_STEP
from .paths import SRC_ROOT


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from utils.emd_processing import sig_to_imf as user_sig_to_imf
except Exception:
    user_sig_to_imf = None


def sig_to_imf_paper(signal, max_imfs=MAX_IMFS):
    if user_sig_to_imf is not None:
        try:
            imfs_filtered, residue, reconstructed = user_sig_to_imf(signal, max_imfs=max_imfs)
            if reconstructed is not None and len(reconstructed) == len(signal):
                imfs_filtered = (
                    np.atleast_2d(imfs_filtered)
                    if getattr(imfs_filtered, "size", 0)
                    else np.array([])
                )
                residue = residue if residue is not None else np.zeros_like(signal)
                return imfs_filtered, residue, reconstructed
        except Exception:
            pass

    emd = EMD()
    emd.emd(signal)
    imfs, residue = emd.get_imfs_and_residue()
    if imfs is None or (hasattr(imfs, "size") and imfs.size == 0):
        return np.array([]), residue if residue is not None else np.zeros_like(signal), signal.copy()

    imfs = np.atleast_2d(imfs)
    if imfs.shape[0] < max_imfs:
        pad_rows = np.zeros((max_imfs - imfs.shape[0], imfs.shape[1]))
        imfs = np.vstack([imfs, pad_rows])
    else:
        imfs = imfs[:max_imfs]

    if imfs.shape[0] > 1:
        imfs_filtered = imfs[1:max_imfs]
        reconstructed = np.sum(imfs_filtered, axis=0) + (residue if residue is not None else 0.0)
    else:
        imfs_filtered = np.array([])
        reconstructed = residue if residue is not None else signal.copy()

    return imfs_filtered, residue if residue is not None else np.zeros_like(signal), reconstructed


def segment_signal(signal, window=WINDOW_LEN, step=WINDOW_STEP):
    starts = range(0, len(signal) - window + 1, step)
    return np.array([signal[start : start + window] for start in starts])


def temporal_features(signal):
    if signal.size == 0:
        return np.zeros(7)
    mean_value = np.mean(signal)
    std_value = np.std(signal)
    return np.array(
        [
            mean_value,
            std_value,
            float(skew(signal)),
            float(kurtosis(signal)),
            float(np.ptp(signal)),
            float(np.sqrt(np.mean(signal**2))),
            float(np.sum(signal**2)),
        ],
        dtype=float,
    )


def compute_psd(signal, fs=FS, nperseg=1024):
    nperseg = min(len(signal), nperseg)
    if nperseg < 8:
        return np.array([0.0]), np.array([np.sum(signal**2) + 1e-12])
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)
    return frequencies, np.maximum(psd, 1e-12)


def freq_a_features(signal, fs=FS):
    if signal.size == 0:
        return np.zeros(6)
    frequencies, psd = compute_psd(signal, fs)
    total_energy = np.sum(psd)
    fm = np.sum(frequencies * psd) / (total_energy + 1e-12)
    fsd = np.sqrt(np.sum(((frequencies - fm) ** 2) * psd) / (total_energy + 1e-12))
    fsk = np.sum(((frequencies - fm) ** 3) * psd) / ((total_energy + 1e-12) * (fsd**3 + 1e-12))
    fkr = np.sum(((frequencies - fm) ** 4) * psd) / ((total_energy + 1e-12) * (fsd**4 + 1e-12))
    cumulative = np.cumsum(psd)
    median_index = np.searchsorted(cumulative, 0.5 * total_energy)
    fmed = float(frequencies[min(median_index, len(frequencies) - 1)])
    return np.array([fm, fsd, fsk, fkr, total_energy, fmed], dtype=float)


def freq_b_features(signal, fs=FS):
    if signal.size == 0:
        return np.zeros(8)
    frequencies, psd = compute_psd(signal, fs)
    total_energy = np.sum(psd)
    spectral_centroid = np.sum(frequencies * psd) / (total_energy + 1e-12)
    normalized = psd / (np.sum(psd) + 1e-12)
    spectral_flatness = float(np.sum((np.diff(normalized, prepend=normalized[0])) ** 2))
    cumulative = np.cumsum(psd)
    rolloff_index = np.searchsorted(cumulative, 0.85 * total_energy)
    rolloff = float(frequencies[min(rolloff_index, len(frequencies) - 1)])
    geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd)
    flatness_ratio = float(geo_mean / (arith_mean + 1e-12))
    crest_ratio = float(np.max(psd) / (arith_mean + 1e-12))
    if len(psd) >= 2:
        indexes = np.arange(1, len(psd) + 1)
        decay = float(np.sum((psd[:-1] - psd[1:]) / (indexes[:-1] + 1e-12)) / (np.sum(psd) + 1e-12))
    else:
        decay = 0.0
    try:
        slope, _ = np.polyfit(frequencies, psd, 1)
    except Exception:
        slope = 0.0
    spread = float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / (total_energy + 1e-12)))
    return np.array(
        [spectral_centroid, spectral_flatness, rolloff, flatness_ratio, crest_ratio, decay, float(slope), spread],
        dtype=float,
    )


def compose_feature_sets_from_reconstructed(reconstructed):
    temporal = temporal_features(reconstructed)
    freq_a = freq_a_features(reconstructed)
    freq_b = freq_b_features(reconstructed)
    return {
        "F1": temporal,
        "F2": freq_a,
        "F3": freq_b,
        "F4": np.concatenate((temporal, freq_a)),
        "F5": np.concatenate((freq_a, freq_b)),
        "F6": np.concatenate((temporal, freq_b)),
        "F7": np.concatenate((temporal, freq_a, freq_b)),
    }
