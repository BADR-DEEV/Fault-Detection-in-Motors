# preprocessing.py

import numpy as np
from scipy.signal import resample_poly, welch
from training.config import TARGET_FS, WINDOW_SEC, ORDER_BANDS, ORDER_BINS


def normalize_axes(sig):
    """
    RMS normalize each axis (SKF-style)
    """
    if sig.ndim == 1:
        sig = sig[:, None]

    rms = np.sqrt(np.mean(sig ** 2, axis=0)) + 1e-8
    return sig / rms


def estimate_rpm(sig, fs):
    """
    Tachless RPM estimation
    """
    f, Pxx = welch(sig, fs=fs, nperseg=4096)
    mask = (f > 10) & (f < 80)
    if not mask.any():
        return 0.0
    return f[mask][np.argmax(Pxx[mask])] * 60


def window_signal(sig, fs):
    win = int(WINDOW_SEC * fs)
    if sig.shape[0] < win:
        sig = np.pad(sig, ((0, win - sig.shape[0]), (0, 0)), mode="edge")
    start = (sig.shape[0] - win) // 2
    return sig[start:start + win]


def order_spectrum(sig, fs, rpm, max_order):
    shaft_hz = max(rpm / 60.0, 1.0)
    f, Pxx = welch(sig, fs=fs, nperseg=2048)
    orders = f / shaft_hz
    tgt = np.linspace(0, max_order, ORDER_BINS)
    spec = np.interp(tgt, orders, Pxx, left=0, right=0)
    return spec


def compute_multiband_orders(sig, fs, rpm):
    """
    Physics-guided features (this is what your model was missing)
    """
    bands = []

    for lo, hi in ORDER_BANDS:
        spec = order_spectrum(sig[:, 0], fs, rpm, hi)
        orders = np.linspace(0, hi, ORDER_BINS)

        mask = (orders >= lo) & (orders <= hi)
        band = spec[mask]

        band = np.log1p(band)
        band /= (np.sum(band) + 1e-8)

        bands.append(band)

    return np.stack(bands, axis=0)


def preprocess(raw_sig, original_fs):
    sig = normalize_axes(raw_sig)

    sig = resample_poly(sig, TARGET_FS, original_fs, axis=0)
    sig = window_signal(sig, TARGET_FS)

    rpm = estimate_rpm(sig[:, 0], TARGET_FS)

    features = compute_multiband_orders(sig, TARGET_FS, rpm)
    return features.astype(np.float32), rpm
