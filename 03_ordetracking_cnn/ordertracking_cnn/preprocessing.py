from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import interpolate

from .config import ORDERS_PER_REV, REVOLUTIONS_PER_WINDOW, SAMPLING_FREQ_RAW, TACH_COL, VIBRATION_COLS


class SpectralOrderPreprocessor:
    def __init__(self, orders_per_rev: int = ORDERS_PER_REV, revolutions: int = REVOLUTIONS_PER_WINDOW):
        self.orders_per_rev = orders_per_rev
        self.revolutions = revolutions
        self.window_size = orders_per_rev * revolutions

    def detect_tach_pulses(self, tach_signal: np.ndarray, sampling_freq: float) -> np.ndarray:
        threshold = np.mean(tach_signal) + 0.5 * np.std(tach_signal)
        binary = tach_signal > threshold
        min_samples = max(1, int(sampling_freq / 5000))
        rising_edges = []
        last_edge = -min_samples
        for index in range(1, len(binary) - 1):
            if not binary[index - 1] and binary[index] and binary[index + 1]:
                if index - last_edge > min_samples:
                    rising_edges.append(int(index))
                    last_edge = index
        return np.array(rising_edges, dtype=np.int32)

    def resample_to_orders(self, vib_signal: np.ndarray, tach_pulses: np.ndarray):
        if len(tach_pulses) < self.revolutions + 1:
            return None
        start_idx = int(tach_pulses[0])
        end_idx = int(tach_pulses[self.revolutions])
        if end_idx <= start_idx or (end_idx - start_idx) < (self.window_size * 0.3):
            return None

        original_indices = np.arange(start_idx, end_idx, dtype=np.float32)
        target_indices = np.linspace(start_idx, end_idx, self.window_size, dtype=np.float32)
        resampled = np.zeros((self.window_size, vib_signal.shape[1]), dtype=np.float32)

        for axis in range(vib_signal.shape[1]):
            try:
                interpolator = interpolate.interp1d(
                    original_indices,
                    vib_signal[start_idx:end_idx, axis],
                    kind="cubic",
                    fill_value="extrapolate",
                )
                resampled[:, axis] = interpolator(target_indices)
            except Exception:
                return None

        return resampled

    def fit_transform(self, files, labels, max_files_per_class: int):
        all_spectra = []
        all_labels = []
        all_sources = []
        class_counts = defaultdict(int)
        hanning_window = np.hanning(self.window_size)[:, None]

        for file_path, class_name in zip(files, labels):
            if class_counts[class_name] >= max_files_per_class and class_name != "Normal":
                continue

            try:
                dataframe = pd.read_csv(file_path, header=None)
                raw_vib = dataframe.values[:, VIBRATION_COLS].astype(np.float32)
                raw_tach = dataframe.values[:, TACH_COL].astype(np.float32)
                pulses = self.detect_tach_pulses(raw_tach, SAMPLING_FREQ_RAW)

                if class_name == "Normal":
                    max_windows = 40
                    pulse_step = max(1, self.revolutions // 2)
                else:
                    max_windows = 15
                    pulse_step = self.revolutions

                win_count = 0
                start_pulse = 0

                while win_count < max_windows and (start_pulse + self.revolutions + 1 <= len(pulses)):
                    pulse_slice = pulses[start_pulse : start_pulse + self.revolutions + 1]
                    window = self.resample_to_orders(raw_vib, pulse_slice)
                    if window is not None:
                        windowed_signal = window * hanning_window
                        spectrum = np.abs(np.fft.rfft(windowed_signal, axis=0)) / self.window_size
                        all_spectra.append(spectrum)
                        all_labels.append(class_name)
                        all_sources.append(str(file_path))
                        win_count += 1
                    start_pulse += pulse_step

                class_counts[class_name] += 1
            except Exception:
                continue

        X = np.array(all_spectra, dtype=np.float32)
        X_log = np.log1p(X * 1000)
        global_max = np.max(X_log) + 1e-8
        X_scaled = X_log / global_max
        return X_scaled, np.array(all_labels), np.array(all_sources)
