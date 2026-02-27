














# import logging
# from os import path
# import random
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scipy.signal as signal

# # ==========================================================
# # CONFIGURATION
# # ==========================================================
# import os
# import random
# from os import path

# RAW_DATA_ROOT = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda"

# WINDOW_SIZE = 132
# STRIDE = 99
# DECIMATION_FACTOR = 13        # 50% overlap for robust feature extraction
# VIBRATION_COLS = [1, 2, 3]    # Axial, Radial, Tangential underhang
# TACH_COL = 0
# AXIS_NAMES = ['Axial', 'Radial', 'Tangential']
# RANDOM_STATE = 42
# SAMPLING_FREQ_RAW = 50000
# SAMPLING_FREQ_DECIMATED = SAMPLING_FREQ_RAW / DECIMATION_FACTOR

# # Build correct directory path (NO leading slash)
# normal_dir = path.join(RAW_DATA_ROOT, "normal")

# # Get all CSV files inside
# files = [f for f in os.listdir(normal_dir) if f.endswith(".csv")]

# # Safety check
# if not files:
#     raise FileNotFoundError("No CSV files found in normal directory.")

# # Pick random file
# random_file_name = random.choice(files)

# # Full absolute path
# random_file = path.join(normal_dir, random_file_name)

# print("Selected file:", random_file)

# FS_RAW = 50000  # 50 kHz
# DECIMATION_FACTOR = 13
# FS_DEC = FS_RAW / DECIMATION_FACTOR

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s | %(levelname)-8s | %(message)s')
# logger = logging.getLogger()

# # ==========================================================
# # LOAD DATA
# # ==========================================================
# df = pd.read_csv(random_file, header=None)
# tach_signal = df.iloc[:, 0].values.astype(float)

# # ==========================================================
# # PULSE DETECTION (ROBUST + INTEGER SAFE)
# # ==========================================================
# def detect_tach_pulses(tach_signal: np.ndarray,
#                        sampling_freq: float) -> np.ndarray:
#     """
#     Detect tachometer rising edges using hysteresis filtering.
#     Returns INTEGER sample indices.
#     """
#     threshold = np.mean(tach_signal) + 0.5 * np.std(tach_signal)
#     binary = tach_signal > threshold

#     min_samples = int(sampling_freq / 5000)  # 5 kHz max pulse rate
#     rising_edges = []
#     last_edge = -min_samples

#     for i in range(1, len(binary) - 1):
#         if not binary[i - 1] and binary[i] and binary[i + 1]:
#             if i - last_edge > min_samples:
#                 rising_edges.append(int(i))
#                 last_edge = i

#     return np.array(rising_edges, dtype=np.int32)


# # ==========================================================
# # RPM CALCULATION
# # ==========================================================
# def compute_instantaneous_rpm(rising_edges: np.ndarray,
#                               sampling_freq: float):

#     if len(rising_edges) < 2:
#         return None, None

#     periods = np.diff(rising_edges) / sampling_freq
#     rpm_inst = 60.0 / periods

#     return rpm_inst, periods


# # ==========================================================
# # VISUALIZATION FUNCTIONS
# # ==========================================================
# def plot_tach_with_edges(tach_signal,
#                          rising_edges,
#                          sampling_freq,
#                          title,
#                          duration_sec=0.2):

#     samples_to_plot = int(duration_sec * sampling_freq)
#     tach = tach_signal[:samples_to_plot]
#     t = np.arange(len(tach)) / sampling_freq

#     plt.figure(figsize=(14, 5))
#     plt.plot(t, tach, linewidth=1.2, label="Tach Signal")

#     threshold = np.mean(tach) + 0.5 * np.std(tach)
#     plt.axhline(threshold, linestyle="--", label="Threshold")

#     valid_edges = rising_edges[rising_edges < samples_to_plot]
#     plt.scatter(t[valid_edges],
#                 tach[valid_edges],
#                 s=60,
#                 zorder=5,
#                 color='green', marker='o',
#                 label="Detected Rising Edges")

#     plt.title(title, fontsize=14, fontweight='bold')
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_instantaneous_rpm(rpm_inst, sampling_freq, title):

#     if rpm_inst is None:
#         logger.warning("Not enough pulses to compute RPM.")
#         return

#     t = np.arange(len(rpm_inst)) / sampling_freq

#     plt.figure(figsize=(12, 4))
#     plt.plot(rpm_inst, linewidth=1.3)
#     plt.title(title, fontsize=13, fontweight='bold')
#     plt.xlabel("Revolution Index")
#     plt.ylabel("Instantaneous RPM")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()



# raw_vib = df.values[:, VIBRATION_COLS]

# raw_tach = df.values[:, TACH_COL]

# # Decimate with anti-aliasing filter
# decimated_vib = scipy.signal.decimate(raw_vib, DECIMATION_FACTOR, axis=0, zero_phase=True)

# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_vibration_axes(signal: np.ndarray, fs: float, title: str = "Vibration Signals"):
#     """
#     Plots a 3-axis vibration signal in a single figure with subplots for Axial, Radial, and Tangential axes.
#     Colors are academic-standard: blue, orange, green.
    
#     Parameters:
#     - signal: np.ndarray, shape (N, 3) for Axial, Radial, Tangential
#     - fs: Sampling frequency in Hz
#     - title: Figure title
#     """
#     if signal.shape[1] != 3:
#         raise ValueError("Signal must have shape (N, 3) for Axial, Radial, Tangential axes")
    
#     t = np.arange(signal.shape[0]) / fs  # Total duration in seconds
#     axis_names = ["Axial", "Radial", "Tangential"]
#     colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green (academic)

#     plt.figure(figsize=(16, 8))
#     for i in range(3):
#         plt.subplot(3, 1, i+1)
#         plt.plot(t, signal[:, i], color=colors[i], linewidth=1.2)
#         plt.title(f"{axis_names[i]} Axis", fontsize=12, fontweight='bold', color=colors[i])
#         plt.xlabel("Time (seconds)")
#         plt.ylabel("Amplitude")
#         plt.xlim(0, t[-1])  # Show full signal duration
#         plt.grid(True, alpha=0.3)

#     plt.suptitle(title, fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# def visulize_vibration_signal(signal, fs, title):
#     """
#     Plots a time-domain signal with a title.
#     """
#     plt.figure(figsize=(14, 5))
#     plt.plot(signal, linewidth=1.2, label="Tach Signal")
#     plt.title(title, fontsize=14, fontweight='bold')
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# # ==========================================================
# # MAIN EXECUTION
# # ==========================================================
# if __name__ == "__main__":

#     visualize_vibration_axes(raw_vib, fs=50000, title="Raw Vibration Signal")
#     visualize_vibration_axes(decimated_vib, fs=3846, title="Decimated Vibration Signal")
#     # ---------------- RAW 50 kHz ----------------
#     rising_raw = detect_tach_pulses(tach_signal, FS_RAW)
#     rpm_raw, periods_raw = compute_instantaneous_rpm(rising_raw, FS_RAW)

#     plt.figure(figsize=(14, 5))
#     plt.plot(tach_signal, linewidth=1.2, label="Tach Signal")

#     plot_tach_with_edges(
#         tach_signal,

#         rising_raw,
        
#         FS_RAW,
#         title="Raw Tachometer Signal (50 kHz) with Detected Pulses"
    
#     )

#     plot_instantaneous_rpm(
#         rpm_raw,
#         FS_RAW,
#         title="Instantaneous RPM (50 kHz Tach)"
#     )

#     # ---------------- DECIMATED ----------------
#     tach_dec = signal.decimate(tach_signal, DECIMATION_FACTOR)
#     rising_dec = detect_tach_pulses(tach_dec, FS_DEC)
#     rpm_dec, periods_dec = compute_instantaneous_rpm(rising_dec, FS_DEC)

#     plot_tach_with_edges(
#         tach_dec,
#         rising_dec,
#         FS_DEC,
#         title=f"Decimated Tach Signal ({FS_DEC:.1f} Hz) with Detected Pulses"
#     )

#     plot_instantaneous_rpm(
#         rpm_dec,
#         FS_DEC,
#         title="Instantaneous RPM (Decimated Tach)"
#     )

#     logger.info(f"RAW pulses detected: {len(rising_raw)}")
#     logger.info(f"DEC pulses detected: {len(rising_dec)}")
#     logger.info(f"RPM: {rpm_dec}")
#     logger.info(f"RPM (raw): {rpm_raw}")