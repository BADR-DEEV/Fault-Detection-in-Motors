import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch, spectrogram
from scipy.stats import kurtosis
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ==========================================
#               CONFIGURATION
# ==========================================
FILE_PATH = '../../data/BMP_dataset/Noraml1.txt'
TARGET_FS = 4000.0  # ADXL357Z Sampling Rate (Hz)
MAX_G = 40.5        # Hardware limit of ADXL357Z (±40g)

# ==========================================
#          DATA LOADING & CLEANING
# ==========================================
def load_and_clean_data(filepath):
    print(f"📥 Loading file: {os.path.basename(filepath)}...")
    try:
        df = pd.read_csv(filepath, sep=',', on_bad_lines='skip', low_memory=False)
    except FileNotFoundError:
        print(f"❌ File not found at {filepath}")
        return None
        
    expected_cols = ['Time_us', 'X_g', 'Y_g', 'Z_g']
    if not all(col in df.columns for col in expected_cols):
        df = df.iloc[:, :4]
        df.columns = expected_cols

    # Convert to numeric, dropping any lines with text
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    # CRITICAL FIX: Drop physically impossible values (Parsing Artifacts)
    # The ADXL357Z cannot measure beyond ±40g.
    valid_mask = (
        (df['X_g'].between(-MAX_G, MAX_G)) & 
        (df['Y_g'].between(-MAX_G, MAX_G)) & 
        (df['Z_g'].between(-MAX_G, MAX_G))
    )
    df = df[valid_mask].reset_index(drop=True)
    
    df['Time_s'] = (df['Time_us'] - df['Time_us'].iloc[0]) / 1e6
    
    # AC-couple the individual axes (removes gravity offset, centers on 0)
    for axis in ['X_g', 'Y_g', 'Z_g']:
        df[f'{axis}_ac'] = df[axis] - df[axis].mean()
        
    return df

# ==========================================
#          STATE DETECTION
# ==========================================
def detect_motor_states(df):
    """Detects OFF vs ON state using rolling energy of all axes."""
    df['Energy'] = df['X_g_ac']**2 + df['Y_g_ac']**2 + df['Z_g_ac']**2
    window_size = int(TARGET_FS * 0.05) # 50ms window
    rolling_energy = df['Energy'].rolling(window=window_size, center=True).mean()
    
    # Baseline from the very first 0.5 seconds (Motor is OFF)
    baseline_samples = int(TARGET_FS * 0.5)
    if len(df) < baseline_samples:
        return 0, len(df)-1
        
    baseline_noise = rolling_energy.iloc[:baseline_samples].mean()
    threshold = baseline_noise + 1.0 # 1g^2 threshold above noise
    
    on_indices = df.index[rolling_energy > threshold].tolist()
    
    if not on_indices:
        return 0, len(df) - 1
        
    start_idx = on_indices[0]
    # Allow 0.5s for startup transient
    steady_state_idx = min(start_idx + int(TARGET_FS * 0.5), len(df) - 1)
    
    return start_idx, steady_state_idx

# ==========================================
#          EXPERT SYSTEM DIAGNOSIS
# ==========================================
def auto_diagnose_status(df_run, fs):
    """Rule-based diagnosis based on Steady-State data."""
    # Find the dominant axis (the one vibrating the most)
    rms_x = np.sqrt(np.mean(df_run['X_g_ac']**2))
    rms_y = np.sqrt(np.mean(df_run['Y_g_ac']**2))
    rms_z = np.sqrt(np.mean(df_run['Z_g_ac']**2))
    
    axes = {'X': rms_x, 'Y': rms_y, 'Z': rms_z}
    dom_axis_name = max(axes, key=axes.get)
    dom_axis_data = df_run[f'{dom_axis_name}_g_ac'].values
    
    rms = axes[dom_axis_name]
    peak = np.max(np.abs(dom_axis_data))
    crest_factor = peak / rms if rms > 0 else 0
    kurt = kurtosis(dom_axis_data, fisher=False)
    
    freqs, psd = welch(dom_axis_data, fs=fs, nperseg=4096)
    dominant_freq = freqs[np.argmax(psd)]
    
    # High Frequency Energy (> 500 Hz) for Dry Running/Friction
    hf_mask = freqs > 500
    hf_ratio = (np.sum(psd[hf_mask]) / np.sum(psd)) * 100 if np.sum(psd) > 0 else 0
    
    status = "UNKNOWN"
    reasoning = []
    
    # Extremely high vibration (Based on the snippet you provided, ±13g is massive)
    if rms > 4.0:
        reasoning.append(f"EXTREME Vibration Detected (RMS: {rms:.1f}g, Peak: {peak:.1f}g).")
        reasoning.append("Pump is operating in a dangerously violent state.")
        status = "SEVERE MECHANICAL DISTRESS / CAVITATION"
    
    if kurt > 4.0 or crest_factor > 5.0:
        reasoning.append(f"High Kurtosis ({kurt:.1f}) & Crest Factor ({crest_factor:.1f}) indicates aggressive impacting.")
        status = "IMPACTS (Bearing / Loose Impeller)"
    
    if hf_ratio > 25.0:
        reasoning.append(f"High-frequency energy ({hf_ratio:.1f}% > 500Hz) indicates severe friction (Dry Running Seal).")
        if "DISTRESS" not in status: status = "DRY RUNNING (FRICTION)"
        
    if status == "UNKNOWN" and rms < 0.5:
        reasoning.append("Vibration is stable and within normal limits.")
        status = "NORMAL / HEALTHY"
        
    stats = {
        'Dom. Axis': dom_axis_name,
        'RMS (g)': rms, 
        'Peak (g)': peak, 
        'Crest Factor': crest_factor, 
        'Kurtosis': kurt, 
        'Dom. Freq (Hz)': dominant_freq, 
        'HF Energy (%)': hf_ratio
    }
    
    return status, reasoning, stats

# ==========================================
#          VISUALIZATION
# ==========================================
def compute_fft(data, fs):
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(data)) / n * 2 # Multiply by 2 for single-sided amplitude
    return freqs, fft_mag

def plot_full_analysis(df, df_run, start_idx, steady_state_idx, dom_axis_name):
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    # =====================================================
    # 1. FFT (BIG WINDOW)
    # =====================================================
    plt.figure(figsize=(16, 6))

    freqs_x, fft_x = compute_fft(df_run['X_g_ac'].values, TARGET_FS)
    freqs_y, fft_y = compute_fft(df_run['Y_g_ac'].values, TARGET_FS)
    freqs_z, fft_z = compute_fft(df_run['Z_g_ac'].values, TARGET_FS)

    plt.plot(freqs_x, fft_x, color='red', label='X Axis', lw=1.2)
    plt.plot(freqs_y, fft_y, color='blue', label='Y Axis', lw=1.2)
    plt.plot(freqs_z, fft_z, color='purple', label='Z Axis', lw=1.2)

    plt.title('Frequency Spectrum (Amplitude vs Frequency)', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude (g)', fontsize=12)
    plt.xlim(0, 2000)
    # =====================================================
# Add Harmonics (1X, 2X)
# =====================================================
    rpm = 2900
    f_1x = rpm / 60.0
    f_2x = 2 * f_1x

    # Plot vertical harmonic lines
    plt.axvline(f_1x, color='black', linestyle='--', linewidth=2, label=f'1X ({f_1x:.1f} Hz)')
    plt.axvline(f_2x, color='green', linestyle='--', linewidth=2, label=f'2X ({f_2x:.1f} Hz)')

    # Annotate them clearly
    plt.text(f_1x, plt.ylim()[1]*0.8, '1X', color='black', ha='center', fontsize=11, fontweight='bold')
    plt.text(f_2x, plt.ylim()[1]*0.7, '2X', color='green', ha='center', fontsize=11, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 2. 2D SPECTROGRAM (BIG WINDOW)
    # =====================================================
    plt.figure(figsize=(14, 8))

    dom_data = df_run[f'{dom_axis_name}_g_ac'].values
    f, t_spec, Sxx = spectrogram(dom_data, fs=TARGET_FS, nperseg=1024, noverlap=512)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    pcm = plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='jet')

    plt.title(f'2D Spectrogram (Dominant Axis: {dom_axis_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(pcm, label='Power (dB)')
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 3. 3D SPECTROGRAM (SEPARATE BIG WINDOW)
    # =====================================================
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample to avoid lag
    max_t_bins, max_f_bins = 70, 70
    t_step = max(1, len(t_spec) // max_t_bins)
    f_step = max(1, len(f) // max_f_bins)

    t_down = t_spec[::t_step]
    f_down = f[::f_step]
    Sxx_down = Sxx_dB[::f_step, ::t_step]

    T, F = np.meshgrid(t_down, f_down)

    surf = ax.plot_surface(T, F, Sxx_down, cmap='jet', edgecolor='none')

    ax.set_title(f'3D Spectrogram (Dominant Axis: {dom_axis_name})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Power (dB)')
    ax.view_init(elev=35, azim=-50)

    fig.colorbar(surf, shrink=0.6, aspect=10, label='Power (dB)')
    plt.tight_layout()
    plt.show()
# ==========================================
#          MAIN EXECUTION
# ==========================================
def main():
    print("="*60)
    print(" 🛠️ MOTOR VIBRATION DIAGNOSTIC SYSTEM")
    print("="*60)
    
    df = load_and_clean_data(FILE_PATH)
    if df is None or len(df) < 100:
        print("❌ Not enough valid data to process.")
        return
        
    start_idx, steady_state_idx = detect_motor_states(df)
    df_run = df.iloc[steady_state_idx:].copy()
    
    print("\n⚙️ Running Automated Diagnosis...")
    status, reasoning, stats = auto_diagnose_status(df_run, TARGET_FS)
    
    print("\n" + "="*60)
    print(" 🚨 AUTOMATED DIAGNOSTIC REPORT")
    print("="*60)
    print(f" ► MOST LIKELY STATUS : {status}")
    print(" ► REASONING:")
    for r in reasoning:
        print(f"    - {r}")
        
    print(f"\n 📊 SIGNAL STATISTICS (Dominant Axis: {stats['Dom. Axis']}):")
    for k, v in stats.items():
        if k != 'Dom. Axis':
            print(f"    {k:<15}: {v:.2f}")
    print("="*60)
    
    plot_full_analysis(df, df_run, start_idx, steady_state_idx, stats['Dom. Axis'])

if __name__ == '__main__':
    main()









































































#     import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from scipy.signal import welch, spectrogram, find_peaks
# from scipy.stats import kurtosis
# import warnings

# warnings.filterwarnings('ignore')
# plt.style.use('seaborn-v0_8-darkgrid')

# # ==========================================
# #               CONFIGURATION
# # ==========================================
# FILE_PATH = '../../data/BMP_dataset/New40gNormal.txt'
# TARGET_FS = 4000.0  
# MAX_G = 40.5        

# # ==========================================
# #          DATA LOADING & CLEANING
# # ==========================================
# def load_and_clean_data(filepath):
#     print(f"📥 Loading file: {os.path.basename(filepath)}...")
#     try:
#         df = pd.read_csv(filepath, sep=',', on_bad_lines='skip', low_memory=False)
#     except FileNotFoundError:
#         print(f"❌ File not found at {filepath}")
#         return None
        
#     expected_cols = ['Time_us', 'X_g', 'Y_g', 'Z_g']
#     if not all(col in df.columns for col in expected_cols):
#         df = df.iloc[:, :4]
#         df.columns = expected_cols

#     for col in expected_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df = df.dropna().reset_index(drop=True)
    
#     valid_mask = (
#         (df['X_g'].between(-MAX_G, MAX_G)) & 
#         (df['Y_g'].between(-MAX_G, MAX_G)) & 
#         (df['Z_g'].between(-MAX_G, MAX_G))
#     )
#     df = df[valid_mask].reset_index(drop=True)
    
#     df['Time_s'] = (df['Time_us'] - df['Time_us'].iloc[0]) / 1e6
    
#     for axis in ['X_g', 'Y_g', 'Z_g']:
#         df[f'{axis}_ac'] = df[axis] - df[axis].mean()
        
#     return df

# # ==========================================
# #          STATE DETECTION
# # ==========================================
# def detect_motor_states(df):
#     df['Energy'] = df['X_g_ac']**2 + df['Y_g_ac']**2 + df['Z_g_ac']**2
#     window_size = int(TARGET_FS * 0.05)
#     rolling_energy = df['Energy'].rolling(window=window_size, center=True).mean()
    
#     baseline_samples = int(TARGET_FS * 0.5)
#     if len(df) < baseline_samples:
#         return 0, len(df)-1
        
#     baseline_noise = rolling_energy.iloc[:baseline_samples].mean()
#     threshold = baseline_noise + 1.0 
    
#     on_indices = df.index[rolling_energy > threshold].tolist()
#     if not on_indices:
#         return 0, len(df) - 1
        
#     start_idx = on_indices[0]
#     steady_state_idx = min(start_idx + int(TARGET_FS * 0.5), len(df) - 1)
    
#     return start_idx, steady_state_idx

# # ==========================================
# #          EXPERT SYSTEM DIAGNOSIS
# # ==========================================
# def compute_fft(data, fs):
#     n = len(data)
#     freqs = np.fft.rfftfreq(n, d=1/fs)
#     fft_mag = np.abs(np.fft.rfft(data)) / n * 2 
#     return freqs, fft_mag

# def auto_diagnose_status(df_run, fs):
#     rms_x = np.sqrt(np.mean(df_run['X_g_ac']**2))
#     rms_y = np.sqrt(np.mean(df_run['Y_g_ac']**2))
#     rms_z = np.sqrt(np.mean(df_run['Z_g_ac']**2))
    
#     axes = {'X': rms_x, 'Y': rms_y, 'Z': rms_z}
#     dom_axis_name = max(axes, key=axes.get)
#     dom_axis_data = df_run[f'{dom_axis_name}_g_ac'].values
    
#     rms = axes[dom_axis_name]
#     peak = np.max(np.abs(dom_axis_data))
#     crest_factor = peak / rms if rms > 0 else 0
#     kurt = kurtosis(dom_axis_data, fisher=False)
    
#     freqs, fft_mag = compute_fft(dom_axis_data, fs)
    
#     # Peak Detection for Diagnosis logic
#     peaks_idx, _ = find_peaks(fft_mag, height=max(fft_mag)*0.2, distance=10)
#     top_freqs = freqs[peaks_idx]
    
#     status = "UNKNOWN"
#     reasoning = []
    
#     if rms > 4.0:
#         reasoning.append(f"EXTREME Vibration Detected (RMS: {rms:.1f}g, Peak: {peak:.1f}g).")
#         status = "SEVERE MECHANICAL DISTRESS"
    
#     if kurt > 4.0 or crest_factor > 5.0:
#         reasoning.append(f"High Kurtosis ({kurt:.1f}) indicates aggressive impacting.")
#         status = "IMPACTS (Bearing / Loose Impeller)"
        
#     # Specific Frequency Rules (120Hz & 400Hz)
#     has_120 = any(115 <= f <= 125 for f in top_freqs)
#     has_400 = any(390 <= f <= 410 for f in top_freqs)
    
#     if has_120:
#         reasoning.append("Peak near 120 Hz detected. This often indicates Misalignment or 2x Line Frequency electrical issues.")
#     if has_400:
#         reasoning.append("Large peak near 400 Hz detected. Highly characteristic of internal component ringing/rubbing (Expanded Impeller scraping housing).")
        
#     stats = {
#         'Dom. Axis': dom_axis_name,
#         'RMS (g)': rms, 
#         'Peak (g)': peak, 
#         'Crest Factor': crest_factor, 
#         'Kurtosis': kurt, 
#         'Dom. Freq (Hz)': freqs[np.argmax(fft_mag)]
#     }
    
#     return status, reasoning, stats

# # ==========================================
# #          VISUALIZATION
# # ==========================================
# def plot_full_analysis(df, df_run, start_idx, steady_state_idx, dom_axis_name):
#     fig = plt.figure(figsize=(18, 16))
#     gs = GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1.5, 2, 2])
    
#     time_start = df['Time_s'].iloc[start_idx]
#     time_steady = df['Time_s'].iloc[steady_state_idx]
    
#     # 1. TIME DOMAIN
#     ax_time = fig.add_subplot(gs[0, :])
#     ax_time.plot(df['Time_s'], df['X_g_ac'], label='X Axis', color='red', alpha=0.7, lw=0.5)
#     ax_time.plot(df['Time_s'], df['Y_g_ac'], label='Y Axis', color='blue', alpha=0.7, lw=0.5)
#     ax_time.plot(df['Time_s'], df['Z_g_ac'], label='Z Axis', color='purple', alpha=0.7, lw=0.5)
#     ax_time.axhline(0, color='green', linestyle='--', lw=2, label='Zero Reference')
#     ax_time.axvline(time_start, color='black', linestyle=':', lw=2, label='Motor ON')
#     ax_time.axvline(time_steady, color='orange', linestyle=':', lw=2, label='Steady-State')
#     ax_time.set_title('Time Domain: AC-Coupled Acceleration', fontsize=12, fontweight='bold')
#     ax_time.set_ylabel('Amplitude (g)')
#     ax_time.legend(loc='upper right')
    
#     # 2. FREQUENCY VS AMPLITUDE (With Auto-Peak Detection)
#     ax_fft = fig.add_subplot(gs[1, :])
    
#     colors = {'X': 'red', 'Y': 'blue', 'Z': 'purple'}
    
#     for axis in ['X', 'Y', 'Z']:
#         freqs, fft_mag = compute_fft(df_run[f'{axis}_g_ac'].values, TARGET_FS)
#         ax_fft.plot(freqs, fft_mag, color=colors[axis], label=f'{axis} Axis', alpha=0.8, lw=1)
        
#         # Auto-detect peaks above 0.2g
#         peaks, _ = find_peaks(fft_mag, height=0.2, distance=20)
#         # Sort and take top 3 peaks for this axis
#         top_peaks = sorted(peaks, key=lambda p: fft_mag[p], reverse=True)[:3]
        
#         for p in top_peaks:
#             ax_fft.plot(freqs[p], fft_mag[p], "x", color='black', markersize=8)
#             ax_fft.annotate(f"{freqs[p]:.1f} Hz\n({fft_mag[p]:.2f}g)", 
#                             xy=(freqs[p], fft_mag[p]), 
#                             xytext=(5, 5), textcoords="offset points", 
#                             fontsize=9, fontweight='bold', color='black',
#                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

#     ax_fft.set_title('Frequency Spectrum (Amplitude vs Frequency) - Top Peaks Annotated', fontsize=12, fontweight='bold')
#     ax_fft.set_xlabel('Frequency (Hz)')
#     ax_fft.set_ylabel('Amplitude (g)')
#     ax_fft.set_xlim(0, 2000)
#     ax_fft.legend(loc='upper right')
    
#     # 3. 2D SPECTROGRAM
#     dom_data = df_run[f'{dom_axis_name}_g_ac'].values
#     ax_spec2d = fig.add_subplot(gs[2:, 0])
#     f, t_spec, Sxx = spectrogram(dom_data, fs=TARGET_FS, nperseg=1024, noverlap=512)
#     Sxx_dB = 10 * np.log10(Sxx + 1e-10)
#     pcm = ax_spec2d.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='jet')
#     ax_spec2d.set_title(f'2D Spectrogram (Dominant Axis: {dom_axis_name})', fontsize=12, fontweight='bold')
#     ax_spec2d.set_ylabel('Frequency (Hz)')
#     fig.colorbar(pcm, ax=ax_spec2d, label='Power (dB)')
    
#     # 4. 3D SPECTROGRAM
#     ax_spec3d = fig.add_subplot(gs[2:, 1], projection='3d')
#     max_t_bins, max_f_bins = 70, 70
#     t_step = max(1, len(t_spec) // max_t_bins)
#     f_step = max(1, len(f) // max_f_bins)
#     T, F = np.meshgrid(t_spec[::t_step], f[::f_step])
#     surf = ax_spec3d.plot_surface(T, F, Sxx_dB[::f_step, ::t_step], cmap='jet', edgecolor='none', alpha=0.9)
#     ax_spec3d.set_title(f'3D Spectrogram (Dominant Axis: {dom_axis_name})', fontsize=12, fontweight='bold')
#     ax_spec3d.set_zlabel('Power (dB)')
#     ax_spec3d.view_init(elev=35, azim=-50)
    
#     plt.tight_layout()
#     plt.show()

# # ==========================================
# #          MAIN EXECUTION
# # ==========================================
# def main():
#     print("="*60)
#     print(" 🛠️ PUMP VIBRATION DIAGNOSTIC SYSTEM")
#     print("="*60)
    
#     df = load_and_clean_data(FILE_PATH)
#     if df is None: return
        
#     start_idx, steady_state_idx = detect_motor_states(df)
#     df_run = df.iloc[steady_state_idx:].copy()
    
#     status, reasoning, stats = auto_diagnose_status(df_run, TARGET_FS)
    
#     print("\n" + "="*60)
#     print(" 🚨 AUTOMATED DIAGNOSTIC REPORT")
#     print("="*60)
#     print(f" ► MOST LIKELY STATUS : {status}")
#     print(" ► REASONING:")
#     for r in reasoning:
#         print(f"    - {r}")
        
#     print(f"\n 📊 SIGNAL STATISTICS (Dominant Axis: {stats['Dom. Axis']}):")
#     for k, v in stats.items():
#         if k != 'Dom. Axis':
#             print(f"    {k:<15}: {v:.2f}")
#     print("="*60)
    
#     plot_full_analysis(df, df_run, start_idx, steady_state_idx, stats['Dom. Axis'])

# if __name__ == '__main__':
#     main()