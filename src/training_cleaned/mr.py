import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows
import seaborn as sns

# --- Constants ---
FS = 50000  # Sampling Frequency
TACH_COL = 0
VIB_COL = 2 # Using Radial (Column 2) for best spectral view

# --- User Provided RPM Function ---
def calculate_rpm_from_tach(tach_signal, sampling_freq, pulses_per_rev=60):
    """
    Physics-accurate RPM from tachometer.
    Defaults to 60 pulses/rev for MaFaulDa dataset.
    """
    # Ensure numpy array
    tach_signal = np.array(tach_signal)
    
    if len(tach_signal) < 10:
        return None

    # Threshold
    threshold = (np.max(tach_signal) + np.min(tach_signal)) / 2
    binary = tach_signal > threshold

    # Physics-based minimum spacing
    # MaFaulDa max speed approx 3800 RPM. 
    max_expected_rpm = 3800  
    # Calculate min samples between pulses based on max speed
    samples_per_pulse = (sampling_freq * 60) / (max_expected_rpm * pulses_per_rev)

    # Safety factor
    min_distance = int(samples_per_pulse * 0.5)

    rising_edges = []
    last_edge = -min_distance

    # Iterative edge detection (User logic)
    # Note: converted binary to int for boolean comparison
    binary_int = binary.astype(int)
    
    for i in range(1, len(binary_int) - 1):
        if binary_int[i - 1] == 0 and binary_int[i] == 1:
            if i - last_edge > min_distance:
                rising_edges.append(i)
                last_edge = i

    if len(rising_edges) < 2:
        return None

    # RPM calculation
    time_between = (rising_edges[-1] - rising_edges[0]) / sampling_freq
    
    # Logic: (Total Pulses - 1) / Pulses_Per_Rev = Total Revolutions
    revolutions = (len(rising_edges) - 1) / pulses_per_rev

    if time_between <= 0 or revolutions == 0:
        return None

    rpm = (revolutions / time_between) * 60
    return rpm

def plot_normalized_spectrum(file_path):
    # 1. Load Data
    # MaFaulDa CSVs usually have no header. 
    # Col 0: Tach, 1: Axial, 2: Radial, 3: Tangential
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    tach_signal = df.iloc[:, TACH_COL].values
    vib_signal = df.iloc[:, VIB_COL].values # Radial

    # 2. Calculate RPM using your function
    # MaFaulDa uses a 60-tooth gear tachometer
    rpm = calculate_rpm_from_tach(tach_signal, FS, pulses_per_rev=1)
    
    if rpm is None:
        print("Could not calculate RPM. Check signal quality.")
        rpm = 0
        f_1x = 0
    else:
        f_1x = rpm / 60.0 # Fundamental frequency in Hz

    print(f"Calculated RPM: {rpm:.2f}")
    print(f"1x Frequency: {f_1x:.2f} Hz")

    # 3. FFT Processing
    # Remove DC component
    vib_signal = vib_signal - np.mean(vib_signal)
    
    # Apply Hanning Window to smooth signal/reduce leakage
    n_samples = len(vib_signal)
    window = windows.hann(n_samples)
    vib_windowed = vib_signal * window
    
    # Compute FFT
    fft_values = rfft(vib_windowed)
    fft_freqs = rfftfreq(n_samples, 1 / FS)
    
    # Compute Amplitude and Normalize [0, 1]
    amplitude = np.abs(fft_values)
    norm_amplitude = amplitude / np.max(amplitude)

    # 4. Visualization (Matching your reference image)
    plt.style.use('seaborn-v0_8-whitegrid') # Clean white grid style
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spectrum (Red line)
    ax.plot(fft_freqs, norm_amplitude, color='#e74c3c', linewidth=1.5, label='Signal')

    # Add the 1x RPM Marker (Blue Dashed)
    if f_1x > 0:
        ax.axvline(x=f_1x, color='#2980b9', linestyle='--', linewidth=2, label='1x RPM')
        
        # Add Text Annotation
        ax.text(f_1x, 1.02, '1x RPM', color='#2980b9', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Formatting
    ax.set_title("Standard Frequency Spectrum - Normalized", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Normalized Amplitude (0-1)", fontsize=12)
    
    # Zoom in to relevant frequencies (e.g., 0 to 150Hz or 5x RPM)
    # The reference image is zoomed in, not showing the full 25kHz range
    zoom_max = max(150, f_1x * 5) 
    ax.set_xlim(0, zoom_max)
    ax.set_ylim(0, 1.1)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    plt.show()

# --- Execution ---
# Replace this string with the actual path to your .csv file
# Example path structure common in MaFaulDa:
file_path = r"C:\dev_work\personal\Ai-Driven-Vibrations-motor\data\raw_mafulda\imbalance\15g\61.44.csv" 

# Uncomment the line below to run with a real file
# plot_normalized_spectrum(file_path)

# # --- Dummy Data Generation (For demonstration if you don't have the file ready) ---
# def create_dummy_mafaulda_data():
#     """Generates a dummy signal mimicking MaFaulDa structure for testing."""
#     duration = 1.0
#     t = np.linspace(0, duration, int(FS * duration))
    
#     # Simulate 30 Hz rotation (1800 RPM)
#     true_freq = 30.0 
    
#     # 1. Create Tach Signal (60 pulses per rev)
#     # Square wave at 30 * 60 = 1800 Hz
#     tach = 5 * (0.5 * (1 + np.sign(np.sin(2 * np.pi * true_freq * 60 * t))))
    
#     # 2. Create Vibration Signal (Strong 1x, some noise)
#     vib = 0.5 * np.sin(2 * np.pi * true_freq * t) + \
#           0.1 * np.sin(2 * np.pi * 2 * true_freq * t) + \
#           0.05 * np.random.randn(len(t))
    
#     # Create DataFrame
#     df = pd.DataFrame({0: tach, 1: vib*0.1, 2: vib, 3: vib*0.1})
#     df.to_csv("dummy_mafaulda.csv", header=False, index=False)
#     return "dummy_mafaulda.csv"

# Run with dummy data
# dummy_file = create_dummy_mafaulda_data()
plot_normalized_spectrum(file_path)