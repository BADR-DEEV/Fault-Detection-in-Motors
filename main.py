# Install dependencies if not installed
# !pip install scipy numpy matplotlib PyEMD

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PyEMD import EMD
import os
from scipy.fft import fft, fftfreq

from load_dataset import load_file

# -------------------------------
# 1️⃣ Load a MATLAB dataset
# -------------------------------

# Example: change this path to your actual MATLAB files


data = load_file("F1.mat", "Faulty")

# print ("shape of H" + data['H'].shape)
# print ("zero value of H" + data['H'][0])


# # data = loadmat(os.path.join(path, file))

# Assume each MATLAB file contains x(t), y(t), z(t)
# Example keys: data['x'], data['y'], data['z']
H = np.array(data['H'])        # Convert to numpy array
print("Shape of H:", H.shape)  # Should be (N, 3)

x = H[:, 0]  # X-axis signal (column 1)
y = H[:, 1]  # Y-axis signal (column 2)
z = H[:, 2]  # Z-axis signal (column 3)


# -------------------------------
# 2️⃣ Combine the three channels
# -------------------------------
S = np.sqrt(x**2 + y**2 + z**2)

# Optional normalization (helps EMD stability)
S = (S - np.mean(S)) / np.std(S)

# -------------------------------
# 3️⃣ Empirical Mode Decomposition
# -------------------------------
emd = EMD()
imfs = emd.emd(S)

num_imfs = imfs.shape[0]
print(f"Extracted {num_imfs} IMFs")

# -------------------------------
# 4️⃣ Visualize IMFs
# -------------------------------
fig, axs = plt.subplots(num_imfs + 1, 1, figsize=(10, 10))
axs[0].plot(S)
axs[0].set_title("Original Vibration Signal S(t)")

for i in range(num_imfs):
    axs[i + 1].plot(imfs[i])
    axs[i + 1].set_title(f"IMF {i + 1}")

plt.tight_layout()
plt.show()

# -------------------------------
# 5️⃣ Reconstruct the preprocessed signal
# -------------------------------
# Discard IMF1 (noisy) and sum IMF2–IMF10 + residue
reconstructed = np.sum(imfs[1:10, :], axis=0) if num_imfs >= 10 else np.sum(imfs[1:, :], axis=0)

plt.figure(figsize=(10,4))
plt.plot(S, label='Original')
plt.plot(reconstructed, label='Preprocessed (IMF2–IMF10)', alpha=0.8)
plt.legend()
plt.title("Signal Reconstruction After EMD Preprocessing")
plt.show()



fs = 1000  # Sampling frequency (Hz) – change to your actual one
N = len(S)
T = 1 / fs
yf = fft(S)
xf = fftfreq(N, T)[:N//2]

plt.figure(figsize=(10,4))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title("Frequency Spectrum of Faulty Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()