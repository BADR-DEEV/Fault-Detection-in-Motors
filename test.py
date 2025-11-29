import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Example: path to one .mat file
file_path = "Dataset/Healthy/H1.mat"   # change to your dataset folder

# Load the .mat file
mat = scipy.io.loadmat(file_path)

# Check what keys are inside
print(mat.keys())

signal = mat['H'].squeeze()   # flatten if needed
print("Signal shape:", signal.shape)
print("First 10 samples:", signal[:10])


# Load a healthy signal
healthy_file = "Dataset/Healthy/H1.mat"
healthy = scipy.io.loadmat(healthy_file)['H'].squeeze()

# Load a faulty signal
faulty_file = "Dataset/Faulty/F1.mat"
faulty = scipy.io.loadmat(faulty_file)['H'].squeeze()

# Plot them

plt.figure(figsize=(12,4))
plt.plot(signal[:,0], label='X-axis')
plt.plot(signal[:,1], label='Y-axis')
plt.plot(signal[:,2], label='Z-axis')
plt.title("Healthy Motor Vibration Signal (3 Axes)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

plt.subplot(2,1,2)
plt.plot(faulty[:,0], label='X-axis')
plt.plot(faulty[:,1], label='Y-axis')
plt.plot(faulty[:,2], label='Z-axis')
plt.title("Faulty Motor Vibration Signal")

plt.tight_layout()
plt.show()


def plot_fft(signal, fs, title):
    N = len(signal)
    Y = np.abs(np.fft.fft(signal))[:N//2]
    f = np.fft.fftfreq(N, 1/fs)[:N//2]
    plt.plot(f, Y)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

fs = 1000  # dataset sampling rate
plot_fft(healthy, fs, "FFT of Healthy Signal")
plot_fft(faulty, fs, "FFT of Faulty Signal")
