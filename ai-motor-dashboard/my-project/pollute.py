import json
import numpy as np
import scipy.signal

# 1. Load the clean golden samples you made earlier
with open("dashboard_data.json", "r") as f:
    data = json.load(f)

corrupted_data = {}

def add_noise(signal, level=0.5):
    """Adds white gaussian noise"""
    noise = np.random.normal(0, level, len(signal))
    return signal + noise

def scale_volume(signal, factor=0.5):
    """Simulates the sensor being further away (quieter signal)"""
    return signal * factor

def drift_speed(signal, speed_factor=1.05):
    """Resamples the signal to simulate a 5% speed difference"""
    # If motor runs 5% faster, the signal gets compressed
    new_len = int(len(signal) / speed_factor)
    return scipy.signal.resample(signal, new_len)[:1024] # Cut to fit

print("Generating Adversarial Test Data...")

for fault_type, signals in data.items():
    ax = np.array(signals["ax"])
    ay = np.array(signals["ay"])
    az = np.array(signals["az"])

    # --- TEST 1: NOISY ENVIRONMENT ---
    # Simulates a factory floor with other machines running nearby
    corrupted_data[f"{fault_type}_noisy"] = {
        "ax": add_noise(ax, level=1.0).tolist(), # Heavy noise
        "ay": add_noise(ay, level=1.0).tolist(),
        "az": add_noise(az, level=1.0).tolist()
    }

    # --- TEST 2: WEAK SENSOR (Volume Drop) ---
    # Simulates a sensor that is loose or far from the bearing
    corrupted_data[f"{fault_type}_weak"] = {
        "ax": scale_volume(ax, 0.4).tolist(), # 60% quieter
        "ay": scale_volume(ay, 0.4).tolist(),
        "az": scale_volume(az, 0.4).tolist()
    }

    # --- TEST 3: RPM DRIFT (Speed Change) ---
    # Simulates the motor running 5% faster than the training data
    # This shifts the frequency peaks in the FFT
    # IF YOUR MODEL PASSES THIS, IT IS EXCELLENT.
    corrupted_data[f"{fault_type}_drift"] = {
        "ax": drift_speed(ax, 1.05).tolist(),
        "ay": drift_speed(ay, 1.05).tolist(),
        "az": drift_speed(az, 1.05).tolist()
    }

# Save to new JSON
with open("stress_test_data.json", "w") as f:
    json.dump(corrupted_data, f)

print("Saved 'stress_test_data.json'. Import this into React!")