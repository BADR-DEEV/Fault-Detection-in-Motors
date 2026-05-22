import serial
import json
import numpy as np
import requests
import time

# --- CONFIGURATION ---
ESP32_PORT = 'COM10' 
BAUD_RATE = 115200
FASTAPI_URL = "http://127.0.0.1:8000/predict"

TARGET_SAMPLES = 2000 
EXPECTED_FS = 1000.0 

def process_and_send(json_payload):
    try:
        data = json.loads(json_payload)
        
        # Back to the standard Acceleration buffers that we know work!
        raw_x = data["buffer"]["accX"]["buffer"]
        raw_y = data["buffer"]["accY"]["buffer"]
        raw_z = data["buffer"]["accZ"]["buffer"]
        
        min_len = min(len(raw_x), len(raw_y), len(raw_z))
        if min_len < 100:
            print("⚠️ Error: Received empty or invalid snapshot from Phyphox.")
            return

        print(f"\n✅ Captured {min_len} samples over 2 seconds (~{min_len/2:.1f} Hz).")
        
        # --- THE MAGIC TRICK: REMOVING GRAVITY ---
        # Subtracting the mean instantly deletes the static 9.81m/s^2 gravity pull 
        # from whatever angle the phone is resting at.
        x_clean = np.array(raw_x[:min_len]) - np.mean(raw_x[:min_len])
        y_clean = np.array(raw_y[:min_len]) - np.mean(raw_y[:min_len])
        z_clean = np.array(raw_z[:min_len]) - np.mean(raw_z[:min_len])

        print(f"🔄 Resampling to {TARGET_SAMPLES} samples to match AI Model (1000 Hz)...")
        
        # Stretch to exactly 2000 samples to simulate 1000 Hz
        old_indices = np.linspace(0, 1, min_len)
        new_indices = np.linspace(0, 1, TARGET_SAMPLES)
        
        x_stretched = np.interp(new_indices, old_indices, x_clean).tolist()
        y_stretched = np.interp(new_indices, old_indices, y_clean).tolist()
        z_stretched = np.interp(new_indices, old_indices, z_clean).tolist()

        api_payload = {
            "ax": x_stretched,
            "ay": y_stretched,
            "az": z_stretched,
            "fs": EXPECTED_FS 
        }

        print("🚀 Sending to AI Server...")
        response = requests.post(FASTAPI_URL, json=api_payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print(f"🔥 PREDICTION: {result['prediction']}")
            print(f"📊 CONFIDENCE: {result['confidence']:.2f}")
            print(f"⚙️ SEVERITY:   {result['severity_estimate']}")
            print("="*50 + "\n")
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error processing payload: {e}")

def main():
    print(f"Starting Continuous Bridge on {ESP32_PORT}...")
    
    while True:
        try:
            ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
            print(f"✅ Connected to ESP32 on {ESP32_PORT}.")
            
            is_recording = False
            json_payload = ""

            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                        
                    if line == "---SNAPSHOT_START---":
                        is_recording = True
                        json_payload = ""
                        print("\n[!] Trigger Crossed! Downloading snapshot from ESP32...")
                        continue
                        
                    if line == "---SNAPSHOT_END---":
                        is_recording = False
                        process_and_send(json_payload)
                        continue

                    if is_recording:
                        json_payload += line
                    else:
                        print(f"[ESP32] {line}")

        except serial.SerialException:
            print(f"⏳ Waiting for ESP32 on {ESP32_PORT}...")
            time.sleep(3)

if __name__ == '__main__':
    main()