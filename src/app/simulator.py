import socket
import torch
import numpy as np
import time
import random

# CONFIG
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CACHE_PATH = "../../data/vis_cache_2.pt" # <--- POINT TO YOUR CACHE FILE

print("Loading dataset...")
cache = torch.load(CACHE_PATH)
X_data = cache['X'] # (N, 7, 1024) - Channel 0 is Tachometer
y_data = cache['y']
class_names = cache['class_names']

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Sending data to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        # Pick random sample
        idx = random.randint(0, len(X_data) - 1)
        
        # --- DATA PREP ---
        # Your cache has 7 channels (Tach + 6 Vib).
        # Your model was trained on 6 channels.
        # We must slice off the first channel [1:]
        raw_tensor = X_data[idx] # (7, 1024)
        input_data = raw_tensor[1:].numpy() # (6, 1024)
        
        true_label = class_names[y_data[idx]]
        
        # Serialize to bytes
        message = input_data.astype(np.float32).tobytes()
        
        # Send
        sock.sendto(message, (UDP_IP, UDP_PORT))
        print(f"Sent Sample #{idx} | True Label: {true_label}")
        
        time.sleep(1.0) # Send one per second

except KeyboardInterrupt:
    print("Stopped.")