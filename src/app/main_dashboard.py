import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import socket
import threading
import queue
import time
from scipy.stats import kurtosis

# ==========================================
# 1. CONFIGURATION
# ==========================================
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
# Update this path to where your model actually is
MODEL_PATH = "../../models/cnn_acc99.8_20260118-205008.pth" 
FS = 50000 / 12  # ~4167 Hz
DEVICE = torch.device("cpu")

st.set_page_config(layout="wide", page_title="AI Motor Guard")

# ==========================================
# 2. SHARED RESOURCES (CRITICAL FIX)
# ==========================================
@st.cache_resource
def get_shared_queue():
    """Creates a thread-safe queue that persists across browser refreshes."""
    return queue.Queue(maxsize=1)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MultiChannelCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(input_channels, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 5, 1, 2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.layer4 = nn.Sequential(nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.layer5 = nn.Sequential(nn.Conv1d(256, 256, 7, 1, 3), nn.BatchNorm1d(256), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv1d(256, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(32 * 64, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ==========================================
# 4. HELPER FUNCTIONS (From Your Snippet)
# ==========================================
@st.cache_resource
def load_ai_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint['class_names']
        # 6 Input channels (Vibration sensors only)
        model = MultiChannelCNN(num_classes=len(class_names), input_channels=6)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, class_names
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    return freqs, mag

def compute_statistics(signal):
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    peak = np.max(np.abs(signal))
    crest = peak / rms if rms > 0 else 0
    return rms, kurt, crest

def compute_gradcam(model, input_tensor, target_class):
    """
    Adapted from your snippet to run inside the Streamlit loop.
    """
    target_layer = model.layer6[0]
    gradients = []
    activations = []
    
    # Hooks
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks using newer PyTorch API compatibility
    h1 = target_layer.register_full_backward_hook(backward_hook)
    h2 = target_layer.register_forward_hook(forward_hook)
    
    # Zero grad and run
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()
    
    # Extract data
    grads = gradients[0].cpu().data.numpy()[0]
    fmaps = activations[0].cpu().data.numpy()[0]
    
    # Remove hooks immediately to clean up
    h1.remove()
    h2.remove()
    
    # Calculate Weights and CAM
    weights = np.mean(grads, axis=1)
    cam = np.zeros(fmaps.shape[1], dtype=np.float32)
    for i, w in enumerate(weights): 
        cam += w * fmaps[i]
        
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
        
    # Resize to match 1024 length
    return np.interp(np.linspace(0, 1024, 1024), np.linspace(0, 1024, 64), cam)

# ==========================================
# 5. BACKGROUND THREAD (LISTENER)
# ==========================================
def udp_server_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening on port {UDP_PORT}...")
    except OSError:
        return # Already running

    data_queue = get_shared_queue()

    while True:
        try:
            # Receiving 6 channels * 1024 floats
            data, _ = sock.recvfrom(65536) 
            array_data = np.frombuffer(data, dtype=np.float32).reshape(6, 1024)
            
            if data_queue.full():
                data_queue.get()
            data_queue.put(array_data)
        except Exception as e:
            print(f"UDP Error: {e}")

# Start Listener (Only once)
@st.cache_resource
def start_listener():
    t = threading.Thread(target=udp_server_thread, daemon=True)
    t.start()
    return t

start_listener()

# ==========================================
# 6. MAIN DASHBOARD UI
# ==========================================
model, class_names = load_ai_model()
data_queue = get_shared_queue()

st.title("🏭 AI Motor Health Guard")
st.markdown("Status: **Listening...**")

status_col, metric_col, stats_col = st.columns([1, 1, 1])
chart_placeholder = st.empty()

if st.toggle("Start Monitoring", value=True):
    while True:
        if not data_queue.empty():
            # 1. Get Data
            raw_data = data_queue.get()
            
            # 2. Prepare Tensor
            tensor = torch.FloatTensor(raw_data).unsqueeze(0).to(DEVICE)
            tensor.requires_grad = True # Required for GradCAM
            
            # 3. Inference
            outputs = model(tensor)
            probs = torch.softmax(outputs, 1)
            pred_idx = torch.argmax(probs, 1).item()
            pred_label = class_names[pred_idx]
            confidence = probs[0, pred_idx].item()
            
            # 4. Compute GradCAM & Stats
            saliency = compute_gradcam(model, tensor, pred_idx)
            
            # Analyze Sensor 0 (uhang_x)
            signal_x = raw_data[0]
            rms, kurt, crest = compute_statistics(signal_x)
            freqs, mags = compute_fft(signal_x, FS)

            # 5. Update UI
            # --- Status ---
            color = "#28a745" if "normal" in pred_label.lower() else "#dc3545"
            status_col.markdown(f"""
            <div style="padding: 15px; background-color: {color}; color: white; border-radius: 8px; text-align: center;">
                <h3 style='margin:0'>DIAGNOSIS</h3>
                <h1 style='margin:0'>{pred_label.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            metric_col.metric("Confidence", f"{confidence*100:.1f}%")
            
            stats_col.markdown(f"""
            **RMS:** {rms:.2f}  
            **Kurtosis:** {kurt:.2f}  
            **Crest Factor:** {crest:.2f}
            """)

            # --- Plots ---
            with chart_placeholder.container():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Plot 1: Time Domain + GradCAM Heatmap
                ax1.plot(signal_x, 'k', alpha=0.6, label='Vibration')
                # Overlay Heatmap
                im = ax1.imshow(saliency.reshape(1, -1), aspect='auto', cmap='jet', alpha=0.6, 
                               extent=[0, 1024, np.min(signal_x), np.max(signal_x)])
                ax1.set_title(f"Time Domain & AI Attention (GradCAM)")
                ax1.legend(loc="upper right")
                
                # Plot 2: Frequency Domain
                ax2.plot(freqs, mags, 'b')
                ax2.set_title("Frequency Spectrum")
                ax2.set_xlim(0, 500) # Zoom into motor frequencies
                ax2.set_xlabel("Hz")
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)

        time.sleep(0.1)