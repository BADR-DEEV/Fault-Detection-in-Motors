# infer.py

import torch
import numpy as np
from training.preprocessing import preprocess
from training.model import PdMCNN
from training.ood import MahalanobisOOD
from training.config import CLASSES, OOD_THRESHOLD


device = "cuda" if torch.cuda.is_available() else "cpu"

model = PdMCNN(len(CLASSES)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

ood = MahalanobisOOD()
ood.mean = np.load("ood_mean.npy")
ood.cov_inv = np.load("ood_covinv.npy")


def infer(signal, fs):
    x, rpm = preprocess(signal, fs)
    x = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, emb = model(x, return_embedding=True)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    dist = ood.distance(emb.cpu().numpy()[0])
    unknown_prob = 1 - ood.probability(dist, OOD_THRESHOLD)

    ranked = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)

    return {
        "Primary": ranked[0],
        "Secondary": ranked[1],
        "UnknownProbability": float(unknown_prob),
        "RPM": float(rpm)
    }
