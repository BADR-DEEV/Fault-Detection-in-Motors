# model.py

import torch
import torch.nn as nn


class PdMCNN(nn.Module):
    """
    SKF / Siemens–style feature embedding network
    """
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.embedding = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.features(x).squeeze(-1)
        emb = self.embedding(x)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits
