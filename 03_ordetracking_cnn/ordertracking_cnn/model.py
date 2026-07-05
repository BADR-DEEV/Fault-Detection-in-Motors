import torch
import torch.nn as nn


class SpectralCNN(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
            nn.Dropout1d(0.2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        x_tensor = x_tensor.permute(0, 2, 1)
        x_tensor = self.features(x_tensor)
        x_tensor = self.flatten(x_tensor)
        return self.classifier(x_tensor)
