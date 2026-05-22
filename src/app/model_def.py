# FILE: src/utils/model_defs.py
import torch
import torch.nn as nn


class OrderNet(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int, num_classes: int):
        super().__init__()
        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=k//2, bias=False),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.SiLU(),
            )

        self.backbone = nn.Sequential(
            block(in_ch, 64, 7),
            nn.MaxPool1d(2),
            block(64, 128, 5),
            nn.MaxPool1d(2),
            block(128, 256, 3),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, feats):
        z = self.backbone(x).squeeze(-1)
        z = torch.cat([z, feats], dim=1)
        return self.head(z)
