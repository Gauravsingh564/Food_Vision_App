# Script/model_builder.py

import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int = 10,
                 output_shape: int = 3):
        super().__init__()

        # ── Feature extractor ────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # e.g. 64→32
            # Block 2
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 32→16
            # Block 3
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 16→8
            # Global pool → 1×1
            nn.AdaptiveAvgPool2d(1)
        )

        # ── Compute flatten size dynamically ─────────────
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, 64, 64)  # replace 64 if you trained on a different size
            feat = self.features(dummy)
            n_feats = feat.numel() // feat.shape[0]

        # ── Classifier head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feats, output_shape)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
