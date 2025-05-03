# Script/model_builder.py

import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 img_size: int = 224):
        super().__init__()

        # ── Feature extractor ─────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112

            # Block 2
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56
        )

        # ── Dynamically compute flatten size ───────────
        # run a dummy input through features to get shape
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, img_size, img_size)
            feat_out = self.features(dummy)
            num_feats = feat_out.numel() // feat_out.shape[0]  # batch dim

        # ── Classifier head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_feats, output_shape)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
