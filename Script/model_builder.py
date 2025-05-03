import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    A TinyVGG convolutional network with three convolutional blocks,
    BatchNorm, dropout, and adaptive pooling for robust feature extraction.
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int = 10,
                 output_shape: int = 3,
                 dropout_p: float = 0.4):
        super().__init__()

        # ── Convolutional feature extractor ───────────────────────────
        self.features = nn.Sequential(
            # Block 1: channels -> hidden_units
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # spatial dims /2
            nn.Dropout2d(dropout_p),

            # Block 2: hidden_units -> hidden_units*2
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Dropout2d(dropout_p),

            # Block 3: hidden_units*2 -> hidden_units*4
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Dropout2d(dropout_p),

            # Global average pooling to 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ── Fully connected classifier head ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten (batch, hidden_units*4, 1,1) -> (batch, hidden_units*4)
            nn.Linear(hidden_units * 4, hidden_units * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_units * 2, output_shape)
        )

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Example usage:
# model = TinyVGG(input_shape=3, hidden_units=10, output_shape=3, dropout_p=0.4)
# Train and save: torch.save(model.state_dict(), 'models/05_going_modular_script_mode_tinyvgg_model.pth')
