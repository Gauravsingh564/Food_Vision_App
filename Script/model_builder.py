import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    A TinyVGG convolutional network with two convolutional blocks
    and dynamic flatten size computation for compatibility.
    """
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 img_size: int = 64):
        super().__init__()

        # Conv Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, img_size, img_size)
            x = self.conv_block_1(dummy)
            x = self.conv_block_2(x)
            flatten_size = x.numel() // x.shape[0]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flatten_size, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
