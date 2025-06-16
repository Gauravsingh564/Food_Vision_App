import torch
from torch import nn
import torchvision.models as models
def create_transfer_model(num_classes: int,
                          pretrained: bool = True,
                          freeze_base: bool = True,
                          dropout: float = 0.2) -> nn.Module:
    """
    Builds a transfer-learning model using EfficientNet-B0 as the backbone.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.
        freeze_base: If True, freeze the backbone feature layers.
        dropout: Dropout rate for the classifier head.

    Returns:
        model: An nn.Module ready for training or inference.
    """
    # Load EfficientNet-B0 with pretrained weights if desired
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze the base feature extractor if requested
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features  # typically 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )

    return model
