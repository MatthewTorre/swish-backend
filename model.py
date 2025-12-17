# model.py

import torch
import torch.nn as nn
from torchvision import models


class ResNetTemporalAvg(nn.Module):
    """
    Temporal ResNet model.
    Expects input of shape:
        (B, T, C, H, W)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Base CNN backbone
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )  # removes final FC layer

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Merge batch and time
        x = x.view(B * T, C, H, W)

        feats = self.feature_extractor(x)     # (B*T, 512, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1) # (B*T, 512)

        feats = feats.view(B, T, -1)           # (B, T, 512)
        pooled = feats.mean(dim=1)              # temporal average

        return self.fc(pooled)


def load_model(
    weights_path: str | None = None,
    device: str = "cpu",
    num_classes: int = 2,
):
    """
    Loads the model and optional weights.
    """
    model = ResNetTemporalAvg(num_classes=num_classes).to(device)

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
):
    """
    Runs inference.

    x: (B, T, C, H, W)
    returns: logits (B, num_classes)
    """
    return model(x)
