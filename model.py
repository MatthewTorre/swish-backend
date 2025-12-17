import torch
import torch.nn as nn
from torchvision import models


class ResNetTemporalAvg(nn.Module):
    """
    Expects input of shape (B, T, C, H, W)
    Outputs logits of shape (B, num_classes)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # (B*T, 512)
        feats = feats.view(B, T, -1)                               # (B, T, 512)
        pooled = feats.mean(dim=1)                                 # (B, 512)
        return self.fc(pooled)


def load_model(weights_path: str | None = None, device: str = "cpu", num_classes: int = 2):
    model = ResNetTemporalAvg(num_classes=num_classes).to(device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)

