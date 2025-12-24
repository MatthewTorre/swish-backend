# model.py
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms

# -------------------------
# Model definition
# -------------------------

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
        feats = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)
        pooled = feats.mean(dim=1)
        return self.fc(pooled)

# -------------------------
# Preprocessing
# -------------------------

NUM_FRAMES = 16
FRAME_SIZE = (224, 224)

_transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    """
    Returns tensor of shape (T, C, H, W)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"No frames in video: {video_path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(_transform(frame))

    cap.release()
    return torch.stack(frames)  # (T, C, H, W)

# -------------------------
# Model loading
# -------------------------

def load_model(
    weights_path: str,
    device: str = "cpu",
    num_classes: int = 2
) -> nn.Module:
    model = ResNetTemporalAvg(num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def predict_video(
    model: nn.Module,
    video_path: str,
    device: str = "cpu"
) -> dict:
    frames = extract_frames(video_path)              # (T, C, H, W)
    batch = frames.unsqueeze(0).to(device)           # (1, T, C, H, W)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    conf, pred = torch.max(probs, dim=1)

    return {
        "prediction": "MAKE" if pred.item() == 1 else "MISS",
        "confidence": float(conf.item())
    }
