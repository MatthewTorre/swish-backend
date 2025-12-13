import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# -------------------------
# Model definition
# -------------------------

class ResNetTemporalAvg(nn.Module):

    import os
    import cv2
    import torch
    import numpy as np
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms

    # -------------------------
    # Model definition
    # -------------------------

    class ResNetTemporalAvg(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            base = models.resnet18(pretrained=False)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.feature_extractor(x).squeeze(-1).squeeze(-1)
            feats = feats.view(B, T, -1)
            pooled = feats.mean(dim=1)
            return self.fc(pooled)

    # -------------------------
    # Preprocessing
    # -------------------------

    FRAME_SIZE = (224, 224)
    NUM_FRAMES = 16

    transform = transforms.Compose([
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    def extract_frames_from_video(path, num_frames=NUM_FRAMES):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            raise ValueError("No frames in video")

        indices = np.linspace(0, total - 1, num_frames).astype(int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(transform(frame))

        cap.release()
        return torch.stack(frames)  # (T, C, H, W)

    # -------------------------
    # Model loader
    # -------------------------

    def load_model(weights_path, device="cpu"):
        model = ResNetTemporalAvg(num_classes=2)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # -------------------------
    # Inference
    # -------------------------

    def predict_video(model, video_path, device="cpu"):
        frames = extract_frames_from_video(video_path)
        batch = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        return {
            "is_make": bool(pred.item() == 1),
            "confidence": float(conf.item())
        }


