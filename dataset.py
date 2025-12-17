import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


FRAME_SIZE = (224, 224)
NUM_FRAMES = 16

_transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"No frames in video: {video_path}")

    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(_transform(frame))

    cap.release()
    return torch.stack(frames)  # (T, C, H, W)


class ShotDataset(Dataset):
    """
    Expects directory layout:
      root/
        made/*.mp4
        missed/*.mp4
    """

    def __init__(self, root_dir: str):
        self.samples: list[tuple[str, int]] = []
        for cls_name, label in [("missed", 0), ("made", 1)]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".mp4"):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise ValueError(f"No .mp4 files found under: {root_dir} (made/ missed/)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames(path)                 # (T, C, H, W)
        return frames, torch.tensor(label, dtype=torch.long)
