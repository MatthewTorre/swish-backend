import os
import torch
from fastapi import FastAPI, UploadFile, File

from model import load_model
from dataset import extract_frames

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "shot_model.pth")

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(WEIGHTS_PATH, device=DEVICE, num_classes=2)
    return _model


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save upload to temp file (simple + reliable)
    temp_path = "temp_upload.mp4"
    data = await file.read()
    with open(temp_path, "wb") as f:
        f.write(data)

    frames = extract_frames(temp_path).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
    model = get_model()

    with torch.no_grad():
        logits = model(frames)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "is_make": bool(pred.item() == 1),
        "confidence": float(conf.item()),
    }
