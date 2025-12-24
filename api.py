import os
import uuid
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException

from model import load_model
from dataset import extract_frames

# --------------------------------------------------
# App setup
# --------------------------------------------------

app = FastAPI(title="Swish Shot Classification API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to model weights (env var for flexibility)
WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH",
    "shot_model.pth"   # default for local dev
)

# Lazy-loaded global model
_model = None


def get_model():
    """
    Load the model once and reuse it for all requests.
    """
    global _model
    if _model is None:
        _model = load_model(
            weights_path=WEIGHTS_PATH,
            device=DEVICE,
            num_classes=2
        )
    return _model


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "model_loaded": _model is not None
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a video file (.mp4, .mov),
    runs shot classification,
    returns prediction + confidence.
    """

    # Basic validation
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    # Create temp file
    temp_filename = f"/tmp/{uuid.uuid4()}.mp4"

    try:
        # Save uploaded video
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract frames
        frames = extract_frames(temp_filename)  # (T, C, H, W)
        frames = frames.unsqueeze(0).to(DEVICE) # (1, T, C, H, W)

        # Run inference
        model = get_model()
        with torch.no_grad():
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        return {
            "is_make": bool(pred.item() == 1),
            "confidence": float(conf.item())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
