from fastapi import FastAPI
from model import predict_video

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict():
    result = predict_video()
    return result
