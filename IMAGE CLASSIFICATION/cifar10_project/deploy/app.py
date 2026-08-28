import io
import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_checkpoint  # noqa: E402
torch.backends.cudnn.enabled = False

MODEL_NAME = os.environ.get("MODEL_NAME", "baseline")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
DEVICE_PREF = os.environ.get("DEVICE", "cuda")

app = FastAPI(
    title="CIFAR-10 Classifier API",
    description="Serves a from-scratch CNN trained on CIFAR-10.",
    version="1.0.0",
)

_model = None
_device = None


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    checkpoint: str


@app.on_event("startup")
def load_model() -> None:
    global _model, _device
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"CHECKPOINT_PATH is not set or doesn't exist: {CHECKPOINT_PATH!r}. "
            f"Set the CHECKPOINT_PATH environment variable to a trained .pt file "
            f"before starting the server."
        )
    _device = get_device(DEVICE_PREF)
    _model = build_model(MODEL_NAME, num_classes=10, dropout=0.0).to(_device)
    load_checkpoint(CHECKPOINT_PATH, _model, map_location=str(_device))
    _model.eval()
    print(f"Loaded {MODEL_NAME} from {CHECKPOINT_PATH} on {_device}")


def preprocess(image_bytes: bytes) -> torch.Tensor:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std = np.array(CIFAR10_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(arr.astype(np.float32))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "model not loaded",
        model=MODEL_NAME,
        device=str(_device),
        checkpoint=CHECKPOINT_PATH or "",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    x = preprocess(image_bytes).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    return PredictionResponse(
        predicted_class=CIFAR10_CLASSES[top_idx],
        confidence=float(probs[top_idx].item()),
        probabilities={CIFAR10_CLASSES[i]: float(probs[i].item()) for i in range(len(CIFAR10_CLASSES))},
    )


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 classifier API. POST an image to /predict, or see /docs.",
        "classes": list(CIFAR10_CLASSES),
    }
