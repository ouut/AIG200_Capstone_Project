"""
FastAPI Application — FashionMNIST Classifier
=============================================
Serves a pre-trained FashionMNIST model via REST API.

Endpoints:
  GET  /              — API information
  GET  /health        — Health check
  POST /predict       — Predict fashion item from image

Authentication: X-API-Key header required for /predict
"""

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import joblib
import logging
from contextlib import asynccontextmanager

# ─────────────────────────────────────
# Configuration
# ─────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
API_KEY = os.environ.get("API_KEY", "fashion-mnist-api-key-2026")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
STATE_DICT_PATH = os.path.join(MODEL_DIR, "best_model_state.pt")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────
# Model Architecture (must match training)
# ─────────────────────────────────────
import torch.nn as nn


class CNN(nn.Module):
    """CNN for FashionMNIST — must match training architecture exactly."""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


class DNN(nn.Module):
    """DNN for FashionMNIST."""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.3):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(self.flatten(x))


# ─────────────────────────────────────
# Global state
# ─────────────────────────────────────
model: Optional[nn.Module] = None
preprocessor: Optional[object] = None
class_names: List[str] = []
model_metadata: dict = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained model and preprocessing pipeline."""
    global model, preprocessor, class_names, model_metadata

    logger.info(f"Loading model from {MODEL_DIR}...")
    logger.info(f"Using device: {device}")

    # Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            model_metadata = json.load(f)
        class_names = model_metadata.get("class_names", [])
        logger.info(f"Model type: {model_metadata.get('model_type', 'unknown')}")
        logger.info(f"Accuracy: {model_metadata.get('accuracy', 0):.2f}%")
    else:
        # Fallback class names
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        logger.warning("metadata.json not found, using default class names")

    # Load model — try TorchScript first, fall back to state dict
    model_type = model_metadata.get("model_type", "CNN")

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading TorchScript model: {MODEL_PATH}")
        model = torch.jit.load(MODEL_PATH, map_location=device)
        model.eval()
    elif os.path.exists(STATE_DICT_PATH):
        logger.info(f"Loading state dict: {STATE_DICT_PATH}")
        checkpoint = torch.load(STATE_DICT_PATH, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", model_type)
        class_names = checkpoint.get("class_names", class_names)

        if model_type == "CNN":
            model = CNN(num_classes=len(class_names))
        else:
            model = DNN(num_classes=len(class_names))

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Loaded {model_type} from state dict (acc: {checkpoint.get('accuracy', 0):.2f}%)")
    else:
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH} or {STATE_DICT_PATH}. "
            f"Run train.py first."
        )

    # Load preprocessor
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Preprocessor loaded")
    else:
        logger.warning("Preprocessor not found — using default normalization")

    logger.info("✅ Model loading complete!")
    return model


# ─────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    load_model()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="FashionMNIST Classifier API",
    description="Classify fashion items into 10 categories using a CNN model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow testing from browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────
# Authentication
# ─────────────────────────────────────
async def verify_api_key(request: Request):
    """Dependency: verify X-API-Key header."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ─────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────
class PredictionRequest(BaseModel):
    """Input: image as a flat list of 784 pixels (0-255 grayscale)."""
    image: List[float] = Field(
        ...,
        description="Image as a flat list of 784 pixel values (0-255, 28x28 grayscale)",
        min_length=784,
        max_length=784,
    )


class PredictionResponse(BaseModel):
    """Output: predicted class with confidence and all probabilities."""
    predicted_class: int = Field(..., description="Index of predicted class (0-9)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Confidence score for the prediction")
    all_probabilities: List[float] = Field(..., description="Probabilities for all 10 classes")
    model_type: str = Field(..., description="Type of model used for prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    device: str


# ─────────────────────────────────────
# Endpoints
# ─────────────────────────────────────
@app.get("/", response_model=dict)
async def root():
    """API information."""
    return {
        "name": "FashionMNIST Classifier API",
        "version": "1.0.0",
        "description": "Classify fashion items into 10 categories",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Make a prediction (requires X-API-Key)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_type=model_metadata.get("model_type", "unknown"),
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    """
    Predict the fashion item category from a 28x28 grayscale image.

    **Input**: JSON with `image` field — a list of 784 float values (0-255).
    **Auth**: Requires `X-API-Key` header.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to numpy array and reshape
        pixel_array = np.array(request.image, dtype=np.float32).reshape(28, 28)

        # Preprocess
        if preprocessor is not None:
            # Use the fitted preprocessor
            input_tensor = preprocessor.preprocess_single(pixel_array)
        else:
            # Manual preprocessing
            if pixel_array.max() > 1.0:
                pixel_array = pixel_array / 255.0
            tensor = torch.tensor(pixel_array, dtype=torch.float32)
            tensor = (tensor - 0.5) / 0.5
            input_tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)

        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            all_probs = probabilities[0].cpu().tolist()

        logger.info(f"Prediction: class={predicted_class} ({class_names[predicted_class]}), "
                    f"confidence={confidence:.4f}")

        return PredictionResponse(
            predicted_class=predicted_class,
            class_name=class_names[predicted_class] if class_names else f"Class {predicted_class}",
            confidence=round(confidence, 4),
            all_probabilities=[round(p, 4) for p in all_probs],
            model_type=model_metadata.get("model_type", "unknown"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─────────────────────────────────────
# Entry point
# ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
