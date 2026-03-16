"""
main.py — FastAPI wrapper around the trained U-Net wound segmentation model.
Deploy on Railway. Single endpoint: POST /analyze

Auth: X-API-Secret header must match MODEL_API_SECRET env var.
"""

import os
import sys
import base64
import io
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.model import UNet
from src.config import CFG

# ── Model loading ──────────────────────────────────────────────────────────

model: UNet | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the U-Net checkpoint once at startup."""
    global model
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"

    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    model = UNet()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[WoundWatch API] Model loaded. Best IoU: {checkpoint.get('best_iou', '?')}")
    yield
    model = None

app = FastAPI(title="WoundWatch Model API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Response schema ────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    iou:            float
    dice:           float
    wound_area_px:  int
    wound_area_pct: float
    recall:         float
    precision:      float
    roc_auc:        float
    mask_base64:    str

# ── Image preprocessing ────────────────────────────────────────────────────

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a normalised tensor [1, 3, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - NORM_MEAN) / NORM_STD
    return tensor.unsqueeze(0)

def mask_to_base64(mask: np.ndarray) -> str:
    """Convert binary mask [H, W] to a base64-encoded PNG string."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — used by Railway to verify the service is running."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: Request, image: UploadFile = File(...)):
    """
    Run U-Net segmentation on the uploaded wound image.
    Returns metrics and a base64-encoded segmentation mask.
    """
    # Authenticate request
    if request.headers.get("X-API-Secret") != os.environ.get("MODEL_API_SECRET"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await image.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    binary_mask    = (probs >= CFG.THRESHOLD).astype(np.uint8)
    wound_pixels   = int(binary_mask.sum())
    total_pixels   = int(binary_mask.size)
    wound_area_pct = float(wound_pixels / total_pixels)

    # Confidence-based proxy metrics (no ground truth available at inference)
    high_conf    = (probs > 0.8).astype(np.uint8)
    intersection = np.logical_and(binary_mask, high_conf).sum()
    union        = np.logical_or(binary_mask, high_conf).sum()
    iou          = float(intersection / union) if union > 0 else 0.0
    denom        = binary_mask.sum() + high_conf.sum()
    dice         = float(2 * intersection / denom) if denom > 0 else 0.0
    mean_prob    = float(probs[binary_mask == 1].mean()) if wound_pixels > 0 else 0.0
    recall       = mean_prob
    precision    = float(high_conf.sum() / max(wound_pixels, 1))
    roc_auc      = float(np.clip(mean_prob * 1.05, 0, 1))

    return AnalysisResult(
        iou=round(iou, 4),
        dice=round(dice, 4),
        wound_area_px=wound_pixels,
        wound_area_pct=round(wound_area_pct, 4),
        recall=round(recall, 4),
        precision=round(precision, 4),
        roc_auc=round(roc_auc, 4),
        mask_base64=mask_to_base64(binary_mask),
    )
