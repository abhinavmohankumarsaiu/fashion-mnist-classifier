from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fashion_cnn_model.keras"
MAX_FILE_BYTES = 10 * 1024 * 1024

# This is the official Fashion-MNIST integer-label order. Do not alphabetize it.
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

model: tf.keras.Model | None = None


def load_classifier() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file is missing: {MODEL_PATH.name}. Run train_model.py first."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = load_classifier()
    # Warm the graph so the first real request is not unusually slow.
    model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
    yield


app = FastAPI(
    title="Fashion-MNIST CNN Classifier",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,https://fashion-ai-mnist.abhinav261065.chatgpt.site",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/css", StaticFiles(directory=str(BASE_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(BASE_DIR / "js")), name="js")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a normal product image to Fashion-MNIST-like model input."""
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")

    grayscale = ImageOps.grayscale(image)
    grayscale.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    if pixels.size == 0:
        raise ValueError("The uploaded image has no pixels.")

    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    # Preserve canonical dataset samples exactly; they already match training.
    if pixels.shape == (28, 28) and float(np.median(border)) <= 127:
        return (pixels.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)

    if float(np.median(border)) > 127:
        pixels = 255 - pixels

    # Remove residual background while keeping soft clothing edges.
    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    background_level = float(np.median(border))
    foreground = np.clip(pixels.astype(np.float32) - background_level, 0, 255)
    peak = float(foreground.max())
    if peak < 8:
        raise ValueError("No clothing item could be detected in the image.")

    mask = foreground > max(8.0, peak * 0.08)
    rows, cols = np.where(mask)
    if not rows.size or not cols.size:
        raise ValueError("No clothing item could be detected in the image.")

    cropped = foreground[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    crop_image = Image.fromarray(np.uint8(cropped))
    width, height = crop_image.size
    scale = min(22 / width, 22 / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    crop_image = crop_image.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - resized_size[0]) // 2, (28 - resized_size[1]) // 2)
    canvas.paste(crop_image, offset)

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array.reshape(1, 28, 28, 1)


def probability_vector(raw_output: np.ndarray) -> np.ndarray:
    output = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if output.shape != (len(CLASS_NAMES),):
        raise RuntimeError(f"Expected 10 model outputs, received {output.shape}.")
    return tf.nn.softmax(output).numpy()


@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "starting",
        "ready": model is not None,
        "model": "Fashion-MNIST CNN",
        "classes": list(CLASS_NAMES),
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="The model is still starting. Try again.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    contents = await file.read(MAX_FILE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image_array = preprocess_image(image)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logits = model.predict(image_array, verbose=0)[0]
    probabilities = probability_vector(logits)
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        {
            "label": CLASS_NAMES[int(index)],
            "confidence": round(float(probabilities[index]), 6),
        }
        for index in top_indices
    ]
    legacy_predictions = [
        {
            "class": item["label"],
            "confidence": round(item["confidence"] * 100, 2),
        }
        for item in top_predictions
    ]

    return {
        "success": True,
        "filename": file.filename,
        "model": "Fashion-MNIST CNN",
        "prediction": top_predictions[0],
        "alternatives": top_predictions[1:],
        "predictions": legacy_predictions,
    }

