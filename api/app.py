from pathlib import Path
import json
import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


# ============================================================
# APPLELEAF9 MLOps FASTAPI PREDICTION SERVICE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "appleleaf9_improved_best.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "improved_class_names.json"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# ============================================================
# LOAD MODEL
# ============================================================

print("============================================================")
print("Loading AppleLeaf9 Improved Model...")
print(f"Model path: {MODEL_PATH}")
print("============================================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Number of classes: {len(CLASS_NAMES)}")


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AppleLeaf9 Disease Detection API",
    description="Apple leaf disease classification using improved CNN",
    version="2.0.0"
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AppleLeaf9 Disease Detection API",
        "status": "running",
        "model": "appleleaf9_improved_best",
        "model_type": "Improved CNN",
        "classes": len(CLASS_NAMES)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # --------------------------------------------------------
    # Top 3 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        top_predictions.append(
            {
                "class": CLASS_NAMES[int(index)],
                "class_index": int(index),
                "confidence": round(
                    float(predictions[index]),
                    4
                )
            }
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": round(
            confidence,
            4
        ),
        "top_3_predictions": top_predictions
    }