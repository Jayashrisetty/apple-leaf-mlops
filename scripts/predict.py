from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf


# ============================================================
# APPLELEAF9 MODEL PREDICTION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "appleleaf9_baseline_best.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "class_names.json"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# CHECK MODEL AND CLASS FILE
# ============================================================

print("=" * 70)
print("APPLELEAF9 IMAGE PREDICTION")
print("=" * 70)

if not MODEL_PATH.exists():
    print("\nERROR: Model file not found:")
    print(MODEL_PATH)
    sys.exit(1)

if not CLASS_NAMES_PATH.exists():
    print("\nERROR: Class names file not found:")
    print(CLASS_NAMES_PATH)
    sys.exit(1)


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print(
        r"python predict.py " 
        r'"C:\path\to\leaf_image.jpg"'
    )
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.exists():
    print("\nERROR: Image file not found:")
    print(IMAGE_PATH)
    sys.exit(1)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print("\n" + "=" * 70)
print("LOADING IMAGE")
print("=" * 70)

print("\nImage:")
print(IMAGE_PATH)

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)

image_array = tf.keras.utils.img_to_array(
    image
)

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# GENERATE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTION")
print("=" * 70)

predictions = model.predict(
    image_array,
    verbose=0
)

probabilities = predictions[0]

predicted_class_index = int(
    np.argmax(probabilities)
)

predicted_class = class_names[
    predicted_class_index
]

confidence = float(
    probabilities[predicted_class_index]
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"\nPredicted class : {predicted_class}"
)

print(
    f"Class index     : {predicted_class_index}"
)

print(
    f"Confidence      : {confidence:.2%}"
)


# ============================================================
# DISPLAY TOP 3 PREDICTIONS
# ============================================================

top_indices = np.argsort(
    probabilities
)[::-1][:3]

print("\nTop 3 predictions:")

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank}. "
        f"{class_names[index]:<25} "
        f"{probabilities[index]:.2%}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)