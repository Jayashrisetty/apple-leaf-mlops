from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# APPLELEAF9 MODEL EVALUATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops"
)

TEST_DIR = BASE_DIR / "data" / "splits" / "test"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "appleleaf9_baseline_best.keras"
)

CLASS_FILE = (
    BASE_DIR
    / "models"
    / "class_names.json"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "model"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("APPLELEAF9 MODEL EVALUATION")
print("=" * 70)

print("\nTest dataset:")
print(TEST_DIR)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not CLASS_FILE.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_FILE}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

print(
    f"\nTest images: "
    f"{len(test_ds.file_paths)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n")
print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(
    f"\nTest Accuracy: {accuracy:.4%}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_file = (
    REPORT_DIR
    / "classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "APPLELEAF9 CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test Accuracy: {accuracy:.6%}\n\n"
    )

    f.write(report)

    f.write("\n\nCONFUSION MATRIX\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        np.array2string(cm)
    )

print("\nClassification report saved:")
print(report_file)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(cm)

plt.title(
    "AppleLeaf9 Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

# Add values inside cells
for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_file = (
    REPORT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved:")
print(cm_file)


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

summary_file = (
    REPORT_DIR
    / "per_class_performance.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "APPLELEAF9 PER-CLASS PERFORMANCE\n"
    )

    f.write("=" * 70 + "\n\n")

    for class_name in class_names:

        metrics = report_dict[class_name]

        f.write(
            f"{class_name}\n"
        )

        f.write(
            f"Precision : "
            f"{metrics['precision']:.4f}\n"
        )

        f.write(
            f"Recall    : "
            f"{metrics['recall']:.4f}\n"
        )

        f.write(
            f"F1-score  : "
            f"{metrics['f1-score']:.4f}\n"
        )

        f.write(
            f"Support   : "
            f"{int(metrics['support'])}\n\n"
        )


print("\nPer-class performance saved:")
print(summary_file)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nFinal test accuracy: "
    f"{accuracy:.4%}"
)

print("\nGenerated files:")

print(
    f"1. {report_file}"
)

print(
    f"2. {cm_file}"
)

print(
    f"3. {summary_file}"
)

print("\n" + "=" * 70)