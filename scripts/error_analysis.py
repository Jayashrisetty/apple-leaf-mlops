from pathlib import Path
import json
import shutil
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# APPLELEAF9 MODEL ERROR ANALYSIS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops"
)

TEST_DIR = PROJECT_DIR / "data" / "splits" / "test"

MODEL_PATH = PROJECT_DIR / "models" / "appleleaf9_baseline_best.keras"

CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "model"
    / "error_analysis"
)

REPORT_FILE = REPORT_DIR / "error_analysis_report.txt"

MISCLASSIFIED_FILE = (
    REPORT_DIR / "misclassified_images.txt"
)

# NEW: directory for visual inspection
ERROR_CASES_DIR = (
    REPORT_DIR / "error_cases"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("APPLELEAF9 MODEL ERROR ANALYSIS")
print("=" * 70)

print("\nTest dataset:")
print(TEST_DIR)

print("\nModel:")
print(MODEL_PATH)

if not TEST_DIR.exists():
    print("\nERROR: Test dataset does not exist.")
    raise SystemExit

if not MODEL_PATH.exists():
    print("\nERROR: Model file does not exist.")
    raise SystemExit

if not CLASS_NAMES_PATH.exists():
    print("\nERROR: class_names.json does not exist.")
    raise SystemExit


# ============================================================
# CREATE REPORT DIRECTORIES
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Create error-case directory
ERROR_CASES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# IMPORTANT:
# Do not calculate image count using cardinality * batch size.
# The final batch may contain fewer than BATCH_SIZE images.

test_image_count = len(test_dataset.file_paths)

print(f"\nTest images: {test_image_count}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

all_true_labels = []
all_predictions = []
all_probabilities = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    all_true_labels.extend(
        labels.numpy()
    )

    all_predictions.extend(
        predicted_classes
    )

    all_probabilities.extend(
        predictions
    )


y_true = np.array(all_true_labels)

y_pred = np.array(all_predictions)

probabilities = np.array(all_probabilities)


print(
    f"\nPredictions generated: "
    f"{len(y_pred)}"
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

misclassified_count = np.sum(
    y_true != y_pred
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"\nTest Accuracy       : {accuracy:.4%}"
)

print(
    f"Correct predictions : "
    f"{len(y_true) - misclassified_count}"
)

print(
    f"Misclassified       : "
    f"{misclassified_count}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 70)

per_class_errors = {}

for class_index, class_name in enumerate(class_names):

    total = np.sum(
        y_true == class_index
    )

    correct = np.sum(
        (y_true == class_index)
        &
        (y_pred == class_index)
    )

    errors = total - correct

    error_rate = (
        errors / total
        if total > 0
        else 0
    )

    per_class_errors[class_name] = {
        "total": int(total),
        "correct": int(correct),
        "errors": int(errors),
        "error_rate": float(error_rate)
    }

    print(
        f"{class_name:<25} "
        f"Total: {total:<5} "
        f"Correct: {correct:<5} "
        f"Errors: {errors:<5} "
        f"Error rate: {error_rate:.2%}"
    )


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

print("\n" + "=" * 70)
print("MOST COMMON CLASS CONFUSIONS")
print("=" * 70)

confusions = []

for true_class in range(len(class_names)):

    for predicted_class in range(len(class_names)):

        if true_class == predicted_class:
            continue

        count = cm[
            true_class,
            predicted_class
        ]

        if count > 0:

            confusions.append(
                (
                    int(count),
                    class_names[true_class],
                    class_names[predicted_class]
                )
            )


confusions.sort(
    reverse=True
)


for count, true_name, predicted_name in confusions[:20]:

    print(
        f"{true_name:<25} "
        f"-> {predicted_name:<25} "
        f": {count}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# COLLECT MISCLASSIFIED IMAGE PATHS
# ============================================================

print("\n" + "=" * 70)
print("COLLECTING MISCLASSIFIED IMAGES")
print("=" * 70)

misclassified_images = []

image_paths = test_dataset.file_paths


for index in range(len(y_true)):

    true_class = y_true[index]

    predicted_class = y_pred[index]

    if true_class != predicted_class:

        confidence = float(
            probabilities[index][predicted_class]
        )

        misclassified_images.append(
            {
                "image": image_paths[index],
                "actual": class_names[true_class],
                "predicted": class_names[predicted_class],
                "confidence": confidence
            }
        )


print(
    f"\nMisclassified images found: "
    f"{len(misclassified_images)}"
)


# ============================================================
# CREATE VISUAL ERROR-CASE FOLDERS
# ============================================================

print("\n" + "=" * 70)
print("CREATING VISUAL ERROR-CASE FOLDERS")
print("=" * 70)


# Remove old error-case folders only.
# Dataset train/val/test directories are NOT touched.

if ERROR_CASES_DIR.exists():

    for item in ERROR_CASES_DIR.iterdir():

        if item.is_dir():

            shutil.rmtree(item)

        elif item.is_file():

            item.unlink()


ERROR_CASES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Keep count of each confusion pair
error_case_counts = Counter()


for item in misclassified_images:

    actual = item["actual"]

    predicted = item["predicted"]

    source_image = Path(
        item["image"]
    )

    # Create safe folder names
    actual_folder = actual.replace(
        " ",
        "_"
    )

    predicted_folder = predicted.replace(
        " ",
        "_"
    )

    pair_folder_name = (
        f"{actual_folder}_to_{predicted_folder}"
    )

    pair_folder = (
        ERROR_CASES_DIR / pair_folder_name
    )

    pair_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Add unique number to avoid filename collision
    error_case_counts[pair_folder_name] += 1

    number = error_case_counts[
        pair_folder_name
    ]

    destination_name = (
        f"{number:03d}_"
        f"{source_image.name}"
    )

    destination = (
        pair_folder / destination_name
    )

    shutil.copy2(
        source_image,
        destination
    )


# ============================================================
# PRINT ERROR CASE SUMMARY
# ============================================================

print("\nVisual error cases created:")

for folder_name, count in sorted(
    error_case_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{folder_name:<55} : {count}"
    )


print(
    f"\nTotal visual error images copied: "
    f"{sum(error_case_counts.values())}"
)

print(
    "\nError-case directory:"
)

print(ERROR_CASES_DIR)


# ============================================================
# SAVE MISCLASSIFIED IMAGE REPORT
# ============================================================

with open(
    MISCLASSIFIED_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "APPLELEAF9 MISCLASSIFIED IMAGE REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total test images: {len(y_true)}\n"
    )

    f.write(
        f"Correct predictions: "
        f"{len(y_true) - misclassified_count}\n"
    )

    f.write(
        f"Misclassified images: "
        f"{misclassified_count}\n\n"
    )

    for number, item in enumerate(
        misclassified_images,
        start=1
    ):

        f.write(
            f"Misclassified {number}\n"
        )

        f.write(
            f"Image     : {item['image']}\n"
        )

        f.write(
            f"Actual    : {item['actual']}\n"
        )

        f.write(
            f"Predicted : {item['predicted']}\n"
        )

        f.write(
            f"Confidence: "
            f"{item['confidence']:.4f}\n"
        )

        f.write("-" * 70 + "\n")


# ============================================================
# SAVE COMPLETE ERROR ANALYSIS REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "APPLELEAF9 MODEL ERROR ANALYSIS REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test dataset: {TEST_DIR}\n"
    )

    f.write(
        f"Model: {MODEL_PATH}\n\n"
    )

    f.write(
        f"Total test images: {len(y_true)}\n"
    )

    f.write(
        f"Correct predictions: "
        f"{len(y_true) - misclassified_count}\n"
    )

    f.write(
        f"Misclassified images: "
        f"{misclassified_count}\n"
    )

    f.write(
        f"Test accuracy: {accuracy:.4%}\n\n"
    )

    # --------------------------------------------------------
    # Per-class errors
    # --------------------------------------------------------

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "PER-CLASS ERROR ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for class_name, data in per_class_errors.items():

        f.write(
            f"{class_name:<25} "
            f"Total: {data['total']:<5} "
            f"Correct: {data['correct']:<5} "
            f"Errors: {data['errors']:<5} "
            f"Error rate: "
            f"{data['error_rate']:.2%}\n"
        )

    # --------------------------------------------------------
    # Common confusions
    # --------------------------------------------------------

    f.write(
        "\n"
        + "=" * 70
        + "\n"
    )

    f.write(
        "MOST COMMON CLASS CONFUSIONS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for count, true_name, predicted_name in confusions[:20]:

        f.write(
            f"{true_name:<25} "
            f"-> {predicted_name:<25} "
            f": {count}\n"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    f.write(
        "\n"
        + "=" * 70
        + "\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    f.write(
        "\n\n"
        + "=" * 70
        + "\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nError analysis report:"
)

print(REPORT_FILE)

print(
    "\nMisclassified image list:"
)

print(MISCLASSIFIED_FILE)

print(
    "\nVisual error cases:"
)

print(ERROR_CASES_DIR)

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "\nOpen the error_cases folder and inspect "
    "the most common confusion groups."
)

print(
    "\nDO NOT retrain the model yet."
)