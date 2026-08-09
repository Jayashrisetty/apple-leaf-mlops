from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json


# ============================================================
# APPLELEAF9 BASELINE CNN TRAINING
# ============================================================

BASE_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops"
)

TRAIN_DIR = BASE_DIR / "data" / "splits" / "train"
VAL_DIR = BASE_DIR / "data" / "splits" / "val"
TEST_DIR = BASE_DIR / "data" / "splits" / "test"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 70)
print("APPLELEAF9 BASELINE CNN TRAINING")
print("=" * 70)

print("\nTrain directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)

print("\nTest directory:")
print(TEST_DIR)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {num_classes}")


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# ============================================================
# BASELINE CNN
# ============================================================

model = keras.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),

        data_augmentation,

        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "appleleaf9_baseline_best.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4%}")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "appleleaf9_baseline_final.keras"

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = MODEL_DIR / "class_names.json"

with open(class_file, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved:")
print(class_file)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = REPORT_DIR / "training_history.json"

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(history_file, "w", encoding="utf-8") as f:
    json.dump(history_data, f, indent=4)

print("\nTraining history saved:")
print(history_file)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary_file = REPORT_DIR / "baseline_training_report.txt"

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("APPLELEAF9 BASELINE CNN TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset: AppleLeaf9-final\n")
    f.write(f"Image size: {IMAGE_SIZE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Random seed: {SEED}\n\n")

    f.write("Classes:\n")

    for i, name in enumerate(class_names):
        f.write(f"{i}: {name}\n")

    f.write("\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.6%}\n")

print("\nTraining report saved:")
print(summary_file)

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)