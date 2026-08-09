from pathlib import Path
import json
import numpy as np
import tensorflow as tf


# ============================================================
# APPLELEAF9 IMPROVED CNN TRAINING
# EXPERIMENT 2
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops"
)

TRAIN_DIR = PROJECT_DIR / "data" / "splits" / "train"
VAL_DIR = PROJECT_DIR / "data" / "splits" / "val"
TEST_DIR = PROJECT_DIR / "data" / "splits" / "test"

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports" / "model" / "improved"

BEST_MODEL_PATH = MODEL_DIR / "appleleaf9_improved_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "appleleaf9_improved_final.keras"

HISTORY_PATH = REPORT_DIR / "improved_training_history.json"
REPORT_PATH = REPORT_DIR / "improved_training_report.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("APPLELEAF9 IMPROVED CNN TRAINING")
print("EXPERIMENT 2")
print("=" * 70)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not TRAIN_DIR.exists():
    print("\nERROR: Training directory does not exist.")
    raise SystemExit

if not VAL_DIR.exists():
    print("\nERROR: Validation directory does not exist.")
    raise SystemExit

if not TEST_DIR.exists():
    print("\nERROR: Test directory does not exist.")
    raise SystemExit


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY PATHS
# ============================================================

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)

print("\nTest directory:")
print(TEST_DIR)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


print("\nLoading validation dataset...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

num_classes = len(class_names)

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {num_classes}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = MODEL_DIR / "improved_class_names.json"

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# COUNT TRAINING IMAGES
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_counts = {}

for index, class_name in enumerate(class_names):

    class_directory = TRAIN_DIR / class_name

    image_count = len(
        list(class_directory.glob("*"))
    )

    class_counts[index] = image_count

    print(
        f"{class_name:<25}: "
        f"{image_count}"
    )


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

total_images = sum(
    class_counts.values()
)

class_weights = {}

for class_index in range(num_classes):

    count = class_counts[class_index]

    class_weights[class_index] = (
        total_images
        /
        (num_classes * count)
    )


print("\nClass weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index]:<25}: "
        f"{weight:.4f}"
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.15
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )

    ],
    name="data_augmentation"
)


# ============================================================
# BUILD IMPROVED CNN
# ============================================================

print("\n" + "=" * 70)
print("BUILDING IMPROVED CNN")
print("=" * 70)


model = tf.keras.Sequential(

    [

        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        data_augmentation,

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation(
            "relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation(
            "relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation(
            "relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 4
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation(
            "relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),

        # ----------------------------------------------------
        # DENSE LAYER
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.5
        ),

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )

    ],

    name="appleleaf9_improved_cnn"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_callback = (
    tf.keras.callbacks.ModelCheckpoint(

        filepath=BEST_MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    )
)


early_stopping_callback = (
    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=5,

        restore_best_weights=True,

        verbose=1
    )
)


reduce_lr_callback = (
    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    )
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING IMPROVED MODEL TRAINING")
print("=" * 70)


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {}

for key, values in history.history.items():

    history_data[key] = [
        float(value)
        for value in values
    ]


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print(
    f"\nTest Loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy:.4%}"
)


# ============================================================
# SAVE TRAINING REPORT
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    )
    + 1
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "APPLELEAF9 IMPROVED CNN TRAINING REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "EXPERIMENT: 2\n\n"
    )

    f.write(
        f"Training images: "
        f"{total_images}\n"
    )

    f.write(
        f"Validation images: "
        f"{len(list(VAL_DIR.glob('*/*')))}\n"
    )

    f.write(
        f"Test images: "
        f"{len(list(TEST_DIR.glob('*/*')))}\n\n"
    )

    f.write(
        f"Number of classes: "
        f"{num_classes}\n\n"
    )

    f.write(
        "Architecture:\n"
    )

    f.write(
        "- Data augmentation\n"
    )

    f.write(
        "- Batch Normalization\n"
    )

    f.write(
        "- 4 convolutional blocks\n"
    )

    f.write(
        "- Global Average Pooling\n"
    )

    f.write(
        "- Dense(256)\n"
    )

    f.write(
        "- Dropout(0.5)\n"
    )

    f.write(
        "- Softmax output\n\n"
    )

    f.write(
        "Class weighting: Enabled\n\n"
    )

    f.write(
        f"Epochs configured: "
        f"{EPOCHS}\n"
    )

    f.write(
        f"Epochs completed: "
        f"{len(history.history['loss'])}\n"
    )

    f.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4%}\n\n"
    )

    f.write(
        f"Test loss: "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{test_accuracy:.4%}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nBest validation accuracy:")
print(f"{best_val_accuracy:.4%}")

print("\nTest accuracy:")
print(f"{test_accuracy:.4%}")

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nTraining history:")
print(HISTORY_PATH)

print("\nTraining report:")
print(REPORT_PATH)

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nDo NOT run error analysis yet."
)

print(
    "Send me the complete training output first."
)