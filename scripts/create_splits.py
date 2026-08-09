from pathlib import Path
import random
import shutil

# ============================================================
# APPLELEAF9 - REPRODUCIBLE STRATIFIED DATASET SPLITTING
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-final"
)

OUTPUT_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops\data\splits"
)

# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# FUNCTIONS
# ============================================================

def get_images(class_dir):
    """Return all valid images inside a class directory."""
    return [
        p
        for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_files(files, destination):
    """Copy images into destination directory."""
    destination.mkdir(parents=True, exist_ok=True)

    for file in files:
        shutil.copy2(
            file,
            destination / file.name
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("APPLELEAF9 STRATIFIED DATASET SPLITTING")
    print("=" * 70)

    print("\nSource dataset:")
    print(SOURCE_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Check source dataset
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():
        print("\nERROR: Source dataset does not exist.")
        print(SOURCE_DIR)
        return

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove previous split completely
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print("\nRemoving previous dataset splits...")
        shutil.rmtree(OUTPUT_DIR)

        print("Old splits removed.")

    # --------------------------------------------------------
    # Create fresh output directories
    # --------------------------------------------------------

    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    test_dir = OUTPUT_DIR / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    classes = sorted(
        [
            d
            for d in SOURCE_DIR.iterdir()
            if d.is_dir()
            and not d.name.startswith("CLEANING")
        ],
        key=lambda x: x.name.lower()
    )

    print(f"\nClasses found: {len(classes)}")

    total_train = 0
    total_val = 0
    total_test = 0

    split_report = []

    # Use a dedicated random generator
    rng = random.Random(SEED)

    # ========================================================
    # STRATIFIED SPLITTING
    # ========================================================

    for class_dir in classes:

        class_name = class_dir.name

        print(f"\nProcessing: {class_name}")

        images = get_images(class_dir)

        # Deterministic shuffle
        rng.shuffle(images)

        total = len(images)

        # ----------------------------------------------------
        # Calculate split sizes
        # ----------------------------------------------------

        train_count = int(total * TRAIN_RATIO)

        val_count = int(total * VAL_RATIO)

        test_count = (
            total
            - train_count
            - val_count
        )

        # ----------------------------------------------------
        # Split files
        # ----------------------------------------------------

        train_files = images[:train_count]

        val_files = images[
            train_count:
            train_count + val_count
        ]

        test_files = images[
            train_count + val_count:
        ]

        # ----------------------------------------------------
        # Destination directories
        # ----------------------------------------------------

        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name

        # ----------------------------------------------------
        # Copy files
        # ----------------------------------------------------

        copy_files(
            train_files,
            train_class_dir
        )

        copy_files(
            val_files,
            val_class_dir
        )

        copy_files(
            test_files,
            test_class_dir
        )

        # ----------------------------------------------------
        # Update totals
        # ----------------------------------------------------

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        split_report.append(
            (
                class_name,
                total,
                len(train_files),
                len(val_files),
                len(test_files)
            )
        )

        print(
            f"{class_name:<25}"
            f"Total: {total:<5}"
            f"Train: {len(train_files):<5}"
            f"Val: {len(val_files):<5}"
            f"Test: {len(test_files):<5}"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    total_images = (
        total_train
        + total_val
        + total_test
    )

    print("\n" + "=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    print(f"Train images : {total_train}")
    print(f"Validation   : {total_val}")
    print(f"Test images  : {total_test}")
    print(f"Total        : {total_images}")

    print("\nActual ratios:")

    print(
        f"Train       : "
        f"{total_train / total_images:.2%}"
    )

    print(
        f"Validation  : "
        f"{total_val / total_images:.2%}"
    )

    print(
        f"Test        : "
        f"{total_test / total_images:.2%}"
    )

    # ========================================================
    # SAVE SPLIT REPORT
    # ========================================================

    report_dir = Path(
        r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops\reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = (
        report_dir /
        "dataset_split_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "APPLELEAF9 DATASET SPLIT REPORT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Source dataset: {SOURCE_DIR}\n"
        )

        f.write(
            f"Output directory: {OUTPUT_DIR}\n"
        )

        f.write(
            f"Random seed: {SEED}\n\n"
        )

        f.write("Split ratios:\n")
        f.write("Train: 70%\n")
        f.write("Validation: 15%\n")
        f.write("Test: 15%\n\n")

        f.write(
            f"{'Class':<25}"
            f"{'Total':<10}"
            f"{'Train':<10}"
            f"{'Val':<10}"
            f"{'Test':<10}\n"
        )

        f.write("-" * 65 + "\n")

        for row in split_report:

            class_name, total, train, val, test = row

            f.write(
                f"{class_name:<25}"
                f"{total:<10}"
                f"{train:<10}"
                f"{val:<10}"
                f"{test:<10}\n"
            )

        f.write("\n")

        f.write(
            f"Total images: {total_images}\n"
        )

        f.write(
            f"Training images: {total_train}\n"
        )

        f.write(
            f"Validation images: {total_val}\n"
        )

        f.write(
            f"Test images: {total_test}\n"
        )

    print("\nReport saved to:")
    print(report_file)

    print("\n" + "=" * 70)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()