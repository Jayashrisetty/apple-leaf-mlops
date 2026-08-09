from pathlib import Path
import hashlib

# ============================================================
# APPLELEAF9 TRAIN / VAL / TEST LEAKAGE CHECK
# ============================================================

SPLIT_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\apple-leaf-mlops\data\splits"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_image_hashes(split_dir):
    """Return image hashes and their file paths."""

    hashes = {}

    for class_dir in split_dir.iterdir():

        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # MD5 is sufficient for detecting exact file duplicates
            file_hash = hashlib.md5(
                image_path.read_bytes()
            ).hexdigest()

            hashes[file_hash] = image_path

    return hashes


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("APPLELEAF9 SPLIT LEAKAGE CHECK")
print("=" * 70)

print("\nSplit directory:")
print(SPLIT_DIR)

if not SPLIT_DIR.exists():

    print("\nERROR: Split directory does not exist.")
    exit()


# ------------------------------------------------------------
# Check each split
# ------------------------------------------------------------

train_dir = SPLIT_DIR / "train"
val_dir = SPLIT_DIR / "val"
test_dir = SPLIT_DIR / "test"

print("\nScanning TRAIN...")
train_hashes = get_image_hashes(train_dir)
print(f"Train images: {len(train_hashes)}")

print("\nScanning VALIDATION...")
val_hashes = get_image_hashes(val_dir)
print(f"Validation images: {len(val_hashes)}")

print("\nScanning TEST...")
test_hashes = get_image_hashes(test_dir)
print(f"Test images: {len(test_hashes)}")


# ============================================================
# CROSS-SPLIT DUPLICATE CHECK
# ============================================================

train_val = set(train_hashes) & set(val_hashes)
train_test = set(train_hashes) & set(test_hashes)
val_test = set(val_hashes) & set(test_hashes)

total_leakage = (
    len(train_val)
    + len(train_test)
    + len(val_test)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE RESULTS")
print("=" * 70)

print(
    f"\nTrain <-> Validation duplicates : {len(train_val)}"
)

print(
    f"Train <-> Test duplicates       : {len(train_test)}"
)

print(
    f"Validation <-> Test duplicates  : {len(val_test)}"
)

print(
    f"\nTotal cross-split duplicates    : {total_leakage}"
)


# ============================================================
# SHOW DUPLICATE FILES IF FOUND
# ============================================================

if train_val:

    print("\nTRAIN <-> VALIDATION DUPLICATES")

    for file_hash in train_val:

        print("\nTRAIN:")
        print(train_hashes[file_hash])

        print("VALIDATION:")
        print(val_hashes[file_hash])


if train_test:

    print("\nTRAIN <-> TEST DUPLICATES")

    for file_hash in train_test:

        print("\nTRAIN:")
        print(train_hashes[file_hash])

        print("TEST:")
        print(test_hashes[file_hash])


if val_test:

    print("\nVALIDATION <-> TEST DUPLICATES")

    for file_hash in val_test:

        print("\nVALIDATION:")
        print(val_hashes[file_hash])

        print("TEST:")
        print(test_hashes[file_hash])


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if total_leakage == 0:

    print("RESULT: PASS")
    print("No exact duplicate images exist across train/val/test.")

else:

    print("RESULT: FAIL")
    print(
        f"{total_leakage} cross-split duplicate(s) detected."
    )

print("=" * 70)