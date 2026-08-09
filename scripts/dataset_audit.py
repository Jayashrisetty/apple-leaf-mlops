from pathlib import Path
from PIL import Image
from collections import Counter
import hashlib

# ============================================================
# DATASET LOCATION
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-final"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# STORAGE
# ============================================================

class_counts = Counter()
image_sizes = Counter()

corrupted_images = []
image_hashes = {}
duplicate_images = []

total_images = 0

# ============================================================
# CHECK DATASET PATH
# ============================================================

print("=" * 70)
print("APPLELEAF9 DATASET AUDIT")
print("=" * 70)

print("\nDataset path:")
print(DATASET_DIR)

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder was not found!")
    print("Check the dataset path.")
    exit()

# ============================================================
# SCAN DATASET
# ============================================================

for class_dir in sorted(DATASET_DIR.iterdir()):

    # Ignore files
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    print(f"\nChecking: {class_name}")

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1
        class_counts[class_name] += 1

        try:

            # ------------------------------------------------
            # Check image integrity
            # ------------------------------------------------

            with Image.open(image_path) as img:
                img.verify()

            # ------------------------------------------------
            # Read image dimensions
            # ------------------------------------------------

            with Image.open(image_path) as img:
                width, height = img.size
                image_sizes[(width, height)] += 1

            # ------------------------------------------------
            # Detect exact duplicate files
            # ------------------------------------------------

            file_hash = hashlib.md5(
                image_path.read_bytes()
            ).hexdigest()

            if file_hash in image_hashes:

                duplicate_images.append(
                    (
                        str(image_path),
                        image_hashes[file_hash]
                    )
                )

            else:

                image_hashes[file_hash] = str(image_path)

        except Exception as e:

            corrupted_images.append(
                (
                    str(image_path),
                    str(e)
                )
            )

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name:<25} : {count}")

print("\n")
print("=" * 70)
print("TOTAL IMAGES")
print("=" * 70)

print(f"Total images: {total_images}")

print("\n")
print("=" * 70)
print("IMAGE DIMENSIONS")
print("=" * 70)

for size, count in image_sizes.most_common(20):
    print(f"{size[0]} x {size[1]} : {count}")

print("\n")
print("=" * 70)
print("CORRUPTED IMAGES")
print("=" * 70)

print(f"Corrupted images: {len(corrupted_images)}")

for image, error in corrupted_images[:20]:
    print(f"\n{image}")
    print(error)

print("\n")
print("=" * 70)
print("DUPLICATE IMAGES")
print("=" * 70)

print(f"Duplicate images: {len(duplicate_images)}")

for duplicate, original in duplicate_images[:20]:
    print(f"\nDuplicate : {duplicate}")
    print(f"Original  : {original}")

print("\n")
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)