from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

DATASET_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-main"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 70)
print("APPLELEAF9 FAST CROSS-CLASS DUPLICATE INSPECTION")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: Collect images
# ---------------------------------------------------------

images = []

for class_dir in sorted(DATASET_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    print(f"Scanning: {class_dir.name}")

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(
                (
                    class_dir.name,
                    image_path
                )
            )

print()
print(f"Images found: {len(images)}")

# ---------------------------------------------------------
# STEP 2: Generate perceptual hashes
# ---------------------------------------------------------

print()
print("Generating perceptual hashes...")

hash_groups = defaultdict(list)

failed = 0

for index, (class_name, image_path) in enumerate(images, 1):

    try:

        with Image.open(image_path) as img:

            img = img.convert("RGB")

            # Average hash
            img_hash = imagehash.phash(img)

            hash_groups[str(img_hash)].append(
                (
                    class_name,
                    image_path
                )
            )

    except Exception as e:

        failed += 1

    if index % 1000 == 0:
        print(f"Processed: {index}/{len(images)}")

# ---------------------------------------------------------
# STEP 3: Find exact perceptual-hash matches
# ---------------------------------------------------------

print()
print("=" * 70)
print("SEARCHING FOR CROSS-CLASS DUPLICATES")
print("=" * 70)

cross_class_duplicates = []

for hash_value, group in hash_groups.items():

    if len(group) < 2:
        continue

    # Compare only images sharing the same hash
    for i in range(len(group)):

        for j in range(i + 1, len(group)):

            class1, image1 = group[i]
            class2, image2 = group[j]

            if class1 != class2:

                cross_class_duplicates.append(
                    (
                        class1,
                        image1,
                        class2,
                        image2
                    )
                )

# ---------------------------------------------------------
# STEP 4: Results
# ---------------------------------------------------------

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Total images scanned : {len(images)}")
print(f"Failed images        : {failed}")
print(
    f"Cross-class duplicate pairs : "
    f"{len(cross_class_duplicates)}"
)

# ---------------------------------------------------------
# STEP 5: Display examples
# ---------------------------------------------------------

if cross_class_duplicates:

    print()
    print("CROSS-CLASS DUPLICATE EXAMPLES")
    print("-" * 70)

    for index, (
        class1,
        image1,
        class2,
        image2
    ) in enumerate(cross_class_duplicates[:50], 1):

        print()
        print(f"Pair {index}")
        print(f"Class 1 : {class1}")
        print(f"Image 1 : {image1}")
        print(f"Class 2 : {class2}")
        print(f"Image 2 : {image2}")

else:

    print()
    print("No cross-class duplicates detected.")

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)