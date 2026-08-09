from pathlib import Path
from PIL import Image
from collections import Counter
import hashlib
import shutil

# ============================================================
# APPLELEAF9 SAME-CLASS DUPLICATE REMOVAL
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-cleaned"
)

OUTPUT_DIR = Path(
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
# HEADER
# ============================================================

print("=" * 70)
print("APPLELEAF9 SAME-CLASS DUPLICATE REMOVAL")
print("=" * 70)

print("\nSource dataset:")
print(SOURCE_DIR)

print("\nFinal dataset:")
print(OUTPUT_DIR)

# ============================================================
# CHECK SOURCE
# ============================================================

if not SOURCE_DIR.exists():
    print("\nERROR: AppleLeaf9-cleaned was not found!")
    exit()

if OUTPUT_DIR.exists():
    print("\nERROR: AppleLeaf9-final already exists!")
    print("Delete/rename it first if you want to create it again.")
    exit()

# ============================================================
# FIND DUPLICATES
# ============================================================

print("\nScanning for same-class duplicates...")

duplicate_files = set()
duplicate_pairs = []

total_images = 0
class_counts = Counter()

for class_dir in sorted(SOURCE_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    # Ignore non-dataset folders such as CLEANING
    if class_dir.name.upper() == "CLEANING":
        continue

    class_name = class_dir.name

    print(f"Checking: {class_name}")

    hashes = {}

    for image_path in sorted(class_dir.rglob("*")):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1
        class_counts[class_name] += 1

        try:
            # Verify image
            with Image.open(image_path) as img:
                img.verify()

            # Calculate exact file hash
            file_hash = hashlib.md5(
                image_path.read_bytes()
            ).hexdigest()

            if file_hash in hashes:

                original = hashes[file_hash]

                duplicate_files.add(image_path)

                duplicate_pairs.append(
                    (
                        class_name,
                        image_path,
                        original
                    )
                )

            else:

                hashes[file_hash] = image_path

        except Exception as e:

            print(f"\nWARNING: Could not process:")
            print(image_path)
            print(e)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE SCAN COMPLETE")
print("=" * 70)

print(f"\nImages scanned      : {total_images}")
print(f"Same-class duplicates: {len(duplicate_files)}")

# ============================================================
# SHOW DUPLICATES
# ============================================================

if duplicate_pairs:

    print("\nDUPLICATES TO REMOVE")
    print("-" * 70)

    for i, (class_name, duplicate, original) in enumerate(
        duplicate_pairs, start=1
    ):

        print(f"\nDuplicate {i}")
        print(f"Class    : {class_name}")
        print(f"REMOVE   : {duplicate}")
        print(f"KEEP     : {original}")

else:

    print("\nNo same-class duplicates found.")

# ============================================================
# CREATE FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING APPLELEAF9-FINAL")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
removed = 0

for class_dir in sorted(SOURCE_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    if class_dir.name.upper() == "CLEANING":
        continue

    class_name = class_dir.name

    output_class_dir = OUTPUT_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Do NOT copy duplicate
        if image_path in duplicate_files:

            removed += 1

            print(
                f"REMOVE : {class_name}\\{image_path.name}"
            )

            continue

        # Preserve relative path
        relative_path = image_path.relative_to(class_dir)

        destination = output_class_dir / relative_path

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            image_path,
            destination
        )

        copied += 1

# ============================================================
# FINAL CLASS COUNTS
# ============================================================

final_class_counts = Counter()

for class_dir in sorted(OUTPUT_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            final_class_counts[class_dir.name] += 1

final_total = sum(final_class_counts.values())

# ============================================================
# REPORT
# ============================================================

report_dir = OUTPUT_DIR / "CLEANING"
report_dir.mkdir(parents=True, exist_ok=True)

report_file = report_dir / "FINAL_CLEANING_REPORT.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("APPLELEAF9 FINAL CLEANING REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Source dataset : {SOURCE_DIR}\n")
    f.write(f"Final dataset  : {OUTPUT_DIR}\n\n")

    f.write(f"Original images : {total_images}\n")
    f.write(f"Removed images  : {removed}\n")
    f.write(f"Final images    : {final_total}\n\n")

    f.write("=" * 70 + "\n")
    f.write("FINAL CLASS DISTRIBUTION\n")
    f.write("=" * 70 + "\n\n")

    for class_name, count in final_class_counts.items():
        f.write(f"{class_name:<25} : {count}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("REMOVED DUPLICATES\n")
    f.write("=" * 70 + "\n\n")

    for i, (class_name, duplicate, original) in enumerate(
        duplicate_pairs,
        start=1
    ):

        f.write(f"Duplicate {i}\n")
        f.write(f"Class    : {class_name}\n")
        f.write(f"Removed  : {duplicate}\n")
        f.write(f"Original : {original}\n\n")

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)

print(f"\nOriginal images : {total_images}")
print(f"Removed images  : {removed}")
print(f"Final images    : {final_total}")

print("\nFINAL CLASS DISTRIBUTION")

for class_name, count in final_class_counts.items():
    print(f"{class_name:<25} : {count}")

print("\nFinal dataset:")
print(OUTPUT_DIR)

print("\nCleaning report:")
print(report_file)

print("\nIMPORTANT:")
print("AppleLeaf9-main was NOT modified.")
print("AppleLeaf9-cleaned was NOT modified.")
print("A new AppleLeaf9-final dataset was created.")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)