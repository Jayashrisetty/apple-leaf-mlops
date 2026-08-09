from pathlib import Path
import shutil

# ============================================================
# APPLELEAF9 CONFIRMED DUPLICATE CLEANING
# ============================================================

SOURCE = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-main"
)

OUTPUT = Path(
    r"C:\Users\jayshear\OneDrive\Jay-Mlops\AppleLeaf9-cleaned"
)

# ------------------------------------------------------------
# CONFIRMED CROSS-CLASS DUPLICATES
#
# For each pair, the SECOND image is removed.
# The FIRST image is retained.
# ------------------------------------------------------------

DUPLICATE_PAIRS = [
    # Alternaria <-> Grey spot
    (
        r"Alternaria leaf spot\Alternaria leaf spot (142).jpg",
        r"Grey spot\Grey spot (291).jpg",
    ),
    (
        r"Alternaria leaf spot\Alternaria leaf spot (93).jpg",
        r"Grey spot\Grey spot (330).jpg",
    ),
    (
        r"Alternaria leaf spot\Alternaria leaf spot (94).jpg",
        r"Grey spot\Grey spot (328).jpg",
    ),

    # Rust <-> Scab
    (
        r"Rust\Rust (1165).jpg",
        r"Scab\Scab (69).jpg",
    ),
    (
        r"Rust\Rust (1218).jpg",
        r"Scab\Scab (193).jpg",
    ),
    (
        r"Rust\Rust (1405).jpg",
        r"Scab\Scab (360).jpg",
    ),
    (
        r"Rust\Rust (1524).jpg",
        r"Scab\Scab (518).jpg",
    ),
    (
        r"Rust\Rust (1540).jpg",
        r"Scab\Scab (361).jpg",
    ),
    (
        r"Rust\Rust (1599).jpg",
        r"Scab\Scab (251).jpg",
    ),
    (
        r"Rust\Rust (1639).jpg",
        r"Scab\Scab (252).jpg",
    ),
    (
        r"Rust\Rust (1671).jpg",
        r"Scab\Scab (326).jpg",
    ),
    (
        r"Rust\Rust (1692).jpg",
        r"Scab\Scab (505).jpg",
    ),
    (
        r"Rust\Rust (1733).jpg",
        r"Scab\Scab (29).jpg",
    ),
    (
        r"Rust\Rust (1872).jpg",
        r"Scab\Scab (259).jpg",
    ),
    (
        r"Rust\Rust (1884).jpg",
        r"Scab\Scab (579).jpg",
    ),
    (
        r"Rust\Rust (1894).jpg",
        r"Scab\Scab (364).jpg",
    ),
    (
        r"Rust\Rust (1895).jpg",
        r"Scab\Scab (554).jpg",
    ),
    (
        r"Rust\Rust (1994).jpg",
        r"Scab\Scab (50).jpg",
    ),
    (
        r"Rust\Rust (2060).jpg",
        r"Scab\Scab (114).jpg",
    ),
    (
        r"Rust\Rust (2203).jpg",
        r"Scab\Scab (140).jpg",
    ),
    (
        r"Rust\Rust (2350).jpg",
        r"Scab\Scab (7).jpg",
    ),
    (
        r"Rust\Rust (2412).jpg",
        r"Scab\Scab (566).jpg",
    ),
    (
        r"Rust\Rust (2448).jpg",
        r"Scab\Scab (14).jpg",
    ),
    (
        r"Rust\Rust (710).jpg",
        r"Scab\Scab (515).jpg",
    ),
    (
        r"Rust\Rust (773).jpg",
        r"Scab\Scab (204).jpg",
    ),
    (
        r"Rust\Rust (781).jpg",
        r"Scab\Scab (530).jpg",
    ),
    (
        r"Rust\Rust (862).jpg",
        r"Scab\Scab (356).jpg",
    ),
    (
        r"Rust\Rust (932).jpg",
        r"Scab\Scab (387).jpg",
    ),
    (
        r"Rust\Rust (963).jpg",
        r"Scab\Scab (439).jpg",
    ),
    (
        r"Rust\Rust (984).jpg",
        r"Scab\Scab (72).jpg",
    ),
]


def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


print("=" * 70)
print("APPLELEAF9 CLEAN DATASET CREATION")
print("=" * 70)

print(f"\nSource : {SOURCE}")
print(f"Output : {OUTPUT}")

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

# ------------------------------------------------------------
# STEP 1: Copy original dataset
# ------------------------------------------------------------

if OUTPUT.exists():
    print("\nOutput dataset already exists.")
    print("Stopping to prevent accidental overwrite.")
    print(f"Delete it manually if you want to recreate: {OUTPUT}")
    raise SystemExit

print("\nCopying original dataset...")
shutil.copytree(SOURCE, OUTPUT)

print("Original dataset copied successfully.")

# ------------------------------------------------------------
# STEP 2: Remove confirmed duplicate copies
# ------------------------------------------------------------

print("\nRemoving confirmed duplicates...")

removed = []
missing = []

for i, (keep_rel, remove_rel) in enumerate(DUPLICATE_PAIRS, start=1):

    remove_path = OUTPUT / remove_rel
    keep_path = OUTPUT / keep_rel

    print(f"Processing {i}/{len(DUPLICATE_PAIRS)}")

    if not keep_path.exists():
        print(f"  WARNING: keep file missing")
        print(f"  {keep_path}")

    if remove_path.exists():
        remove_path.unlink()
        removed.append(remove_rel)
        print(f"  Removed: {remove_rel}")
    else:
        missing.append(remove_rel)
        print(f"  WARNING: already missing: {remove_rel}")

# ------------------------------------------------------------
# STEP 3: Count final images
# ------------------------------------------------------------

print("\nCounting final dataset...")

class_counts = {}

for class_dir in sorted(OUTPUT.iterdir()):

    if not class_dir.is_dir():
        continue

    count = len(get_images(class_dir))
    class_counts[class_dir.name] = count

total = sum(class_counts.values())

# ------------------------------------------------------------
# STEP 4: Save report
# ------------------------------------------------------------

report = OUTPUT / "CLEANING_REPORT.txt"

with open(report, "w", encoding="utf-8") as f:

    f.write("APPLELEAF9 CLEANING REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Original dataset: {SOURCE}\n")
    f.write(f"Cleaned dataset : {OUTPUT}\n\n")

    f.write(f"Confirmed duplicate pairs: {len(DUPLICATE_PAIRS)}\n")
    f.write(f"Removed images: {len(removed)}\n")
    f.write(f"Missing removal files: {len(missing)}\n\n")

    f.write("FINAL CLASS COUNTS\n")
    f.write("-" * 40 + "\n")

    for name, count in class_counts.items():
        f.write(f"{name}: {count}\n")

    f.write("\n")
    f.write(f"FINAL TOTAL: {total}\n")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("\nRemoved images:", len(removed))
print("Final images:", total)

print("\nFINAL CLASS COUNTS")

for name, count in class_counts.items():
    print(f"{name:25s}: {count}")

print(f"\nCleaning report:")
print(report)

print("\nOriginal dataset was NOT modified.")