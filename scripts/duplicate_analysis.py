from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("../AppleLeaf9-main").resolve()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Images with pHash distance <= this value
# will be considered visually similar candidates.
HAMMING_THRESHOLD = 5


# ============================================================
# BK-TREE FOR PERCEPTUAL HASH SEARCH
# ============================================================

class BKTreeNode:

    def __init__(self, hash_value, image_path):
        self.hash_value = hash_value
        self.image_paths = [image_path]
        self.children = {}


class BKTree:

    def __init__(self):
        self.root = None

    def add(self, hash_value, image_path):

        if self.root is None:

            self.root = BKTreeNode(
                hash_value,
                image_path
            )

            return

        node = self.root

        while True:

            distance = hash_value - node.hash_value

            if distance == 0:

                node.image_paths.append(image_path)
                return

            if distance not in node.children:

                node.children[distance] = BKTreeNode(
                    hash_value,
                    image_path
                )

                return

            node = node.children[distance]

    def search(self, hash_value, max_distance):

        results = []

        if self.root is None:
            return results

        stack = [self.root]

        while stack:

            node = stack.pop()

            distance = hash_value - node.hash_value

            if distance <= max_distance:

                for image_path in node.image_paths:

                    results.append(
                        (
                            distance,
                            image_path
                        )
                    )

            lower = distance - max_distance
            upper = distance + max_distance

            for child_distance, child_node in node.children.items():

                if lower <= child_distance <= upper:

                    stack.append(child_node)

        return results


# ============================================================
# DATASET CHECK
# ============================================================

print("=" * 70)
print("APPLELEAF9 NEAR-DUPLICATE ANALYSIS")
print("=" * 70)

print(f"\nDataset:")
print(DATASET_DIR)

if not DATASET_DIR.exists():

    print("\nERROR: Dataset directory not found.")

    raise SystemExit


# ============================================================
# BUILD HASH INDEX
# ============================================================

tree = BKTree()

image_records = []

total_images = 0
failed_images = 0

print("\nGenerating perceptual hashes...\n")


for class_dir in sorted(DATASET_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    print(f"Processing: {class_name}")

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1

        try:

            with Image.open(image_path) as img:

                # Convert to RGB to handle grayscale/RGBA images
                img = img.convert("RGB")

                # Perceptual hash
                phash = imagehash.phash(img)

            tree.add(
                phash,
                str(image_path)
            )

            image_records.append(
                (
                    str(image_path),
                    class_name,
                    phash
                )
            )

        except Exception as e:

            failed_images += 1

            print(
                f"Could not process: {image_path}"
            )


# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

print("\n")
print("=" * 70)
print("SEARCHING FOR VISUALLY SIMILAR IMAGES")
print("=" * 70)

near_duplicates = []

checked_pairs = set()


for index, (image_path, class_name, phash) in enumerate(
    image_records
):

    matches = tree.search(
        phash,
        HAMMING_THRESHOLD
    )

    for distance, matched_path in matches:

        if image_path == matched_path:
            continue

        # Create an order-independent pair
        pair = tuple(
            sorted(
                [
                    image_path,
                    matched_path
                ]
            )
        )

        if pair in checked_pairs:
            continue

        checked_pairs.add(pair)

        near_duplicates.append(
            (
                distance,
                image_path,
                matched_path
            )
        )


# ============================================================
# SORT RESULTS
# ============================================================

near_duplicates.sort(
    key=lambda x: x[0]
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS RESULTS")
print("=" * 70)

print(f"\nTotal images scanned : {total_images}")

print(f"Failed images        : {failed_images}")

print(
    f"Near-duplicate pairs : {len(near_duplicates)}"
)

print(
    f"Hamming threshold    : {HAMMING_THRESHOLD}"
)


# ============================================================
# SHOW RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("NEAR-DUPLICATE EXAMPLES")
print("=" * 70)

if not near_duplicates:

    print("\nNo near-duplicate pairs found.")

else:

    for distance, image1, image2 in near_duplicates[:50]:

        print("\n----------------------------------------")

        print(
            f"Hamming distance: {distance}"
        )

        print(
            f"Image 1:\n{image1}"
        )

        print(
            f"Image 2:\n{image2}"
        )


# ============================================================
# CROSS-CLASS NEAR DUPLICATES
# ============================================================

print("\n")
print("=" * 70)
print("CROSS-CLASS NEAR DUPLICATES")
print("=" * 70)


class_lookup = {
    path: class_name
    for path, class_name, phash
    in image_records
}


cross_class_pairs = []


for distance, image1, image2 in near_duplicates:

    class1 = class_lookup[image1]
    class2 = class_lookup[image2]

    if class1 != class2:

        cross_class_pairs.append(
            (
                distance,
                class1,
                class2,
                image1,
                image2
            )
        )


print(
    f"\nCross-class near-duplicate pairs: "
    f"{len(cross_class_pairs)}"
)


for (
    distance,
    class1,
    class2,
    image1,
    image2
) in cross_class_pairs[:50]:

    print("\n----------------------------------------")

    print(
        f"Hamming distance: {distance}"
    )

    print(
        f"Class 1: {class1}"
    )

    print(
        f"Image 1: {image1}"
    )

    print(
        f"Class 2: {class2}"
    )

    print(
        f"Image 2: {image2}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("NEAR-DUPLICATE ANALYSIS COMPLETE")
print("=" * 70)