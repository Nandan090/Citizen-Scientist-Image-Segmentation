"""
YOLO Dataset Preparation + Training Script
===========================================
Converts SAM2 segmentation masks into YOLOv8 segmentation format,
organizes the dataset, and launches training.

Pipeline:
    SAM2 masks (PNG)  →  YOLO polygon labels (.txt)  →  Train YOLOv8-seg

Expected input structure (from your SAM.py output):
    <OUTPUT_DIR>/
    ├── images/
    │   ├── Fallopia_japonica/
    │   │   ├── img_001.jpg
    │   │   └── ...
    │   └── Lupinus_polyphyllus/
    │       └── ...
    └── masks/
        ├── Fallopia_japonica/
        │   ├── mask_img_001.png
        │   └── ...
        └── Lupinus_polyphyllus/
            └── ...

Output structure (YOLO-ready):
    <OUTPUT_DIR>/
    └── yolo_dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml

Usage:
    python prepare_yolo_dataset.py

Then train with:
    yolo segment train data=yolo_dataset/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=512
"""

import os
import cv2
import shutil
import random
import logging
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — edit these as needed
# ─────────────────────────────────────────────
OUTPUT_DIR       = os.getcwd()
IMAGE_ROOT       = f"{OUTPUT_DIR}/images"       # Your downloaded images
MASK_ROOT        = f"{OUTPUT_DIR}/masks"        # SAM2 output masks
YOLO_DATASET_DIR = f"{OUTPUT_DIR}/yolo_dataset" # Where YOLO dataset will be created

TRAIN_SPLIT      = 0.8          # 80% train, 20% val
BACKGROUND_CLASS = 255          # Pixel value used for background in your masks (from SAM.py)
MIN_CONTOUR_AREA = 100          # Ignore tiny contours (noise), in pixels
RANDOM_SEED      = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1: BUILD CLASS MAP
# ─────────────────────────────────────────────
def build_class_map(mask_root: Path) -> dict[str, int]:
    """
    Derive class names and IDs from sorted top-level mask folders.
    Mirrors the class ordering used in SAM.py.
    """
    species_folders = sorted([
        d.name for d in mask_root.iterdir() if d.is_dir()
    ])
    class_map = {name: idx for idx, name in enumerate(species_folders)}
    logger.info(f"Class map: {class_map}")
    return class_map


# ─────────────────────────────────────────────
# STEP 2: MASK → YOLO POLYGON LABEL
# ─────────────────────────────────────────────
def mask_to_yolo_polygons(
    mask_path: Path,
    class_id: int,
    img_w: int,
    img_h: int,
) -> list[str]:
    """
    Convert a binary mask PNG to YOLO segmentation label lines.

    Each line format:
        class_id x1 y1 x2 y2 ... xN yN   (all normalized 0-1)

    Returns a list of label strings (one per contour/object instance).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"  Could not read mask: {mask_path}")
        return []

    # Resize mask to match image dimensions if needed
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Isolate foreground: pixels with class_id value (not background 255)
    binary = np.where(mask == class_id, 255, 0).astype(np.uint8)

    # Find contours of plant regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for contour in contours:
        # Skip tiny noisy contours
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        # Simplify contour slightly to reduce polygon complexity
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points for a valid polygon
        if len(approx) < 3:
            continue

        # Normalize coordinates to [0, 1]
        points = approx.reshape(-1, 2)
        norm_points = []
        for x, y in points:
            norm_points.append(round(float(x) / img_w, 6))
            norm_points.append(round(float(y) / img_h, 6))

        coords_str = " ".join(map(str, norm_points))
        label_lines.append(f"{class_id} {coords_str}")

    return label_lines


# ─────────────────────────────────────────────
# STEP 3: COLLECT ALL IMAGE-MASK PAIRS
# ─────────────────────────────────────────────
def collect_pairs(
    image_root: Path,
    mask_root: Path,
    class_map: dict[str, int],
) -> list[tuple[Path, Path, int]]:
    """
    Match each image to its corresponding mask file.
    Returns list of (image_path, mask_path, class_id) tuples.
    """
    pairs = []
    missing_masks = 0

    for species_name, class_id in class_map.items():
        img_dir  = image_root / species_name
        mask_dir = mask_root  / species_name

        if not img_dir.exists():
            logger.warning(f"Image folder not found: {img_dir}")
            continue

        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix not in IMAGE_EXTENSIONS:
                continue

            # SAM.py saves masks as mask_<stem>.png
            mask_name = f"mask_{img_path.stem}.png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                missing_masks += 1
                continue

            pairs.append((img_path, mask_path, class_id))

    logger.info(f"Found {len(pairs)} image-mask pairs. Missing masks: {missing_masks}")
    return pairs


# ─────────────────────────────────────────────
# STEP 4: SPLIT INTO TRAIN / VAL
# ─────────────────────────────────────────────
def split_pairs(
    pairs: list,
    train_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """Stratified split: maintain class balance across train/val."""
    random.seed(seed)

    # Group by class
    by_class: dict[int, list] = {}
    for pair in pairs:
        cid = pair[2]
        by_class.setdefault(cid, []).append(pair)

    train_pairs, val_pairs = [], []
    for cid, class_pairs in by_class.items():
        random.shuffle(class_pairs)
        n_train = int(len(class_pairs) * train_ratio)
        train_pairs.extend(class_pairs[:n_train])
        val_pairs.extend(class_pairs[n_train:])
        logger.info(
            f"  Class {cid}: {n_train} train, {len(class_pairs) - n_train} val"
        )

    return train_pairs, val_pairs


# ─────────────────────────────────────────────
# STEP 5: WRITE YOLO DATASET
# ─────────────────────────────────────────────
def write_yolo_dataset(
    train_pairs: list,
    val_pairs: list,
    yolo_dir: Path,
):
    """
    Copy images and write YOLO label .txt files into the standard structure:
        yolo_dir/images/train|val/
        yolo_dir/labels/train|val/
    """
    splits = {"train": train_pairs, "val": val_pairs}
    stats  = {"train": {"images": 0, "skipped": 0}, "val": {"images": 0, "skipped": 0}}

    for split_name, pairs in splits.items():
        img_out_dir   = yolo_dir / "images" / split_name
        label_out_dir = yolo_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nWriting {split_name} split ({len(pairs)} pairs)...")

        for img_path, mask_path, class_id in pairs:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"  Cannot read image: {img_path}")
                stats[split_name]["skipped"] += 1
                continue

            img_h, img_w = img.shape[:2]

            # Convert mask to YOLO polygon labels
            label_lines = mask_to_yolo_polygons(mask_path, class_id, img_w, img_h)

            if not label_lines:
                # No valid contours found — skip this image
                # (YOLO needs at least one annotation per labeled image)
                stats[split_name]["skipped"] += 1
                continue

            # Copy image (use unique name: species_originalname)
            out_stem    = f"{img_path.parent.name}_{img_path.stem}"
            img_out     = img_out_dir   / f"{out_stem}{img_path.suffix}"
            label_out   = label_out_dir / f"{out_stem}.txt"

            shutil.copy2(img_path, img_out)

            with open(label_out, "w") as f:
                f.write("\n".join(label_lines))

            stats[split_name]["images"] += 1

        logger.info(
            f"  {split_name}: {stats[split_name]['images']} written, "
            f"{stats[split_name]['skipped']} skipped (no valid contours)"
        )

    return stats


# ─────────────────────────────────────────────
# STEP 6: WRITE data.yaml
# ─────────────────────────────────────────────
def write_data_yaml(yolo_dir: Path, class_map: dict[str, int]):
    """Write the YOLO data.yaml configuration file."""
    # Sort by class_id to ensure correct order
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    yaml_content = f"""# YOLOv8 Segmentation Dataset
# Auto-generated by prepare_yolo_dataset.py

path: {yolo_dir.resolve()}   # dataset root dir
train: images/train           # train images (relative to path)
val:   images/val             # val images (relative to path)

# Number of classes
nc: {len(class_names)}

# Class names (order matches class IDs from SAM.py)
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    logger.info(f"\ndata.yaml written to: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
# STEP 7: VERIFY INSTALLATION + PRINT TRAIN CMD
# ─────────────────────────────────────────────
def print_training_instructions(yaml_path: Path):
    """Print the commands needed to install and run YOLO training."""
    print("\n" + "="*60)
    print("DATASET READY — Next steps:")
    print("="*60)
    print("\n1. Install Ultralytics (if not already installed):")
    print("   pip install ultralytics")
    print("\n2. Train YOLOv8 segmentation model:")
    print(f"""
   # Nano model (fastest, least accurate)
   yolo segment train \\
       data={yaml_path} \\
       model=yolov8n-seg.pt \\
       epochs=100 \\
       imgsz=512 \\
       batch=16 \\
       device=0

   # Large model (slower, more accurate — recommended if you have GPU)
   yolo segment train \\
       data={yaml_path} \\
       model=yolov8l-seg.pt \\
       epochs=100 \\
       imgsz=512 \\
       batch=8 \\
       device=0
""")
    print("3. Results will be saved to: runs/segment/train/")
    print("   - Best weights: runs/segment/train/weights/best.pt")
    print("   - Training curves: runs/segment/train/results.png")
    print("\n4. Run inference on a new image:")
    print("   yolo segment predict model=runs/segment/train/weights/best.pt source=your_image.jpg")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    image_root = Path(IMAGE_ROOT)
    mask_root  = Path(MASK_ROOT)
    yolo_dir   = Path(YOLO_DATASET_DIR)

    # Validate inputs
    if not image_root.exists():
        logger.error(f"Image root not found: {image_root}")
        return
    if not mask_root.exists():
        logger.error(f"Mask root not found: {mask_root}")
        return

    logger.info("="*60)
    logger.info("YOLO Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Image root : {image_root}")
    logger.info(f"Mask root  : {mask_root}")
    logger.info(f"Output dir : {yolo_dir}")

    # Step 1: Build class map
    class_map = build_class_map(mask_root)

    # Step 2: Collect image-mask pairs
    pairs = collect_pairs(image_root, mask_root, class_map)
    if not pairs:
        logger.error("No image-mask pairs found. Check your IMAGE_ROOT and MASK_ROOT paths.")
        return

    # Step 3: Train/val split
    logger.info(f"\nSplitting dataset ({int(TRAIN_SPLIT*100)}% train / {int((1-TRAIN_SPLIT)*100)}% val)...")
    train_pairs, val_pairs = split_pairs(pairs, TRAIN_SPLIT, RANDOM_SEED)

    # Step 4: Write YOLO dataset
    yolo_dir.mkdir(parents=True, exist_ok=True)
    stats = write_yolo_dataset(train_pairs, val_pairs, yolo_dir)

    # Step 5: Write data.yaml
    yaml_path = write_data_yaml(yolo_dir, class_map)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPARATION COMPLETE")
    logger.info(f"  Train images : {stats['train']['images']}")
    logger.info(f"  Val images   : {stats['val']['images']}")
    logger.info(f"  Skipped      : {stats['train']['skipped'] + stats['val']['skipped']} (no valid contours)")
    logger.info(f"  Classes      : {list(class_map.keys())}")
    logger.info(f"  data.yaml    : {yaml_path}")
    logger.info("="*60)

    # Print training instructions
    print_training_instructions(yaml_path)


if __name__ == "__main__":
    main()