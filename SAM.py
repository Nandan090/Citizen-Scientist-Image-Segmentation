"""
SAM2 Weakly Supervised Segmentation Script
============================================
Uses GradCAM (from trained EfficientNetV2) + SAM2 to automatically
generate pixel-level segmentation masks for invasive plant species.

Pipeline per image:
    1. GradCAM  → heatmap showing where classifier "sees" the plant
    2. Threshold → binary map + contours
    3. SAM2     → precise pixel mask from contour points
    4. Save     → mask .png with class_id / background pixels

Folder structure expected:
    base_dir/
    ├── Fallopia_japonica/
    │   ├── img001.jpg
    │   └── subfolders/     ← also read recursively
    ├── Heracleum_mantegazzianum/
    └── ...

Output mirrors the input structure under OUTPUT_MASK_ROOT.
"""

import os
import random
import logging
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# FIX: Correct SAM2 imports (not SAM1 sam_model_registry)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
# Root output directory (all inputs and outputs live here)
OUTPUT_DIR       = os.getcwd()

# Images downloaded from iNaturalist (from your download script)
BASE_DIR         = f"{OUTPUT_DIR}/images"

# Where segmentation masks will be saved
OUTPUT_MASK_ROOT = f"{OUTPUT_DIR}/masks"

# Trained EfficientNetV2 classifier checkpoint
MODEL_PATH       = f"{OUTPUT_DIR}/checkpoints/best_model.pth"

# SAM2 model files
SAM2_CHECKPOINT  = f"{OUTPUT_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l"

NUM_CLASSES        = 2
BATCH_SIZE         = 8
THRESHOLD_VALUE    = 150   # GradCAM binarization threshold (0-255)
NUM_SAMPLED_POINTS = 2     # Points sampled per contour region
BACKGROUND_CLASS   = 255   # Pixel value for background in saved mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 2. INFERENCE TRANSFORM (no random augmentation)
# ─────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 3. MODEL INITIALIZATION (called ONCE)
# ─────────────────────────────────────────────
def initialize_models():
    """Load EfficientNetV2 classifier and SAM2 predictor once."""

    # --- EfficientNetV2-L classifier ---
    logger.info("Loading EfficientNetV2-L classifier...")
    model = models.efficientnet_v2_l(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    new_state_dict = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in checkpoint.items()
    )
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    logger.info("Classifier loaded.")

    # --- SAM2 predictor ---
    logger.info("Loading SAM2...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    logger.info("SAM2 loaded.")

    return model, predictor


# ─────────────────────────────────────────────
# 4. RECURSIVE IMAGE COLLECTION
# ─────────────────────────────────────────────
def collect_images_recursively(folder: Path) -> list[Path]:
    """
    Walk all subdirectories under `folder` and return
    paths of every image file found at any depth.
    """
    image_paths = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix in IMAGE_EXTENSIONS:
                image_paths.append(Path(root) / fname)
    return sorted(image_paths)


# ─────────────────────────────────────────────
# 5. POINT SAMPLING
# ─────────────────────────────────────────────
def sample_points_within_contour(contour, num_points: int) -> list[tuple]:
    """Randomly sample `num_points` pixel coords inside a contour."""
    rect = cv2.boundingRect(contour)
    x0, y0, w, h = rect
    if w == 0 or h == 0:
        return []

    local_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([[x0, y0]])
    cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(local_mask == 255)
    if len(xs) == 0:
        return []
    if len(xs) <= num_points:
        return [(int(xs[i]) + x0, int(ys[i]) + y0) for i in range(len(xs))]

    indices = random.sample(range(len(xs)), num_points)
    return [(int(xs[i]) + x0, int(ys[i]) + y0) for i in indices]


# ─────────────────────────────────────────────
# 6. BATCH PROCESSING
# ─────────────────────────────────────────────
def process_batch(
    image_paths: list[Path],
    target_class: int,
    model,
    predictor,
    cam,
    save_root: Path,
):
    """
    Run the full GradCAM → SAM2 pipeline on a batch of images.
    Masks are saved mirroring the input folder structure under save_root.
    """
    # --- Load & preprocess batch ---
    batch_tensors = []
    originals = []   # list of (path, PIL image)

    for img_path in image_paths:
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {img_path}: {e}")
            continue
        originals.append((img_path, pil_img))
        batch_tensors.append(inference_transform(pil_img).unsqueeze(0))

    if not originals:
        return

    batch_tensor = torch.cat(batch_tensors).to(DEVICE)

    # --- GradCAM heatmaps for whole batch ---
    targets = [ClassifierOutputTarget(target_class)] * len(originals)
    grayscale_cams = cam(input_tensor=batch_tensor, targets=targets)
    # grayscale_cams shape: (B, H, W) values in [0, 1]

    # --- Per-image SAM2 refinement ---
    for idx, (img_path, pil_img) in enumerate(originals):
        orig_w, orig_h = pil_img.size   # PIL: (width, height)

        # Resize CAM to original image dimensions
        cam_resized = cv2.resize(
            grayscale_cams[idx],
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Binarize
        _, binary = cv2.threshold(
            np.uint8(255 * cam_resized),
            THRESHOLD_VALUE, 255,
            cv2.THRESH_BINARY
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sample prompt points from contours
        all_pts, all_labels = [], []
        for contour in contours:
            pts = sample_points_within_contour(contour, NUM_SAMPLED_POINTS)
            all_pts.extend(pts)
            all_labels.extend([1] * len(pts))

        if not all_pts:
            logger.info(f"No activation contours for {img_path.name}, skipping.")
            continue

        # --- SAM2 prediction ---
        img_np = np.array(pil_img)          # (H, W, 3) uint8
        predictor.set_image(img_np)

        pts_arr    = np.array(all_pts, dtype=np.float32)   # (N, 2)
        labels_arr = np.array(all_labels, dtype=np.int32)  # (N,)

        # Pass 1: multi-mask to find best candidate
        masks, scores, logits = predictor.predict(
            point_coords=pts_arr,
            point_labels=labels_arr,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))

        # Pass 2: refine using best logit
        refined_masks, _, _ = predictor.predict(
            point_coords=pts_arr,
            point_labels=labels_arr,
            mask_input=logits[best_idx][None, :, :],
            multimask_output=False,
        )
        final_mask = refined_masks[0]   # (H, W) bool

        # Build output mask: class_id for plant, BACKGROUND_CLASS elsewhere
        mask_uint8 = np.where(final_mask, target_class, BACKGROUND_CLASS).astype(np.uint8)

        # Mirror input folder structure in output directory
        rel_path  = img_path.relative_to(BASE_DIR)
        save_dir  = save_root / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        out_name  = f"mask_{img_path.stem}.png"
        out_path  = save_dir / out_name
        cv2.imwrite(str(out_path), mask_uint8)
        logger.info(f"Saved: {out_path}")

    torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
def main():
    base_path  = Path(BASE_DIR)
    save_root  = Path(OUTPUT_MASK_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    # Load models ONCE
    model, predictor = initialize_models()
    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    # Build class map from sorted top-level species folders
    species_folders = sorted([
        d for d in base_path.iterdir()
        if d.is_dir()
    ])

    if not species_folders:
        logger.error(f"No species folders found in {BASE_DIR}")
        return

    logger.info(f"Found {len(species_folders)} species folders:")
    for idx, folder in enumerate(species_folders):
        logger.info(f"  [{idx}] {folder.name}")

    # Process each species
    for class_id, species_folder in enumerate(species_folders):
        logger.info(f"\n{'='*55}")
        logger.info(f"Processing: {species_folder.name}  (class_id={class_id})")

        # RECURSIVE image collection ← key fix
        image_paths = collect_images_recursively(species_folder)
        logger.info(f"  Found {len(image_paths)} images (recursively)")

        if not image_paths:
            logger.warning(f"  No images found, skipping.")
            continue

        # Process in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i : i + BATCH_SIZE]
            logger.info(f"  Batch {i//BATCH_SIZE + 1} / {(len(image_paths)-1)//BATCH_SIZE + 1}")
            process_batch(
                image_paths=batch,
                target_class=class_id,
                model=model,
                predictor=predictor,
                cam=cam,
                save_root=save_root,
            )

    logger.info("\n" + "="*55)
    logger.info("All species processed. Masks saved to:")
    logger.info(f"  {save_root.resolve()}")


if __name__ == "__main__":
    main()