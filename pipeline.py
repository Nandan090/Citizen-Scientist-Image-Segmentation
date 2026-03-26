import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import random
import logging
import copy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# FIX 1: Use correct SAM2 import API (not SAM1's sam_model_registry)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os 

# --- 1. CONFIGURATION ---
OUTPUT_DIR       = os.getcwd()
DATA_ROOT        = f"{OUTPUT_DIR}/images"
CHECKPOINT_DIR   = f"{OUTPUT_DIR}/checkpoints"
OUTPUT_MASK_ROOT = f"{OUTPUT_DIR}/masks"
SAM2_CHECKPOINT  = "/opt/conda/lib/python3.12/site-packages/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG      = "sam2.1_hiera_l"

IMG_SIZE = 512
BATCH_SIZE = 8
NUM_CLASSES = 2
NUM_EPOCHS = 60
SAMPLES_PER_CLASS = 8000
THRESHOLD_VALUE = 150
NUM_POINTS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 2. DATA PREPARATION ---
# FIX 2: Separate train and inference transforms.
# Training transform (with augmentation):
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Inference transform (deterministic — no random crop/flip):
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def get_balanced_loaders(data_dir):
    dataset = ImageFolder(root=data_dir, transform=train_transform)
    targets = np.array(dataset.targets)
    balanced_indices = []

    for i in range(len(dataset.classes)):
        class_indices = np.where(targets == i)[0]
        replace = len(class_indices) < SAMPLES_PER_CLASS
        if replace:
            logger.warning(
                f"Class '{dataset.classes[i]}' has only {len(class_indices)} samples; "
                f"oversampling to {SAMPLES_PER_CLASS}."
            )
        sampled = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=replace)
        balanced_indices.extend(sampled)

    np.random.shuffle(balanced_indices)
    split = int(0.8 * len(balanced_indices))
    train_idx, val_idx = balanced_indices[:split], balanced_indices[split:]

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(train_idx), num_workers=4)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            sampler=SubsetRandomSampler(val_idx), num_workers=4)
    return train_loader, val_loader, dataset.classes

# --- 3. MODEL ARCHITECTURE ---
def build_model(weights_path=None):
    model = models.efficientnet_v2_l(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint.items())
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded weights from {weights_path}")

    return model.to(DEVICE)

# --- 4. TRAINING LOOP ---
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2,
                           steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)

    best_val_acc = 0.0
    best_weights_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # -- Training --
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # -- Validation --
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return best_weights_path

# --- 5. SAM2 PREDICTOR ---
# FIX 1 (continued): Use build_sam2 + SAM2ImagePredictor instead of SAM1 API
def get_sam_predictor():
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

# --- 6. POINT SAMPLING ---
def sample_points(contour, num_points):
    rect = cv2.boundingRect(contour)
    mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
    shifted_contour = contour - np.array([[rect[0], rect[1]]])
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return []
    if len(xs) < num_points:
        return [(x + rect[0], y + rect[1]) for x, y in zip(xs, ys)]
    idx = random.sample(range(len(xs)), num_points)
    return [(xs[i] + rect[0], ys[i] + rect[1]) for i in idx]

# --- 7. CLASS-ID MAPPING ---
# FIX 3: Build a real class_name -> class_id mapping from the dataset
def build_class_map(data_dir):
    dataset = ImageFolder(root=data_dir)
    # class_to_idx is e.g. {'class_a': 0, 'class_b': 1, ...}
    return dataset.class_to_idx

# --- 8. EXECUTION ENGINE ---
def run_full_pipeline():
    # Phase 1: Training
    logger.info("Starting Training Phase...")
    train_loader, val_loader, class_names = get_balanced_loaders(DATA_ROOT)
    model = build_model()
    best_weights_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_weights_path):
        best_weights_path = train_model(model, train_loader, val_loader)
    else:
        logger.info(f"Skipping training, loading existing weights from {best_weights_path}")

    # Reload best weights for inference
    model = build_model(weights_path=best_weights_path)
    # FIX 4: Set model to eval mode before GradCAM / inference
    model.eval()

    # Phase 2: Recursive Segmentation
    logger.info("Starting Recursive Segmentation Phase...")
    predictor = get_sam_predictor()

    # FIX 5: Use the last Conv block as the GradCAM target layer (correct layer reference)
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # FIX 3: Build class_id map from folder names
    class_to_idx = build_class_map(DATA_ROOT)
    logger.info(f"Class mapping: {class_to_idx}")

    for root, _, files in os.walk(DATA_ROOT):
        valid_imgs = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not valid_imgs:
            continue

        # FIX 3: Derive class_id from the top-level folder name
        rel_path = os.path.relpath(root, DATA_ROOT)
        top_folder = rel_path.split(os.sep)[0]
        class_id = class_to_idx.get(top_folder, None)
        if class_id is None:
            logger.warning(f"Skipping '{root}': folder '{top_folder}' not in class map.")
            continue

        save_dir = os.path.join(OUTPUT_MASK_ROOT, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in valid_imgs:
            img_path = os.path.join(root, img_name)
            raw_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = raw_img.size

            # FIX 2: Use inference_transform (no random augmentation)
            input_tensor = inference_transform(raw_img).unsqueeze(0).to(DEVICE)

            # Grad-CAM Heatmap
            # FIX 4: GradCAM handles grad enabling internally, but model must be in eval mode
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(class_id)]
            )[0]  # shape: (IMG_SIZE, IMG_SIZE)

            # FIX: Resize CAM back to the ORIGINAL image dimensions (not IMG_SIZE x IMG_SIZE)
            grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))

            # Threshold to binary mask
            cam_uint8 = np.uint8(255 * grayscale_cam_resized)
            _, binary = cv2.threshold(cam_uint8, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_pts = []
            for c in contours:
                all_pts.extend(sample_points(c, NUM_POINTS))

            if all_pts:
                img_np = np.array(raw_img)
                predictor.set_image(img_np)

                pts_array = np.array(all_pts)         # shape: (N, 2)
                labels_array = np.ones(len(all_pts), dtype=np.int32)

                # First pass: get best mask index from multimask output
                masks, scores, logits = predictor.predict(
                    point_coords=pts_array,
                    point_labels=labels_array,
                    multimask_output=True
                )
                best_idx = np.argmax(scores)

                # Second pass: refine using best logit
                final_masks, _, _ = predictor.predict(
                    point_coords=pts_array,
                    point_labels=labels_array,
                    mask_input=logits[best_idx][None, :, :],
                    multimask_output=False
                )

                # FIX: squeeze safely — final_masks shape is (1, H, W)
                final_mask = final_masks[0]  # shape: (H, W), dtype bool

                # Save: foreground pixels get class_id, background gets 255 (common ignore label)
                mask_uint8 = np.where(final_mask, class_id, 255).astype(np.uint8)
                out_path = os.path.join(save_dir, f"mask_{os.path.splitext(img_name)[0]}.png")
                cv2.imwrite(out_path, mask_uint8)
                logger.info(f"Saved mask: {out_path}")
            else:
                logger.warning(f"No contours found for {img_path}, skipping SAM2 step.")

if __name__ == "__main__":
    run_full_pipeline()
