# Citizen-Scientist Image Segmentation (Weakly Supervised)

This repository builds a **weakly-supervised segmentation pipeline** for invasive plant species using:

- **iNaturalist** images (image-level labels from folder names)
- An **EfficientNetV2-L** image classifier
- **Grad-CAM** to localize the class-relevant regions
- **SAM2** (Segment Anything Model 2) to generate high-quality pixel masks
- Optional: conversion of masks to **YOLOv8-seg** format and training a deployable segmentation model

> Core idea: **classification → localization (Grad-CAM) → segmentation (SAM2)**  
> This produces pixel masks without manually annotating masks.

---

## Repository Contents

- `download.py`  
  Downloads iNaturalist images for a list of species and saves them into class folders.

- `pipeline.py`  
  End-to-end script: trains EfficientNetV2-L (if needed) + generates masks using Grad-CAM + SAM2.

- `SAM.py`  
  Mask-generation-focused script (batch processing). Uses a trained EfficientNetV2-L + Grad-CAM + SAM2.

- `yolo.py`  
  Converts generated PNG masks into YOLOv8 segmentation polygon labels, builds a YOLO dataset folder, and prints training commands.

- `train_yolo.py`  
  Minimal Ultralytics YOLOv8-seg training launcher (you must edit the `data=` path).

---

## High-Level Workflow (Step-by-step)

1. **Download images** from iNaturalist into per-species folders  
2. **Train a classifier** (EfficientNetV2-L) to recognize the species (image-level supervision)  
3. **Generate masks** using Grad-CAM → contour points → SAM2 refinement  
4. (Optional) **Convert masks to YOLOv8-seg dataset**  
5. (Optional) **Train YOLOv8-seg** for fast inference/deployment

---

## Folder Structure

Expected inputs/outputs (relative to your current working directory):

```
./images/
  Fallopia_japonica/
    img_001.jpg
    ...
  Lupinus_polyphyllus/
    ...
./checkpoints/
  best_model.pth
./masks/
  Fallopia_japonica/
    mask_img_001.png
    ...
./yolo_dataset/                 # created by yolo.py
  images/train, images/val
  labels/train, labels/val
  data.yaml
```

Mask format:
- Foreground pixels = `class_id` (integer)
- Background pixels = `255`

---

## Requirements

Python packages used by the scripts include:

- `torch`, `torchvision`
- `opencv-python` (`cv2`)
- `numpy`, `Pillow`
- `pytorch-grad-cam`
- `sam2` (SAM2 package/checkpoints)
- For YOLO training: `ultralytics`
- For download: `requests`, `tqdm`

### Install (example)

```bash
pip install torch torchvision
pip install opencv-python pillow numpy tqdm requests
pip install pytorch-grad-cam
pip install ultralytics
```

> SAM2 installation depends on your environment. Ensure the `sam2` Python package is available and the SAM2 checkpoint path(s) in scripts point to real files.

---

## Step 1 — Download iNaturalist Images (`download.py`)

1) Edit the species list in `download.py`:

```python
SPECIES_LIST = [
    "Fallopia japonica",
    "Lupinus polyphyllus",
]
```

2) Run:

```bash
python download.py
```

### Important note about output location
`download.py` currently saves into `OUTPUT_DIR = os.getcwd()`.

The rest of the pipeline expects images under `./images/`.

So either:
- run `download.py` from inside an `images/` directory, **or**
- modify `download.py` so `OUTPUT_DIR` becomes `f"{os.getcwd()}/images"`.

---

## Step 2 — Train Classifier + Generate Masks (`pipeline.py`)

`pipeline.py` is the “all-in-one” script:

- If `./checkpoints/best_model.pth` does not exist: it trains EfficientNetV2-L.
- Then it generates masks recursively under `./images/` and saves them under `./masks/`.

Run:

```bash
python pipeline.py
```

### Key configuration (edit inside `pipeline.py`)
- `DATA_ROOT = "./images"`
- `CHECKPOINT_DIR = "./checkpoints"`
- `OUTPUT_MASK_ROOT = "./masks"`
- `NUM_CLASSES = 2` (see warning below)
- SAM2 paths (`SAM2_CHECKPOINT`, `SAM2_CONFIG`)

### Warning: `NUM_CLASSES`
`NUM_CLASSES` is hardcoded (default `2`).  
If you have more than two species folders in `./images`, update `NUM_CLASSES` and ensure your training/checkpoint match the number of classes.

---

## Step 3 — Generate Masks Only (Batch Mode) (`SAM.py`)

If you already trained the classifier and only want to produce masks:

```bash
python SAM.py
```

It will:
- load `./checkpoints/best_model.pth`
- run Grad-CAM in batches
- prompt SAM2 using sampled points
- write masks into `./masks/`

---

## Step 4 (Optional) — Build YOLOv8-Seg Dataset (`yolo.py`)

Convert masks into YOLO polygon labels and build a YOLO-ready dataset:

```bash
python yolo.py
```

Outputs:
- `./yolo_dataset/images/train|val`
- `./yolo_dataset/labels/train|val`
- `./yolo_dataset/data.yaml`

---

## Step 5 (Optional) — Train YOLOv8 Segmentation

### Option A: Use CLI (recommended)
After running `yolo.py`, train:

```bash
yolo segment train data=yolo_dataset/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=512
```

### Option B: Use `train_yolo.py`
Edit the `data=` path in `train_yolo.py` (it is currently hardcoded to a local machine path), then run:

```bash
python train_yolo.py
```

---

## How the Weakly-Supervised Mask Generation Works

Per image:

1. **EfficientNetV2-L** predicts class features.
2. **Grad-CAM** produces a heatmap showing where the classifier “looked” for a target class.
3. Heatmap is **thresholded** and converted to **contours**.
4. A few **positive points** are sampled inside contours.
5. **SAM2** uses these points as prompts to generate a precise pixel mask.
6. Mask is saved as PNG (class id for foreground, 255 for background).

---

## Troubleshooting

- **No masks are produced / “No contours found”**  
  Try lowering `THRESHOLD_VALUE` (e.g., 150 → 100) or increasing `NUM_POINTS`.

- **SAM2 checkpoint not found**  
  Fix `SAM2_CHECKPOINT` and `SAM2_CONFIG` values in `pipeline.py` / `SAM.py` to match your installation.

- **More than 2 species**  
  Update `NUM_CLASSES` and retrain your classifier. Checkpoints must match the class count.

- **YOLO training script fails**  
  `train_yolo.py` uses a hardcoded `data=` path. Change it to your generated `yolo_dataset/data.yaml`.

---

## License / Data Notes

- iNaturalist images are subject to their individual licenses. This pipeline filters downloads to open licenses (`cc-by, cc-by-nc, cc0` by default).
- Ensure you comply with attribution requirements when using downloaded images.

---

## Acknowledgements

- iNaturalist for citizen-science biodiversity observations
- EfficientNetV2 (TorchVision)
- Grad-CAM (`pytorch-grad-cam`)
- SAM2 (Segment Anything Model 2)
- Ultralytics YOLOv8 segmentation
