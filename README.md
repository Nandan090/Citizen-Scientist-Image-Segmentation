# Citizen-Scientist Image Segmentation (Grad-CAM + SAM2 → YOLOv8-Seg)

This repository implements an end-to-end segmentation project for invasive plant species using a **weakly-supervised mask generation pipeline** and a **YOLOv8 segmentation model**.

**YOLO is a required final stage in this project.**  
We first generate pixel masks automatically (without manual annotation) and then **train YOLOv8-seg** on those masks for a deployable segmentation model.

---

## What this project does (end-to-end)

1. **Download labeled images** from iNaturalist into class folders (image-level supervision).
2. **Train an image classifier** (EfficientNetV2-L).
3. Use **Grad-CAM** on the classifier to find plant regions in each image.
4. Use **SAM2** to refine those regions into high-quality **pixel masks**.
5. Convert masks into **YOLOv8 segmentation polygon labels**.
6. **Train YOLOv8-seg** on the generated dataset.

---

## Repository Files

- `download.py`  
  Downloads iNaturalist images for a list of species and organizes them into class folders.

- `pipeline.py`  
  End-to-end script: trains EfficientNetV2-L (if needed) and generates masks using Grad-CAM + SAM2.

- `SAM.py`  
  Mask-generation-only script (batch processing). Uses trained classifier + Grad-CAM + SAM2 to produce masks.

- `yolo.py`  
  Converts SAM2 masks (`.png`) into YOLOv8 segmentation labels (`.txt` polygons), creates `yolo_dataset/`, writes `data.yaml`, and prints YOLO training commands.

- `train_yolo.py`  
  Trains YOLOv8 segmentation model using Ultralytics (edit the `data=` path before running).

---

## Expected Folder Structure

All paths are relative to your **current working directory** (the scripts use `os.getcwd()`):

```
./images/
  Fallopia_japonica/
    *.jpg
    (subfolders allowed)
  Lupinus_polyphyllus/
    *.jpg
./checkpoints/
  best_model.pth
./masks/
  Fallopia_japonica/
    mask_<image_stem>.png
  Lupinus_polyphyllus/
    mask_<image_stem>.png
./yolo_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

### Mask format
Saved masks are single-channel PNGs where:
- foreground pixels = `class_id` (integer)
- background pixels = `255`

---

## Requirements

Core dependencies used across the pipeline:

- `torch`, `torchvision`
- `opencv-python`
- `numpy`, `Pillow`
- `pytorch-grad-cam`
- `sam2` + SAM2 checkpoints (must be installed/available in your environment)
- `ultralytics` (for YOLOv8-seg training)
- `requests`, `tqdm` (for downloads)

Example install:

```bash
pip install torch torchvision
pip install opencv-python pillow numpy requests tqdm
pip install pytorch-grad-cam
pip install ultralytics
```

> SAM2 installation/checkpoints vary by environment. Ensure the SAM2 checkpoint path(s) configured in the scripts point to real files.

---

# Step-by-step: Run the full project

## Step 1 — Download iNaturalist images (`download.py`)

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
`download.py` currently downloads into `OUTPUT_DIR = os.getcwd()`.

The next steps expect images under `./images/`, so you should do ONE of the following:

- Run `download.py` from inside the `images/` directory, **or**
- Change `OUTPUT_DIR` in `download.py` to:

```python
OUTPUT_DIR = f"{os.getcwd()}/images"
```

---

## Step 2 — Train classifier + generate SAM2 masks (`pipeline.py`)

This script trains EfficientNetV2-L (if `./checkpoints/best_model.pth` is missing) and then generates masks.

Run:

```bash
python pipeline.py
```

Outputs:
- classifier checkpoint: `./checkpoints/best_model.pth`
- segmentation masks: `./masks/.../mask_*.png`

### Key settings to review in `pipeline.py`
- `DATA_ROOT = "./images"`
- `OUTPUT_MASK_ROOT = "./masks"`
- `NUM_CLASSES = 2` (**must match the number of class folders used to train**)
- `THRESHOLD_VALUE`, `NUM_POINTS`
- SAM2 config/checkpoint paths: `SAM2_CONFIG`, `SAM2_CHECKPOINT`

> If you have more than 2 species folders, update `NUM_CLASSES` accordingly and retrain.

---

## Step 3 — Convert masks to YOLOv8 segmentation dataset (`yolo.py`)

Run:

```bash
python yolo.py
```

This creates a YOLO dataset at:

- `./yolo_dataset/images/train|val`
- `./yolo_dataset/labels/train|val`
- `./yolo_dataset/data.yaml`

What happens internally:
- Reads `./masks/` to build class IDs from top-level mask folder names
- Finds contours for each object in each mask
- Writes YOLO polygon labels in the format:

```
class_id x1 y1 x2 y2 ... xN yN
```

(all coordinates normalized to 0–1)

---

## Step 4 — Train YOLOv8 segmentation model 

### Option A: Train via YOLO CLI
After `yolo.py` generates `yolo_dataset/data.yaml`:

```bash
yolo segment train data=yolo_dataset/data.yaml model=yolov8l-seg.pt epochs=60 imgsz=512
```

### Option B: Train via `train_yolo.py` (repo script)
`train_yolo.py` runs Ultralytics training, but you must edit the `data=` path first.

Current file uses a hardcoded path:

```python
data='/home/jovyan/work/Final Project/yolo_dataset/data.yaml'
```

Change it to:

```python
data='yolo_dataset/data.yaml'
```

Then run:

```bash
python train_yolo.py
```

Training outputs typically go to:
- `runs/segment/train/`
- best weights: `runs/segment/train/weights/best.pt`

---

## How the model pipeline works (conceptual)

- **EfficientNetV2-L** learns to classify species from folder labels.
- **Grad-CAM** identifies where the classifier “looks” for that species in the image (coarse heatmap).
- **Threshold + contours** convert heatmap to prompt regions.
- **SAM2** refines prompts into accurate pixel masks.
- Masks are converted into **YOLO polygon annotations**.
- **YOLOv8-seg** is trained on the generated dataset for fast inference.

---

## Troubleshooting

- **No contours / no masks saved**
  - Lower `THRESHOLD_VALUE` (e.g., 150 → 100)
  - Increase `NUM_POINTS` (more SAM2 prompts)
  - Verify classifier checkpoint quality (`best_model.pth`)

- **SAM2 checkpoint/config errors**
  - Update `SAM2_CHECKPOINT` / `SAM2_CONFIG` paths to match your environment.
  - Confirm `sam2` is importable in Python.

- **YOLO dataset has many “skipped” images**
  - Masks may be empty/noisy.
  - Lower `MIN_CONTOUR_AREA` in `yolo.py` or adjust Grad-CAM thresholding.

- **More than 2 species**
  - Update `NUM_CLASSES` and retrain classifier so class IDs align with folders.

---

## Notes on Licensing / Data
- iNaturalist images have per-image licenses. `download.py` filters to open licenses by default (`cc-by, cc-by-nc, cc0`).
- Ensure you follow attribution requirements when distributing or publishing results.

---
