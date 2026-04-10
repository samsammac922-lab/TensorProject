# TensorFlow Image Classifier

A minimal yet production-ready CNN image-classification project that loads its training data from a **`data_training.tar.gz`** archive.

> The training script extracts a valid `data_training.tar.gz` archive automatically. The archive can contain either an image-folder dataset or a raw-record binary plus metadata.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Data Format](#data-format)
5. [Training](#training)
6. [Inference](#inference)
7. [TensorBoard](#tensorboard)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
.
├── data_training.tar.gz   ← compressed archive (train.py extracts automatically)
├── data/                  ← extracted image dataset (optional format)
│   ├── train/
│   └── val/               (optional)
├── data_training.bin      ← extracted raw-record binary dataset (optional format)
├── data_training.meta.json← metadata for the raw binary dataset
├── train.py               ← main training script
├── predict.py             ← inference / prediction script
├── create_sample_data.py  ← helper: generate synthetic test data
├── requirements.txt
└── README.md
```

After training, the following are created automatically:

```
├── data/                  ← extracted archive
├── checkpoints/
│   └── best_model.keras   ← best checkpoint (highest val_accuracy)
├── logs/                  ← TensorBoard event files
└── model_final.keras      ← final saved model
```

---

## Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python      | 3.10            |
| pip         | 23+             |

A CUDA-capable GPU is optional but strongly recommended for faster training.

---

## Environment Setup

### 1 — Clone / download the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2 — Recommended: one-command setup

```bash
./setup.sh
```

`setup.sh` does the following:

- Uses a local virtual environment when `python3` is version `3.10` through `3.13`.
- Falls back to building a `podman` image when the local Python version is not supported by TensorFlow.

### 3 — Create a virtual environment manually

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU support (optional)**
> TensorFlow 2.15+ ships with built-in GPU support on Linux/Windows.  
> Make sure your CUDA and cuDNN versions match the [TensorFlow compatibility matrix](https://www.tensorflow.org/install/pip#software_requirements).

### 5 — Verify the installation

```bash
python - <<'EOF'
import tensorflow as tf
print("TF version :", tf.__version__)
print("GPU devices :", tf.config.list_physical_devices("GPU"))
EOF
```

---

## Data Format

### Automatic extraction

When you run `python train.py`, the script inspects `data_training.tar.gz` and extracts it automatically if needed.

After extraction you should see a `data/` folder in the project root:

```
.
├── data/               ← extracted content (required)
│   ├── train/
│   └── val/            (optional)
├── data_training.tar.gz
└── ...
```

> Do **not** rename or move the `data/` folder — the training script expects it at that exact path.
>
> If extraction fails, the archive layout is wrong. A valid archive must contain one of:
>
> - `data/train/<class_name>/<image files>`
> - `train/<class_name>/<image files>`
> - `data_training.bin` and `data_training.meta.json`

---

### Expected directory layout

```
data/
├── train/
│   ├── class_a/
│   │   ├── img_001.jpg
│   │   └── …
│   └── class_b/
│       └── …
└── val/              ← optional; omit for automatic 80/20 split
    ├── class_a/
    └── class_b/
```

- Supported image formats: **JPEG, PNG, BMP, GIF** (static frames only).
- Sub-folder names become the class labels automatically.
- If no `val/` folder is present, the script applies an 80 / 20 train-validation split automatically.

### Raw binary layout

For raw-record datasets, include `data_training.bin` and `data_training.meta.json` in the archive root.

Example metadata:

```json
{
  "feature_shape": [128, 128, 3],
  "feature_dtype": "uint8",
  "label_dtype": "uint8",
  "label_bytes": 1,
  "record_bytes": 49153,
  "class_names": ["cat", "dog"],
  "validation_split": 0.2
}
```

Each record is interpreted as:

- label bytes first
- raw feature bytes after that

By default the loader assumes labels are sparse integers and features are raw `uint8` pixels reshaped to `feature_shape`.

### Don't have real data yet?

Generate a synthetic archive for quick testing without overwriting the original training archive:

```bash
python create_sample_data.py   # writes sample_data_training.tar.gz
python create_sample_data.py --format raw-bin
python train.py --archive sample_data_training.tar.gz
```

---

## Training

```bash
python train.py
```

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--archive` | `data_training.tar.gz` | Path to the data archive |
| `--epochs`  | `20`                   | Maximum training epochs |

```bash
# Example: custom archive path and more epochs
python train.py --archive /datasets/my_data.tar.gz --epochs 50
```

### What happens during training

1. **Extraction** — `data_training.tar.gz` is unpacked into `data/` automatically (skipped on subsequent runs).
2. **Dataset construction** — `tf.data` pipelines with caching and prefetching.
3. **Augmentation** — random horizontal flip, rotation ±10 °, zoom ±10 %.
4. **Model** — 3-block CNN with BatchNorm, followed by GlobalAveragePooling + Dropout.
5. **Callbacks**
   - `ModelCheckpoint` — saves `checkpoints/best_model.keras` whenever `val_accuracy` improves.
   - `EarlyStopping` — halts training after 5 epochs without improvement in `val_loss`.
   - `ReduceLROnPlateau` — halves the learning rate after 3 stagnant epochs.
   - `TensorBoard` — writes logs to `logs/`.

---

## Inference

```bash
python predict.py --model model_final.keras --image photo.jpg
```

Classify multiple images at once:

```bash
python predict.py \
  --model checkpoints/best_model.keras \
  --image img1.jpg img2.png img3.jpeg \
  --classes '["cat", "dog"]'
```

`--classes` is optional. When omitted, predictions are labelled with numeric indices.

---

## TensorBoard

```bash
tensorboard --logdir logs
```

Open [http://localhost:6006](http://localhost:6006) in your browser to monitor loss, accuracy, and the learning-rate schedule in real time.

---

## Configuration

Key hyper-parameters live at the top of `train.py` as module-level constants:

```python
IMG_SIZE      = (128, 128)   # resize target for all images
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-3
```

Edit them directly, or extend `parse_args()` to expose them as CLI flags.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: data/train` | The archive layout is invalid. Rebuild `data_training.tar.gz` so it contains `data/train/...`, `train/...`, or a valid raw binary pair. |
| `data_training.meta.json is missing` | Raw binary archives require metadata describing record size, feature shape, and class names. |
| `FileNotFoundError: data_training.tar.gz` | Place the archive in the project root. `train.py` will extract it automatically. |
| `No images found` | Verify the archive structure matches the layout shown in [Data Format](#data-format). |
| Out-of-memory (OOM) on GPU | Reduce `BATCH_SIZE` (e.g. 16 or 8). |
| Training accuracy stuck near random | Check class balance; increase `EPOCHS`; lower `LEARNING_RATE`. |
| `ModuleNotFoundError: tensorflow` | Activate the virtual environment: `source .venv/bin/activate`. |
| TensorFlow install fails on Python 3.14+ | Run `./setup.sh` to use the `podman` fallback, or install Python 3.10-3.13 locally. |
| CUDA errors on GPU | Verify CUDA/cuDNN versions with the [TF compatibility matrix](https://www.tensorflow.org/install/pip#software_requirements). |
