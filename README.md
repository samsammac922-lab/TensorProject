# TensorFlow Image Classifier

A minimal yet production-ready CNN image-classification project that loads its training data from a **`data_training.tar.gz`** archive.

> ⚠️ **Important:** The `data_training.tar.gz` file **must be manually decompressed** before running any script. The project reads data from the extracted `data/` directory — it will not work on the compressed archive directly. See [Data Format](#data-format) for exact decompression steps.

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
├── data_training.tar.gz   ← compressed archive (must be extracted first!)
├── data/                  ← extracted data directory (required to train)
│   ├── train/
│   └── val/               (optional)
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

### 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU support (optional)**
> TensorFlow 2.15+ ships with built-in GPU support on Linux/Windows.  
> Make sure your CUDA and cuDNN versions match the [TensorFlow compatibility matrix](https://www.tensorflow.org/install/pip#software_requirements).

### 4 — Verify the installation

```bash
python - <<'EOF'
import tensorflow as tf
print("TF version :", tf.__version__)
print("GPU devices :", tf.config.list_physical_devices("GPU"))
EOF
```

---

## Data Format

### ⚠️ Step 1 — Decompress the archive (required)

**The archive must be fully extracted before running any script.** The project reads images from the `data/` directory on disk — it does not read from the compressed file at runtime.

Place `data_training.tar.gz` in the project root, then extract it:

```bash
# macOS / Linux
tar -xzf data_training.tar.gz

# Windows (PowerShell — built-in, no extra tools needed)
tar -xzf data_training.tar.gz

# Windows (Git Bash / WSL)
tar -xzf data_training.tar.gz

# Windows (7-Zip GUI alternative)
# Right-click data_training.tar.gz → 7-Zip → Extract Here
```

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

### Don't have real data yet?

Generate a synthetic archive for quick testing, then extract it:

```bash
python create_sample_data.py   # writes data_training.tar.gz (~2 MB)
tar -xzf data_training.tar.gz  # extract before training
```

---

## Training

> Before training, make sure you have extracted `data_training.tar.gz` as described in [Data Format](#data-format).

```bash
tar -xzf data_training.tar.gz   # decompress first
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

1. **Extraction** — `data_training.tar.gz` is unpacked into `data/` (skipped on subsequent runs).
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
| `FileNotFoundError: data/train` | The archive has not been extracted yet. Run `tar -xzf data_training.tar.gz` first. |
| `FileNotFoundError: data_training.tar.gz` | Place the archive in the project root, then extract it. |
| `No images found` | Verify the archive structure matches the layout shown in [Data Format](#data-format). |
| Out-of-memory (OOM) on GPU | Reduce `BATCH_SIZE` (e.g. 16 or 8). |
| Training accuracy stuck near random | Check class balance; increase `EPOCHS`; lower `LEARNING_RATE`. |
| `ModuleNotFoundError: tensorflow` | Activate the virtual environment: `source .venv/bin/activate`. |
| CUDA errors on GPU | Verify CUDA/cuDNN versions with the [TF compatibility matrix](https://www.tensorflow.org/install/pip#software_requirements). |
