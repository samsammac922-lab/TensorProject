"""
TensorFlow Image Classification Training Script
Loads training data from data_training.tar.gz and trains a CNN model.
"""

import os
import tarfile
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_ARCHIVE   = "data_training.tar.gz"
EXTRACT_DIR    = "data"
IMG_SIZE       = (128, 128)
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 1e-3
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"


# ── Data Loading ──────────────────────────────────────────────────────────────

def extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract data_training.tar.gz if not already extracted."""
    if Path(extract_to).exists():
        log.info("Data directory '%s' already exists — skipping extraction.", extract_to)
        return

    if not Path(archive_path).exists():
        raise FileNotFoundError(
            f"Training archive not found: '{archive_path}'.\n"
            "Please place data_training.tar.gz in the project root."
        )

    log.info("Extracting '%s' → '%s' ...", archive_path, extract_to)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    log.info("Extraction complete.")


def build_datasets(data_dir: str):
    """
    Build train / validation tf.data.Dataset objects.

    Expected directory layout inside the archive:
        data/
          train/
            class_a/  *.jpg | *.png
            class_b/  ...
          val/           (optional — auto-split used when absent)
            class_a/
            class_b/
    """
    train_dir = Path(data_dir) / "train"
    val_dir   = Path(data_dir) / "val"

    common_kwargs = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    if val_dir.exists():
        log.info("Loading train split from '%s'", train_dir)
        train_ds = keras.utils.image_dataset_from_directory(
            str(train_dir), **common_kwargs
        )
        log.info("Loading val split from '%s'", val_dir)
        val_ds = keras.utils.image_dataset_from_directory(
            str(val_dir), **common_kwargs
        )
    else:
        log.info(
            "No 'val/' folder found — using 80/20 split from '%s'", train_dir
        )
        train_ds = keras.utils.image_dataset_from_directory(
            str(train_dir), validation_split=0.2, subset="training", **common_kwargs
        )
        val_ds = keras.utils.image_dataset_from_directory(
            str(train_dir), validation_split=0.2, subset="validation", **common_kwargs
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    log.info("Classes (%d): %s", num_classes, class_names)

    # Performance optimisation
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    """Build a small CNN with data augmentation head."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_classifier")
    return model


def compile_model(model: keras.Model, num_classes: int) -> None:
    loss = (
        "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    # 1. Extract data
    extract_archive(args.archive, EXTRACT_DIR)

    # 2. Datasets
    train_ds, val_ds, class_names = build_datasets(EXTRACT_DIR)
    num_classes = len(class_names)

    # 3. Model
    model = build_model(num_classes)
    compile_model(model, num_classes)
    model.summary(print_fn=log.info)

    # 4. Callbacks
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    ]

    # 5. Fit
    log.info("Starting training for up to %d epochs …", args.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 6. Save final model
    model.save("model_final.keras")
    log.info("Training complete. Model saved to 'model_final.keras'.")
    log.info(
        "Best val_accuracy: %.4f",
        max(history.history.get("val_accuracy", [0])),
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on data_training.tar.gz")
    parser.add_argument(
        "--archive",
        default=DATA_ARCHIVE,
        help="Path to the training data archive (default: data_training.tar.gz)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum number of training epochs (default: {EPOCHS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
