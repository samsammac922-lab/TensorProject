"""
TensorFlow Image Classification Training Script
Loads training data from data_training.tar.gz and trains a CNN model.
"""

import os
import tarfile
import argparse
import json
import math
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
BIN_DATA_FILE  = "data_training.bin"
BIN_META_FILE  = "data_training.meta.json"
IMG_SIZE       = (128, 128)
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 1e-3
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"


# ── Data Loading ──────────────────────────────────────────────────────────────

def extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract a valid training archive into the expected data directory."""
    data_dir = Path(extract_to)
    train_dir = data_dir / "train"
    bin_path = Path(BIN_DATA_FILE)
    bin_meta_path = Path(BIN_META_FILE)

    if train_dir.exists() or (bin_path.exists() and bin_meta_path.exists()):
        log.info("Training data already exists locally — skipping extraction.")
        return

    if not Path(archive_path).exists():
        raise FileNotFoundError(
            f"Training archive not found: '{archive_path}'.\n"
            "Please place data_training.tar.gz in the project root."
        )

    if data_dir.exists() and not train_dir.exists():
        raise FileExistsError(
            f"Found '{extract_to}', but '{train_dir}' is missing.\n"
            "Please inspect that path before training again."
        )

    log.info("Inspecting archive '%s' ...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError("Training archive is empty.")

        member_paths = []
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive entry: '{member.name}'")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not supported: '{member.name}'")
            if member.name:
                member_paths.append(member_path)

        top_levels = {
            path.parts[0]
            for path in member_paths
            if path.parts and path.parts[0] not in {".", ""}
        }
        data_root = data_dir.name

        if data_root in top_levels:
            extract_root = data_dir.parent
        elif "train" in top_levels or "val" in top_levels:
            extract_root = data_dir
        elif BIN_DATA_FILE in top_levels or BIN_META_FILE in top_levels:
            extract_root = data_dir.parent
        else:
            raise ValueError(
                "Archive layout is invalid. Expected either "
                f"'{data_root}/train/...', 'train/...', or "
                f"'{BIN_DATA_FILE}' + '{BIN_META_FILE}'. Found: {sorted(top_levels)}"
            )

        log.info("Extracting '%s' → '%s' ...", archive_path, extract_root)
        tar.extractall(path=extract_root)

    if train_dir.exists():
        log.info("Extraction complete.")
        return

    if bin_path.exists() and bin_meta_path.exists():
        log.info("Extraction complete.")
        return

    if bin_path.exists() and not bin_meta_path.exists():
        raise FileNotFoundError(
            f"Found '{BIN_DATA_FILE}' after extraction, but '{BIN_META_FILE}' is missing.\n"
            "Raw binary training data requires a metadata sidecar describing the record layout."
        )

    raise FileNotFoundError(
        f"Archive extracted, but neither '{train_dir}' nor "
        f"'{bin_path}' + '{bin_meta_path}' were created. "
        "Please verify the archive structure."
    )


def load_binary_metadata(meta_path: str) -> dict:
    """Load and validate the metadata required to interpret raw binary records."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_shape = meta.get("feature_shape")
    if not feature_shape:
        raise FileNotFoundError(
            f"Metadata '{meta_path}' must define 'feature_shape'."
        )

    if not isinstance(feature_shape, list) or not all(isinstance(v, int) and v > 0 for v in feature_shape):
        raise ValueError("'feature_shape' must be a list of positive integers.")

    feature_dtype = np.dtype(meta.get("feature_dtype", "uint8"))
    label_dtype = np.dtype(meta.get("label_dtype", "uint8"))
    feature_bytes = int(np.prod(feature_shape)) * feature_dtype.itemsize
    label_bytes = int(meta.get("label_bytes", label_dtype.itemsize))
    label_offset = int(meta.get("label_offset_bytes", 0))
    feature_offset = int(meta.get("feature_offset_bytes", label_offset + label_bytes))
    record_bytes = int(
        meta.get(
            "record_bytes",
            max(label_offset + label_bytes, feature_offset + feature_bytes),
        )
    )
    header_bytes = int(meta.get("header_bytes", 0))
    validation_split = float(meta.get("validation_split", 0.2))
    if not 0.0 < validation_split < 1.0:
        raise ValueError("'validation_split' must be between 0 and 1.")

    class_names = meta.get("class_names")
    if class_names is None:
        num_classes = int(meta.get("num_classes", 2))
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = [str(name) for name in class_names]

    return {
        "feature_shape": tuple(feature_shape),
        "feature_dtype": feature_dtype,
        "feature_bytes": feature_bytes,
        "feature_offset": feature_offset,
        "label_dtype": label_dtype,
        "label_bytes": label_bytes,
        "label_offset": label_offset,
        "record_bytes": record_bytes,
        "header_bytes": header_bytes,
        "validation_split": validation_split,
        "class_names": class_names,
        "shuffle_buffer": int(meta.get("shuffle_buffer", 1000)),
    }


def build_binary_datasets(bin_path: str, meta_path: str):
    """
    Build train / validation datasets from a fixed-length raw-record binary file.

    The companion JSON metadata describes how each record is laid out.
    """
    meta = load_binary_metadata(meta_path)
    total_bytes = Path(bin_path).stat().st_size - meta["header_bytes"]
    if total_bytes <= 0:
        raise ValueError(f"Binary data file '{bin_path}' is empty.")
    if total_bytes % meta["record_bytes"] != 0:
        raise ValueError(
            f"Binary data size ({total_bytes}) is not divisible by record size "
            f"({meta['record_bytes']})."
        )

    total_records = total_bytes // meta["record_bytes"]
    val_threshold = int(round(meta["validation_split"] * 1000))
    val_records = sum(
        1 for index in range(total_records)
        if ((index * 9973 + 17) % 1000) < val_threshold
    )
    train_records = total_records - val_records
    train_batches = math.ceil(train_records / BATCH_SIZE)
    val_batches = math.ceil(val_records / BATCH_SIZE)

    log.info("Loading raw binary dataset from '%s'", bin_path)
    log.info("Records: %d | Classes (%d): %s", total_records, len(meta["class_names"]), meta["class_names"])
    log.info("Split: %d train / %d val", train_records, val_records)

    raw_ds = tf.data.FixedLengthRecordDataset(
        filenames=[str(bin_path)],
        record_bytes=meta["record_bytes"],
        header_bytes=meta["header_bytes"],
    )

    feature_dtype = tf.as_dtype(meta["feature_dtype"].name)
    label_dtype = tf.as_dtype(meta["label_dtype"].name)
    feature_shape = meta["feature_shape"]
    feature_offset = meta["feature_offset"]
    feature_bytes = meta["feature_bytes"]
    label_offset = meta["label_offset"]
    label_bytes = meta["label_bytes"]

    def parse_record(index, record):
        bucket = tf.math.floormod(index * 9973 + 17, 1000)
        is_val = bucket < val_threshold

        label_raw = tf.strings.substr(record, label_offset, label_bytes)
        label = tf.io.decode_raw(label_raw, label_dtype)[0]
        label = tf.cast(label, tf.int32)

        feature_raw = tf.strings.substr(record, feature_offset, feature_bytes)
        image = tf.io.decode_raw(feature_raw, feature_dtype)
        image = tf.reshape(image, feature_shape)
        image = tf.cast(image, tf.float32)

        return is_val, image, label

    parsed_ds = raw_ds.enumerate().map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        parsed_ds
        .filter(lambda is_val, _image, _label: tf.logical_not(is_val))
        .map(lambda _is_val, image, label: (image, label), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(meta["shuffle_buffer"])
        .batch(BATCH_SIZE)
        .apply(tf.data.experimental.assert_cardinality(train_batches))
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        parsed_ds
        .filter(lambda is_val, _image, _label: is_val)
        .map(lambda _is_val, image, label: (image, label), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .apply(tf.data.experimental.assert_cardinality(val_batches))
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, meta["class_names"]


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
    bin_path = Path(BIN_DATA_FILE)
    meta_path = Path(BIN_META_FILE)

    if train_dir.exists():
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

    if bin_path.exists():
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Found '{bin_path}', but '{meta_path}' is missing.\n"
                "Raw binary training requires a metadata JSON sidecar."
            )
        return build_binary_datasets(str(bin_path), str(meta_path))

    raise FileNotFoundError(
        "No training data found. Expected either "
        f"'{train_dir}' or '{bin_path}'."
    )


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
