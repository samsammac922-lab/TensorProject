"""
predict.py — Run inference with a saved model on one or more images.

Usage:
    python predict.py --model model_final.keras --image path/to/img.jpg
    python predict.py --model checkpoints/best_model.keras --image img1.jpg img2.png
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

IMG_SIZE = (128, 128)


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.expand_dims(img, 0)          # (1, H, W, 3)


def predict(model_path: str, image_paths: list[str], class_names: list[str]) -> None:
    print(f"Loading model from '{model_path}' …")
    model = keras.models.load_model(model_path)

    for img_path in image_paths:
        if not Path(img_path).exists():
            print(f"  [WARN] File not found: {img_path}")
            continue

        img_tensor = load_image(img_path)
        preds = model.predict(img_tensor, verbose=0)[0]

        if len(preds) == 1:          # binary
            prob  = float(preds[0])
            negative_label = class_names[0] if len(class_names) >= 1 else "0"
            positive_label = class_names[1] if len(class_names) >= 2 else "1"
            label = positive_label if prob >= 0.5 else negative_label
            confidence = prob if prob >= 0.5 else 1 - prob
        else:                         # multi-class
            idx   = int(np.argmax(preds))
            label = class_names[idx] if class_names else str(idx)
            confidence = float(preds[idx])

        print(f"  {img_path:40s}  →  {label}  ({confidence*100:.1f}% confidence)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a saved model")
    parser.add_argument("--model",  default="model_final.keras", help="Path to saved Keras model")
    parser.add_argument("--image",  nargs="+", required=True,    help="Image file(s) to classify")
    parser.add_argument(
        "--classes",
        default=None,
        help='JSON list of class names, e.g. \'["cat","dog"]\'. '
             'Falls back to numeric indices when omitted.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    class_names = json.loads(args.classes) if args.classes else []
    predict(args.model, args.image, class_names)
