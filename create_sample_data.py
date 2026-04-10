"""
create_sample_data.py
Generates a minimal data_training.tar.gz with synthetic images so you can
test the pipeline without real data.

Usage:
    python create_sample_data.py
    python create_sample_data.py --format raw-bin
"""

import argparse
import os
import tarfile
import struct
import zlib
import random
import json
from pathlib import Path

CLASSES  = ["cat", "dog"]
SPLITS   = {"train": 40, "val": 10}   # images per class per split
IMG_SIZE = (128, 128)
DEFAULT_OUT_TAR = "sample_data_training.tar.gz"


def _make_png(path: Path, color: tuple[int, int, int]) -> None:
    """Write a tiny solid-colour PNG without external deps."""
    w, h = IMG_SIZE

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))

    raw_rows = b""
    for _ in range(h):
        row = b"\x00" + bytes(color) * w
        raw_rows += row
    idat = chunk(b"IDAT", zlib.compress(raw_rows))
    iend = chunk(b"IEND", b"")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sig + ihdr + idat + iend)


def build_tree(base: Path) -> None:
    colours = {
        "cat": (200, 130, 80),
        "dog": (100, 160, 220),
    }
    for split, count in SPLITS.items():
        for cls in CLASSES:
            for i in range(count):
                noise = tuple(max(0, min(255, c + random.randint(-30, 30)))
                              for c in colours[cls])
                _make_png(base / split / cls / f"{cls}_{i:04d}.png", noise)
    print(f"Sample tree created under '{base}/'")


def pack(base: Path, out: str) -> None:
    with tarfile.open(out, "w:gz") as tar:
        tar.add(str(base), arcname=base.name)
    print(f"Archive written: {out}")


def build_raw_binary(bin_path: Path, meta_path: Path) -> None:
    colours = {
        "cat": (200, 130, 80),
        "dog": (100, 160, 220),
    }
    records = []
    for split, count in SPLITS.items():
        for cls_idx, cls_name in enumerate(CLASSES):
            for _ in range(count):
                pixels = bytearray()
                for _row in range(IMG_SIZE[1]):
                    for _col in range(IMG_SIZE[0]):
                        noise = [
                            max(0, min(255, value + random.randint(-30, 30)))
                            for value in colours[cls_name]
                        ]
                        pixels.extend(noise)
                records.append(bytes([cls_idx]) + bytes(pixels))

    random.shuffle(records)
    bin_path.write_bytes(b"".join(records))
    meta_path.write_text(
        json.dumps(
            {
                "feature_shape": [IMG_SIZE[1], IMG_SIZE[0], 3],
                "feature_dtype": "uint8",
                "label_dtype": "uint8",
                "label_bytes": 1,
                "record_bytes": 1 + IMG_SIZE[0] * IMG_SIZE[1] * 3,
                "class_names": CLASSES,
                "validation_split": 0.2,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Binary dataset written: {bin_path}")
    print(f"Metadata written: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--format",
        choices=["image-tree", "raw-bin"],
        default="image-tree",
        help="Which dataset format to generate",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT_TAR,
        help=f"Output archive path (default: {DEFAULT_OUT_TAR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import shutil
    args = parse_args()

    tmp = Path("data")
    bin_path = Path("data_training.bin")
    meta_path = Path("data_training.meta.json")

    if tmp.exists():
        shutil.rmtree(tmp)
    if bin_path.exists():
        bin_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    if args.format == "image-tree":
        build_tree(tmp)
        pack(tmp, args.output)
        shutil.rmtree(tmp)      # clean up; train.py will re-extract from the .tar.gz
    else:
        build_raw_binary(bin_path, meta_path)
        with tarfile.open(args.output, "w:gz") as tar:
            tar.add(str(bin_path), arcname=bin_path.name)
            tar.add(str(meta_path), arcname=meta_path.name)
        bin_path.unlink()
        meta_path.unlink()

    print("Done. Run:  python train.py")
