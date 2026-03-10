"""
create_sample_data.py
Generates a minimal data_training.tar.gz with synthetic images so you can
test the pipeline without real data.

Usage:
    python create_sample_data.py
"""

import os
import tarfile
import struct
import zlib
import random
from pathlib import Path

CLASSES  = ["cat", "dog"]
SPLITS   = {"train": 40, "val": 10}   # images per class per split
IMG_SIZE = (128, 128)
OUT_TAR  = "data_training.tar.gz"


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


if __name__ == "__main__":
    import shutil
    tmp = Path("data")
    if tmp.exists():
        shutil.rmtree(tmp)
    build_tree(tmp)
    pack(tmp, OUT_TAR)
    shutil.rmtree(tmp)          # clean up; train.py will re-extract from the .tar.gz
    print("Done. Run:  python train.py")
