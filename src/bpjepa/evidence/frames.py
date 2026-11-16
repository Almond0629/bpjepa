"""Canonical half-frame file helpers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def decode_verified_jpeg(data: bytes) -> Image.Image:
    """Decode exactly the bytes just hashed, not a second file read."""
    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def ordered_half_frames(frame_dir: Path, frame_count: int) -> list[Path]:
    if frame_count < 2 or not frame_dir.is_dir():
        raise ValueError(f"missing or short canonical half: {frame_dir}")
    expected = [frame_dir / f"{index:08d}.jpg" for index in range(1, frame_count + 1)]
    actual = sorted(frame_dir.glob("*.jpg"))
    if actual != expected:
        raise RuntimeError(f"canonical half has missing, extra, or misnamed JPEGs: {frame_dir}")
    return expected
