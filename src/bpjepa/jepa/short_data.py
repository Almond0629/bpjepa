"""Fixed centered 64-frame views of the retained 128-frame training manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision

from bpjepa.common import sha256
from bpjepa.jepa.full_data import _require_train_paths, validate_manifest as validate_parent


TEMPORAL_SAMPLING = {
    "policy": "fixed_center_64_from_128_v1",
    "parent_clip_frames": 128,
    "clip_frames": 64,
    "offset_frames": 32,
    "randomness": "none",
    "samples_per_parent": 1,
}


def derive_manifest(parent: dict, parent_path: Path, parent_hash: str, *, require_full_corpus: bool = True) -> dict:
    """Preserve sample IDs/order/count; frame intervals use zero-based endpoints."""
    validate_parent(parent, require_full_corpus=require_full_corpus)
    _require_train_paths(Path(parent["source_manifest"]), Path(parent["frame_root"]))
    result = copy.deepcopy(parent)
    result.update(
        clip_frames=64,
        clip_seconds=32.0,
        boundary_policy="centered_subclip_of_each_parent_window",
        incomplete_window_policy="inherit parent windows; retain central 64 frames only",
        parent_manifest={"path": str(parent_path.resolve()), "sha256": parent_hash},
        temporal_sampling=copy.deepcopy(TEMPORAL_SAMPLING),
    )
    for entry in result["entries"]:
        relative = Path(entry["frame_dir"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe frame directory")
        entry["start_frame"] += 32
        entry["end_frame_exclusive"] -= 32
    return result


def validate_manifest(data: dict) -> None:
    sampling = data.get("temporal_sampling")
    if sampling != TEMPORAL_SAMPLING or data.get("clip_frames") != 64:
        raise ValueError("64-frame runs require a manifest derived by derive_manifest")
    entries = data.get("entries", [])
    if not entries or any(e["end_frame_exclusive"] - e["start_frame"] != 64 for e in entries):
        raise ValueError("manifest must contain 64-frame clips")


class Centered64Dataset(Dataset):
    """Deterministic integer indexing supports training, resume, and final export."""

    def __init__(self, manifest: Path, *, frame_root: Path | None = None, image_size: int = 224,
                 require_full_corpus: bool = True):
        self.data = json.loads(Path(manifest).read_text())
        validate_manifest(self.data)
        self.frame_root = Path(frame_root) if frame_root is not None else Path(self.data["frame_root"])
        self.entries = self.data["entries"]
        self.image_size = image_size
        if image_size != 224:
            raise ValueError("the production encoder requires 224x224 input")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        directory = self.frame_root / entry["frame_dir"]
        frames = []
        for frame in range(entry["start_frame"] + 1, entry["end_frame_exclusive"] + 1):
            with Image.open(directory / f"{frame:08d}.jpg") as image:
                value = vision.to_tensor(vision.resize(image.convert("RGB"), [224, 224]))
            frames.append(vision.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        clip = torch.stack(frames, dim=1)
        if clip.shape != (3, 64, 224, 224):
            raise RuntimeError(f"invalid 64-frame clip shape: {tuple(clip.shape)}")
        return clip
