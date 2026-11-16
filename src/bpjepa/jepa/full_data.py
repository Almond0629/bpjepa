#!/usr/bin/env python3
"""Deterministic full-corpus SoccerNet clips for latent-prediction pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as vision

from bpjepa.common import atomic_json, refuse_existing, sha256


SCHEMA_VERSION = 1
BOUNDARY_POLICY = "anchor_final_full_window_no_padding"
ORDERING = "lexicographic_game_numeric_half_ascending_start"
AUGMENTATION_POLICY = "phase1_resize_imagenet_normalize_no_random_augmentation"


def _require_train_paths(source_manifest: Path, frame_root: Path) -> None:
    source_manifest = source_manifest.resolve()
    frame_root = frame_root.resolve()
    forbidden = {"valid", "validation", "val", "test"}
    if source_manifest.name != "train.json" or frame_root.name != "train":
        raise ValueError("production data must use train.json and a train frame root")
    if forbidden.intersection(part.lower() for part in (*source_manifest.parts, *frame_root.parts)):
        raise ValueError("validation/test paths are forbidden in pretraining")


def clip_starts(frame_count: int, clip_frames: int, stride_frames: int) -> list[int]:
    """Return complete windows, anchoring one final window at the half boundary."""
    if min(frame_count, clip_frames, stride_frames) < 1:
        raise ValueError("frame_count, clip_frames, and stride_frames must be positive")
    if frame_count < clip_frames:
        raise ValueError("a training half is shorter than one complete clip")
    last = frame_count - clip_frames
    starts = list(range(0, last + 1, stride_frames))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _verified_frame_count(directory: Path, declared: int) -> int:
    if not directory.is_dir():
        raise FileNotFoundError(f"decoded training half is missing: {directory}")
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) != declared:
        raise RuntimeError(
            f"decoded frame count mismatch for {directory}: declared {declared}, found {len(paths)}"
        )
    expected = [f"{index:08d}.jpg" for index in range(1, declared + 1)]
    actual = [path.name for path in paths]
    if actual != expected:
        raise RuntimeError(f"decoded frames are not a contiguous 1-based sequence: {directory}")
    return len(paths)


def build_manifest_data(
    source_manifest: Path,
    frame_root: Path,
    *,
    clip_frames: int = 128,
    stride_frames: int = 128,
    fps: int = 2,
    verify_frames: bool = True,
    require_full_corpus: bool = True,
) -> dict:
    """Build canonical manifest content without writing it."""
    _require_train_paths(source_manifest, frame_root)
    if clip_frames != 128:
        raise ValueError("the paper recipe requires 128-frame parent clips")
    if fps != 2:
        raise ValueError("the frozen SoccerNet extraction rate is 2 fps")
    source = json.loads(source_manifest.read_text())
    if not isinstance(source, dict) or not source:
        raise ValueError("train.json must be a non-empty game mapping")

    entries: list[dict] = []
    halves: list[str] = []
    for game in sorted(source):
        game_record = source[game]
        half_records = game_record.get("halves", {})
        for half in sorted(half_records, key=int):
            record = half_records[half]
            relative = Path(record["UrlLocal"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe frame path in train.json: {relative}")
            directory = (frame_root / relative).resolve()
            try:
                directory.relative_to(frame_root.resolve())
            except ValueError as error:
                raise ValueError(f"frame path escapes the train root: {directory}") from error
            declared = int(record["num_frames"])
            if int(record["fps"]) != fps:
                raise ValueError(f"unexpected fps for {game} half {half}")
            frame_count = _verified_frame_count(directory, declared) if verify_frames else declared
            half_id = f"{game}::{half}"
            halves.append(half_id)
            for start in clip_starts(frame_count, clip_frames, stride_frames):
                entries.append(
                    {
                        "sample_id": len(entries),
                        "game": game,
                        "half": int(half),
                        "half_id": half_id,
                        "frame_dir": relative.as_posix(),
                        "start_frame": start,
                        "end_frame_exclusive": start + clip_frames,
                    }
                )

    if require_full_corpus and (len(source) != 300 or len(halves) != 600):
        raise RuntimeError(
            f"full training corpus must contain 300 games/600 halves, found {len(source)}/{len(halves)}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "frame_root": str(frame_root.resolve()),
        "games": len(source),
        "halves": len(halves),
        "half_ids": halves,
        "fps": fps,
        "clip_frames": clip_frames,
        "clip_seconds": clip_frames / fps,
        "stride_frames": stride_frames,
        "stride_seconds": stride_frames / fps,
        "boundary_policy": BOUNDARY_POLICY,
        "incomplete_window_policy": "discard; final complete window is anchored to the half end",
        "ordering": ORDERING,
        "augmentation_policy": AUGMENTATION_POLICY,
        "epoch_semantics": "one traversal of every base manifest entry; DDP padding is explicit in the frozen run config",
        "samples_per_epoch": len(entries),
        "entries": entries,
    }


def validate_manifest(data: dict, *, expected_hash: str | None = None, path: Path | None = None,
                      require_full_corpus: bool = True) -> None:
    if expected_hash is not None:
        if path is None or sha256(path) != expected_hash:
            raise RuntimeError("clip manifest hash mismatch")
    required = {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "clip_frames": 128,
        "fps": 2,
        "boundary_policy": BOUNDARY_POLICY,
        "ordering": ORDERING,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(f"invalid production manifest {key}: {data.get(key)!r}")
    if len(set(data.get("half_ids", []))) != len(data.get("half_ids", [])) or not data.get("half_ids"):
        raise ValueError("manifest half IDs must be unique and nonempty")
    if require_full_corpus and (data.get("games") != 300 or data.get("halves") != 600 or len(data["half_ids"]) != 600):
        raise ValueError("manifest does not contain the full 300-game/600-half training corpus "
                         "(use a subset only with --allow-subset-corpus for smoke runs)")
    entries = data.get("entries", [])
    if len(entries) != data.get("samples_per_epoch") or not entries:
        raise ValueError("manifest sample count is inconsistent")
    if [entry.get("sample_id") for entry in entries] != list(range(len(entries))):
        raise ValueError("manifest sample IDs are not canonical")
    if any(entry["end_frame_exclusive"] - entry["start_frame"] != 128 for entry in entries):
        raise ValueError("manifest contains non-128-frame clips")
    forbidden = {"valid", "validation", "val", "test"}
    for entry in entries:
        parts = {part.lower() for part in Path(entry["frame_dir"]).parts}
        if parts.intersection(forbidden):
            raise ValueError("manifest contains validation/test data")


class FullSoccerNetClipDataset(Dataset):
    """Load exact clips from a clip manifest. --frame-root overrides the path recorded in the manifest."""

    def __init__(self, manifest: Path, *, frame_root: Path | None = None, image_size: int = 224,
                 with_row_id: bool = False, require_full_corpus: bool = True) -> None:
        self.manifest_path = Path(manifest).resolve()
        if with_row_id:
            manifest_bytes = self.manifest_path.read_bytes()
            self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self.data = json.loads(manifest_bytes)
        else:
            self.manifest_sha256 = None
            self.data = json.loads(self.manifest_path.read_text())
        validate_manifest(self.data, require_full_corpus=require_full_corpus)
        self.frame_root = Path(frame_root) if frame_root is not None else Path(self.data["frame_root"])
        _require_train_paths(Path(self.data["source_manifest"]), self.frame_root)
        self.entries: Sequence[dict] = self.data["entries"]
        self.image_size = int(image_size)
        self.with_row_id = bool(with_row_id)
        if self.image_size != 224:
            raise ValueError("the production encoder requires 224x224 input")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> torch.Tensor:
        entry = self.entries[index]
        directory = self.frame_root / entry["frame_dir"]
        tensors = []
        for frame in range(entry["start_frame"] + 1, entry["end_frame_exclusive"] + 1):
            path = directory / f"{frame:08d}.jpg"
            with Image.open(path) as image:
                value = vision.to_tensor(
                    vision.resize(image.convert("RGB"), [self.image_size, self.image_size])
                )
            tensors.append(
                vision.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
        clip = torch.stack(tensors, dim=1)
        if clip.shape != (3, 128, self.image_size, self.image_size):
            raise RuntimeError(f"invalid production clip shape: {tuple(clip.shape)}")
        if self.with_row_id:
            return clip, int(entry["sample_id"]), self.manifest_sha256
        return clip


class PaddedDistributedEpochSampler(Sampler[int]):
    """Deterministic DDP sampler with explicit minimal whole-update padding."""

    def __init__(
        self,
        dataset_size: int,
        *,
        num_replicas: int,
        rank: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed rank/world size")
        if min(micro_batch_size, gradient_accumulation_steps) < 1:
            raise ValueError("batch and accumulation values must be positive")
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.shuffle = shuffle
        group = num_replicas * micro_batch_size * gradient_accumulation_steps
        self.total_size = math.ceil(dataset_size / group) * group
        self.padding_size = self.total_size - dataset_size
        self.num_samples = self.total_size // num_replicas
        self.epoch = 0
        self.start_index = 0

    def _rank_indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        if self.padding_size:
            indices += (indices * math.ceil(self.padding_size / len(indices)))[: self.padding_size]
        return indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        return iter(self._rank_indices()[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if epoch < 0 or not 0 <= start_index <= self.num_samples:
            raise ValueError("invalid sampler epoch/start index")
        self.epoch = epoch
        self.start_index = start_index

    def state_dict(self) -> dict:
        return {
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "start_index": self.start_index,
            "total_size": self.total_size,
            "padding_size": self.padding_size,
        }

    def load_state_dict(self, state: dict) -> None:
        immutable = (
            "dataset_size",
            "num_replicas",
            "rank",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "seed",
            "shuffle",
            "total_size",
            "padding_size",
        )
        current = self.state_dict()
        for key in immutable:
            if state.get(key) != current[key]:
                raise RuntimeError(f"sampler state mismatch for {key}")
        self.set_epoch(int(state["epoch"]), start_index=int(state["start_index"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic training clip manifest (paper: 26,205 clips).")
    parser.add_argument("--source", type=Path, required=True, help="train.json written by frame extraction")
    parser.add_argument("--frame-root", type=Path, required=True, help="directory of extracted training frames (named train)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-verify-frames", action="store_true", help="trust frame counts declared in train.json")
    parser.add_argument("--allow-subset-corpus", action="store_true", help="smoke runs only; not the paper corpus")
    args = parser.parse_args()
    refuse_existing(args.output)
    data = build_manifest_data(args.source, args.frame_root, verify_frames=not args.no_verify_frames,
                               require_full_corpus=not args.allow_subset_corpus)
    validate_manifest(data, require_full_corpus=not args.allow_subset_corpus)
    atomic_json(args.output, data)
    print(json.dumps({"output": str(args.output), "samples_per_epoch": data["samples_per_epoch"],
                      "entries_sha256": entries_sha256(data), "sha256": sha256(args.output)}))
    return 0


def entries_sha256(data: dict) -> str:
    """Path-independent identity of the row order that target and validity tables index."""
    keep = ("sample_id", "half_id", "frame_dir", "start_frame", "end_frame_exclusive")
    rows = [{k: e[k] for k in keep} for e in data["entries"]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
