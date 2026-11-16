"""Automatic per-run record: resolved configuration, seed, software, inputs and outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from bpjepa import __version__
from bpjepa.common import atomic_json, canonical_digest, sha256

PACKAGES = ("torch", "torchvision", "numpy", "lightning", "omegaconf", "hydra-core", "opencv-python",
            "opencv-python-headless", "pillow", "scipy", "torchaug", "SoccerNet", "tensorboard")


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def _hardware() -> dict:
    try:
        import torch
    except ImportError:
        return {}
    result = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def file_reference(path: str | Path | None, *, hash_file: bool = True) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path) if hash_file and path.is_file() else None}


def write_run_record(output_dir: str | Path, *, stage: str, config: dict, seed, inputs: dict | None = None,
                     outputs: dict | None = None, name: str = "run_record.json") -> Path:
    """Write output_dir/run_record.json. No prior audit record is read or required."""
    path = Path(output_dir) / name
    record = {"stage": stage, "written_at": datetime.now(timezone.utc).isoformat(), "bpjepa_version": __version__,
              "seed": seed, "resolved_config": config, "resolved_config_digest": canonical_digest(config),
              "inputs": inputs or {}, "outputs": outputs or {}, "argv": sys.argv,
              "python": platform.python_version(), "packages": _versions(), "hardware": _hardware(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    atomic_json(path, json.loads(json.dumps(record, default=str)))
    return path
