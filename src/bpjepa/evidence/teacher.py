"""Local-only DINOv3 dense-feature adapter.

The teacher produces normalized ``[T, H_teacher, W_teacher, C]`` features.
Those features determine where future JEPA targets may be sampled; they are
never JEPA prediction targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


AUTHORIZED_RESOLUTIONS = (448, 672)


@dataclass(frozen=True)
class TeacherConfig:
    checkpoint_id: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    feature_layer: str
    input_resolution: int = 448
    patch_size: int = 16
    interpolation: str = "bicubic"
    antialias: bool = True
    input_range: str = "float_rgb_0_1"
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    teacher_family: str = "DINOv3 ViT-B/16"
    environment_id: str = "dinov3_vitb16"
    implementation_id: str = "local_model_adapter_v1"

    @property
    def grid(self) -> tuple[int, int]:
        return (self.input_resolution // self.patch_size,) * 2

    def validate(self, *, require_checkpoint: bool = True) -> None:
        if not self.checkpoint_id or not self.feature_layer or not self.teacher_family:
            raise ValueError("teacher checkpoint, family, and feature-layer identities are required")
        if self.input_resolution not in AUTHORIZED_RESOLUTIONS:
            raise ValueError("teacher resolution must be exactly 448 or 672")
        if self.patch_size != 16 or self.input_resolution % self.patch_size:
            raise ValueError("Phase 2 requires a ViT-B/16 teacher")
        if self.interpolation not in {"bilinear", "bicubic"}:
            raise ValueError("unsupported teacher interpolation")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("RGB normalization must contain three values")
        if not all(math.isfinite(value) for value in (*self.normalization_mean, *self.normalization_std)):
            raise ValueError("RGB normalization values must be finite")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("normalization standard deviations must be positive")
        if require_checkpoint:
            if not self.checkpoint_path or not self.checkpoint_sha256:
                raise RuntimeError("DINOv3 checkpoint path/checksum is unresolved")
            path = Path(self.checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"local teacher checkpoint is missing: {path}")
            if _sha256(path) != self.checkpoint_sha256:
                raise RuntimeError("teacher checkpoint checksum mismatch")

    def canonical_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def provenance(self) -> dict:
        return {
            **asdict(self),
            "config_hash": self.canonical_hash(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }

    @classmethod
    def from_mapping(cls, value: Mapping) -> "TeacherConfig":
        converted = dict(value)
        for key in ("normalization_mean", "normalization_std"):
            if key in converted:
                converted[key] = tuple(converted[key])
        result = cls(**converted)
        result.validate(require_checkpoint=False)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


DenseAdapter = Callable[[nn.Module, torch.Tensor, str], torch.Tensor]


class FrozenDenseTeacher(nn.Module):
    """Inference-only wrapper around an injected local DINO-compatible model."""

    def __init__(
        self,
        model: nn.Module,
        config: TeacherConfig,
        adapter: DenseAdapter,
        *,
        verify_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        config.validate(require_checkpoint=verify_checkpoint)
        self.model = model.eval().requires_grad_(False)
        self.config = config
        self.adapter = adapter

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("teacher frames must have shape [T, 3, H, W]")
        if not torch.is_floating_point(frames) or not torch.isfinite(frames).all():
            raise ValueError("teacher frames must be finite floating-point RGB in [0, 1]")
        if frames.numel() and (frames.min() < 0 or frames.max() > 1):
            raise ValueError("teacher frames must be in [0, 1]")
        kwargs = {"size": (self.config.input_resolution,) * 2, "mode": self.config.interpolation}
        if self.config.interpolation in {"bilinear", "bicubic"}:
            kwargs.update(align_corners=False, antialias=self.config.antialias)
        frames = F.interpolate(frames, **kwargs)
        mean = frames.new_tensor(self.config.normalization_mean)[None, :, None, None]
        std = frames.new_tensor(self.config.normalization_std)[None, :, None, None]
        return (frames - mean) / std

    @torch.inference_mode()
    def extract(self, frames: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized dense features as ``[T, H_grid, W_grid, C]``."""
        self.model.eval()
        dense = self.adapter(self.model, self._preprocess(frames), self.config.feature_layer)
        if dense.ndim < 1 or dense.shape[0] != frames.shape[0]:
            raise ValueError("teacher output count must match input frame count")
        if dense.ndim == 3:
            temporal, tokens, channels = dense.shape
            expected = self.config.grid[0] * self.config.grid[1]
            if tokens != expected:
                raise ValueError(f"teacher returned {tokens} tokens; expected {expected}")
            dense = dense.reshape(temporal, *self.config.grid, channels)
        elif dense.ndim == 4:
            if dense.shape[1:3] == self.config.grid:
                pass
            elif dense.shape[2:] == self.config.grid:
                dense = dense.permute(0, 2, 3, 1)
            else:
                raise ValueError("teacher dense-map geometry does not match configured grid")
        else:
            raise ValueError("adapter must return [T,N,C], [T,H,W,C], or [T,C,H,W]")
        if not torch.isfinite(dense).all():
            raise ValueError("teacher produced non-finite features")
        return F.normalize(dense.float(), dim=-1, eps=1e-12)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.extract(frames)


def load_local_teacher(
    config: TeacherConfig,
    model_factory: Callable[[], nn.Module],
    feature_adapter: DenseAdapter,
    *,
    state_adapter: Callable[[object], Mapping[str, torch.Tensor]] = lambda value: value,
) -> FrozenDenseTeacher:
    """Strictly load a local checkpoint; this function has no network path."""
    config.validate(require_checkpoint=True)
    payload = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
    state = state_adapter(payload)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("state adapter must return a non-empty tensor state_dict")
    model = model_factory()
    model.load_state_dict(state, strict=True)
    return FrozenDenseTeacher(model, config, feature_adapter, verify_checkpoint=True)
