"""Evidence-free random tube-block sampling for the Random V-JEPA baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class MaskBatch:
    context: torch.Tensor
    targets: torch.Tensor


class TargetSampler(Protocol):
    """Interface shared by random and future evidence-guided target samplers."""

    num_targets: int

    def sample(
        self,
        batch_size: int,
        temporal: int,
        height: int,
        width: int,
        *,
        generator: torch.Generator | None = None,
    ) -> MaskBatch: ...


class RandomBlockSampler:
    """Sample fixed-size spatial blocks repeated across the temporal token grid."""

    def __init__(
        self,
        num_targets: int = 4,
        target_scale: float | tuple[float, float] = 0.15,
        target_aspect_ratio: float | tuple[float, float] = (0.75, 1.5),
    ) -> None:
        if num_targets < 1:
            raise ValueError("num_targets must be positive")
        self.target_scale = self._range(target_scale, "target_scale", upper=1)
        self.target_aspect_ratio = self._range(
            target_aspect_ratio, "target_aspect_ratio"
        )
        self.num_targets = num_targets

    @staticmethod
    def _range(
        value: float | tuple[float, float], name: str, upper: float | None = None
    ) -> tuple[float, float]:
        values = (float(value), float(value)) if isinstance(value, (int, float)) else value
        if len(values) != 2 or values[0] <= 0 or values[0] > values[1]:
            raise ValueError(f"{name} must be a positive ordered pair")
        if upper is not None and values[1] >= upper:
            raise ValueError(f"{name} maximum must be below {upper}")
        return float(values[0]), float(values[1])

    @staticmethod
    def _sample_range(
        bounds: tuple[float, float], generator: torch.Generator | None
    ) -> float:
        low, high = bounds
        return low + float(torch.rand((), generator=generator)) * (high - low)

    def block_shape(
        self,
        height: int,
        width: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[int, int]:
        scale = self._sample_range(self.target_scale, generator)
        aspect = self._sample_range(self.target_aspect_ratio, generator)
        area = max(1, int(height * width * scale))
        block_h = max(1, min(height, round(math.sqrt(area * aspect))))
        block_w = max(1, min(width, round(area / block_h)))
        return block_h, block_w

    def sample(
        self,
        batch_size: int,
        temporal: int,
        height: int,
        width: int,
        *,
        generator: torch.Generator | None = None,
    ) -> MaskBatch:
        if min(batch_size, temporal, height, width) < 1:
            raise ValueError("all grid dimensions must be positive")
        block_h, block_w = self.block_shape(height, width, generator=generator)
        for _ in range(256):
            spatial_targets = torch.zeros(
                batch_size, self.num_targets, height, width, dtype=torch.bool
            )
            for batch in range(batch_size):
                for target in range(self.num_targets):
                    top = int(
                        torch.randint(height - block_h + 1, (), generator=generator)
                    )
                    left = int(
                        torch.randint(width - block_w + 1, (), generator=generator)
                    )
                    spatial_targets[
                        batch, target, top : top + block_h, left : left + block_w
                    ] = True
            context_spatial = ~spatial_targets.any(dim=1)
            if context_spatial.any(dim=1).all():
                break
        else:
            raise RuntimeError("random target blocks consumed the entire context")
        union = spatial_targets.any(dim=1)
        context_spatial = ~union
        min_context = int(context_spatial.sum(dim=(1, 2)).min())
        for batch in range(batch_size):
            indices = context_spatial[batch].flatten().nonzero().flatten()
            context_spatial[batch].flatten()[indices[min_context:]] = False
        targets = (
            spatial_targets[:, :, None]
            .expand(-1, -1, temporal, -1, -1)
            .reshape(batch_size, self.num_targets, -1)
        )
        context = (
            context_spatial[:, None]
            .expand(-1, temporal, -1, -1)
            .reshape(batch_size, -1)
        )
        return MaskBatch(context=context, targets=targets)
