"""Independent V-JEPA tube groups with upstream flattened-prefix collation.

Reference: facebookresearch/jepa, src/masks/multiblock3d.py.
Randomness uses the trainer's checkpointed generator rather than worker counters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DEFAULT_MASK = {
    "small": {"num_blocks": 8, "spatial_scale": 0.15,
              "aspect_ratio": [0.75, 1.5], "temporal_scale": 1.0},
    "large": {"num_blocks": 2, "spatial_scale": 0.70,
              "aspect_ratio": [0.75, 1.5], "temporal_scale": 1.0},
}


@dataclass(frozen=True)
class MaskGroup:
    name: str
    context_indices: torch.Tensor
    target_indices: torch.Tensor
    target_union: torch.Tensor  # full tubes before batch truncation


class VJEPAMultiMaskSampler:
    def __init__(self, mask: dict | None = None) -> None:
        mask = DEFAULT_MASK if mask is None else mask
        if set(mask) != {"small", "large"}:
            raise ValueError("mask requires separate small and large groups")
        self.groups = []
        for name in ("small", "large"):
            cfg = mask[name]
            if set(cfg) != {"num_blocks", "spatial_scale", "aspect_ratio", "temporal_scale"}:
                raise ValueError(f"unsupported or missing mask fields in {name}")
            count = cfg["num_blocks"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("num_blocks must be a positive integer")
            scale = self._range(cfg["spatial_scale"])
            aspect = self._range(cfg["aspect_ratio"])
            if scale[1] >= 1:
                raise ValueError("spatial_scale must be below one")
            if self._range(cfg["temporal_scale"]) != (1.0, 1.0):
                raise ValueError("this COMEDIAN candidate requires full temporal tubes")
            self.groups.append((name, count, scale, aspect))

    @staticmethod
    def _range(value):
        pair = (value, value) if isinstance(value, (int, float)) else tuple(value)
        if len(pair) != 2 or not all(math.isfinite(v) for v in pair) or not 0 < pair[0] <= pair[1]:
            raise ValueError("mask ranges must be finite positive ordered pairs")
        return tuple(float(v) for v in pair)

    def sample(self, batch_size, temporal, height, width, *, generator=None):
        if min(batch_size, temporal, height, width) < 1:
            raise ValueError("all grid dimensions must be positive")
        # Upstream uses the same iteration seed for both groups' shape draws.
        seed = int(torch.randint(2**31 - 1, (), generator=generator))
        result = []
        for name, count, scale, aspect in self.groups:
            shape_rng = torch.Generator().manual_seed(seed)
            torch.rand(1, generator=shape_rng)  # fixed temporal-scale draw
            scale_draw = scale[0] + torch.rand(1, generator=shape_rng).item() * (scale[1] - scale[0])
            area = int(height * width * scale_draw)
            ratio = aspect[0] + torch.rand(1, generator=shape_rng).item() * (aspect[1] - aspect[0])
            h = min(height, round(math.sqrt(area * ratio)))
            w = min(width, round(math.sqrt(area / ratio)))
            if min(h, w) < 1 or (h == height and w == width):
                raise ValueError("mask geometry cannot produce nonempty targets and context on this grid")
            contexts, targets, unions = [], [], []
            for _ in range(batch_size):
                for attempt in range(1024):
                    union = torch.zeros(temporal, height, width, dtype=torch.bool)
                    for _ in range(count):
                        top = int(torch.randint(height - h + 1, (), generator=generator))
                        left = int(torch.randint(width - w + 1, (), generator=generator))
                        torch.randint(1, (), generator=generator)  # full-tube start draw
                        union[:, top:top+h, left:left+w] = True
                    context = (~union.flatten()).nonzero().flatten()
                    if context.numel():
                        break
                else:
                    raise RuntimeError(f"{name} mask consumed all context in 1024 attempts")
                contexts.append(context)
                targets.append(union.flatten().nonzero().flatten())
                unions.append(union.flatten())
            nc = min(x.numel() for x in contexts)
            nt = min(x.numel() for x in targets)
            result.append(MaskGroup(name, torch.stack([x[:nc] for x in contexts]),
                                    torch.stack([x[:nt] for x in targets]), torch.stack(unions)))
        return result
