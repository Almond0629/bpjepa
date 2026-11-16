"""Official facebookresearch/DINOv3 ViT-B/16 integration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .teacher import FrozenDenseTeacher, TeacherConfig, load_local_teacher


OFFICIAL_PATCH_LAYER = "x_norm_patchtokens"


def official_vitb16_factory(repo_dir: Path) -> nn.Module:
    """Construct the official architecture without loading or downloading weights."""
    root = repo_dir.resolve()
    if not (root / "dinov3" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"official DINOv3 source is missing: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from dinov3.hub.backbones import dinov3_vitb16

    return dinov3_vitb16(pretrained=False)


def official_patch_adapter(model: nn.Module, images: torch.Tensor, layer: str) -> torch.Tensor:
    if layer != OFFICIAL_PATCH_LAYER:
        raise ValueError(f"official dense layer must be {OFFICIAL_PATCH_LAYER!r}")
    output = model.forward_features(images)
    if not isinstance(output, dict) or layer not in output:
        raise ValueError("official DINOv3 forward_features omitted normalized patch tokens")
    return output[layer]


def load_official_vitb16(
    config: TeacherConfig, repo_dir: Path
) -> FrozenDenseTeacher:
    """Load the checksum-verified local checkpoint into official source code."""
    return load_local_teacher(
        config,
        lambda: official_vitb16_factory(repo_dir),
        official_patch_adapter,
    )
