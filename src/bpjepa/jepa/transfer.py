"""Tensor-only export of a JEPA context encoder into COMEDIAN's spatial slot."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import torch

from .model import ComedianViTTiny


def comedian_artifact(
    encoder: ComedianViTTiny,
) -> dict:
    """Build the portable handoff as a plain tensor state_dict."""
    encoder.eval()
    return {
        f"trunk.{key}": value.detach().cpu().contiguous()
        for key, value in encoder.state_dict().items()
    }


def export_comedian(
    encoder: ComedianViTTiny,
    path: Path,
    verification_clip: torch.Tensor | None = None,
) -> Path | None:
    artifact = comedian_artifact(encoder)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    torch.save(artifact, temporary)
    torch.load(temporary, map_location="cpu", weights_only=True)
    os.replace(temporary, path)
    if verification_clip is None:
        return None
    device = next(encoder.parameters()).device
    with torch.no_grad():
        verification = {
            "input": verification_clip.detach().cpu(),
            "output": encoder(verification_clip.to(device)).detach().cpu(),
        }
    verification_path = path.with_suffix(path.suffix + ".verification.pt")
    temporary = verification_path.with_name(
        f".{verification_path.name}.{uuid.uuid4().hex}.partial"
    )
    torch.save(verification, temporary)
    torch.load(temporary, map_location="cpu", weights_only=True)
    os.replace(temporary, verification_path)
    return verification_path


def load_export(path: Path, encoder: ComedianViTTiny) -> None:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    state = artifact.get("state_dict", artifact)
    state = {key.removeprefix("trunk."): value for key, value in state.items()}
    encoder.load_state_dict(state, strict=True)
