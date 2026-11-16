"""Shared U/E/M support geometry and temporal reduction.

The base sampler and JEPA loss remain unchanged. All indices are time-major.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from bpjepa.samplers.vjepa_multimask import MaskGroup


@dataclass(frozen=True)
class GroupSupport:
    visible: torch.Tensor  # [B, N], original spatial support
    complete: torch.Tensor  # [B, T], exact retained support
    partial: torch.Tensor  # [B, T]
    support_size: torch.Tensor  # [B]

    def to(self, device: torch.device | str) -> GroupSupport:
        return GroupSupport(*(value.to(device) for value in
                              (self.visible, self.complete, self.partial, self.support_size)))


def group_support(group: MaskGroup, temporal: int, height: int, width: int) -> GroupSupport:
    """Verify the sampler's flat-prefix contract and find exact complete tubes."""
    if min(temporal, height, width) < 1:
        raise ValueError("invalid token grid")
    batch, length = group.context_indices.shape
    spatial = height * width
    total = temporal * spatial
    ci, ti, union = group.context_indices, group.target_indices, group.target_union
    if (ci.dtype != torch.long or ti.dtype != torch.long or union.dtype != torch.bool
            or ti.ndim != 2 or ti.shape[0] != batch or union.shape != (batch, total)
            or length == 0 or ti.shape[1] == 0):
        raise ValueError("invalid mask-group tensors")
    if bool((union.reshape(batch, temporal, spatial) != union[:, :spatial, None].transpose(1, 2)).any()):
        raise ValueError("target_union is not a full spatial tube")
    if bool((ci < 0).any()) or bool((ci >= total).any()) or bool((ti < 0).any()) or bool((ti >= total).any()):
        raise ValueError("mask index outside token grid")
    if bool((ci[:, 1:] <= ci[:, :-1]).any()) or bool((ti[:, 1:] <= ti[:, :-1]).any()):
        raise ValueError("mask indices must be strictly ordered")
    visible = ~union[:, :spatial]
    for b in range(batch):
        original_context = (~union[b]).nonzero().flatten()
        original_target = union[b].nonzero().flatten()
        if not torch.equal(ci[b], original_context[:length]):
            raise ValueError("retained context is not the original flat prefix")
        if not torch.equal(ti[b], original_target[:ti.shape[1]]):
            raise ValueError("retained target is not the original flat prefix")
    occupancy = torch.zeros((batch, total), dtype=torch.bool, device=ci.device)
    occupancy.scatter_(1, ci, True)
    occupancy = occupancy.reshape(batch, temporal, spatial)
    complete = (occupancy == visible[:, None, :]).all(dim=-1)
    partial = occupancy.any(dim=-1) & ~complete
    return GroupSupport(visible, complete, partial, visible.sum(dim=-1))


def pool_student(features: torch.Tensor, indices: torch.Tensor, support: GroupSupport) -> torch.Tensor:
    """Mean retained student tokens by timestamp; only complete rows may anchor."""
    if features.ndim != 3 or features.shape[:2] != indices.shape:
        raise ValueError("student feature/index shape mismatch")
    batch, temporal = support.complete.shape
    if features.shape[0] != batch:
        raise ValueError("support batch mismatch")
    spatial = support.visible.shape[1]
    values = features.float()
    pooled = torch.zeros((batch, temporal, values.shape[-1]), device=values.device, dtype=values.dtype)
    pooled.scatter_add_(1, (indices // spatial)[:, :, None].expand_as(values), values)
    counts = torch.zeros((batch, temporal), device=values.device, dtype=values.dtype)
    counts.scatter_add_(1, indices // spatial, torch.ones_like(indices, dtype=values.dtype))
    return pooled / counts.clamp_min(1)[:, :, None]


def pool_teacher(teacher: torch.Tensor, support: GroupSupport) -> torch.Tensor:
    """Pool the existing full teacher pass over each sample's original support."""
    batch = support.visible.shape[0]
    spatial = support.visible.shape[1]
    if teacher.ndim != 3 or teacher.shape[0] != batch or teacher.shape[1] % spatial:
        raise ValueError("teacher feature grid mismatch")
    if not bool(torch.isfinite(teacher).all()):
        raise ValueError("MISSING_NONFINITE_FEATURES: teacher features block production")
    values = teacher.float().reshape(batch, -1, spatial, teacher.shape[-1])
    return (values * support.visible[:, None, :, None]).sum(dim=2) / support.support_size[:, None, None]


def candidates(support: GroupSupport, offsets: tuple[int, ...], v_min: int,
               *, pair_valid: torch.Tensor | None = None,
               key_valid: torch.Tensor | None = None,
               geometry_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchors [B,T] and negatives [B,T,T]; unknown validity never defaults true."""
    if geometry_only:
        if pair_valid is not None or key_valid is not None:
            raise ValueError("geometry-only candidates cannot consume validity metadata")
    elif pair_valid is None or key_valid is None:
        raise ValueError("explicit pair and positive/negative key validity are required")
    if not offsets or len(set(offsets)) != len(offsets) or 0 in offsets or v_min < 1:
        raise ValueError("offsets must be unique, nonzero, and v_min positive")
    batch, temporal = support.complete.shape
    times = torch.arange(temporal, device=support.complete.device)
    allowed = torch.zeros((temporal, temporal), dtype=torch.bool, device=times.device)
    for delta in offsets:
        allowed |= (times[None, :] - times[:, None]) == delta
    allowed = allowed[None].expand(batch, -1, -1).clone()
    if pair_valid is not None:
        if pair_valid.shape != allowed.shape or pair_valid.dtype != torch.bool:
            raise ValueError("pair validity shape/dtype mismatch")
        allowed &= pair_valid
    if key_valid is not None:
        if key_valid.shape != (batch, temporal) or key_valid.dtype != torch.bool:
            raise ValueError("key validity shape/dtype mismatch")
        if bool((pair_valid & ~(key_valid[:, :, None] & key_valid[:, None, :])).any()):
            raise ValueError("pair validity implies valid endpoint keys; inconsistent metadata")
        allowed &= key_valid[:, None, :]
    eligible = support.complete & (support.support_size[:, None] >= v_min) & allowed.any(dim=-1)
    if key_valid is not None:
        eligible &= key_valid  # the same-time positive must be usable too
    return eligible, allowed & eligible[:, :, None]


def temporal_sample_mean(anchor_losses: dict[str, torch.Tensor],
                         eligible: dict[str, torch.Tensor],
                         *, alpha: dict[str, float] | None = None,
                         global_batch: int | None = None,
                         world_size: int = 1) -> torch.Tensor:
    """Fixed group weights, eligible-anchor means, then DDP-correct sample mean."""
    alpha = {"small": 1.0, "large": 0.0} if alpha is None else alpha
    if set(alpha) != {"small", "large"} or any(x < 0 for x in alpha.values()) or abs(sum(alpha.values()) - 1) > 1e-12:
        raise ValueError("alpha must be nonnegative and sum to one")
    if set(anchor_losses) != set(eligible) or set(anchor_losses) != set(alpha):
        raise ValueError("both groups must be supplied")
    batch = next(iter(anchor_losses.values())).shape[0]
    if batch < 1 or world_size < 1 or (global_batch is not None and global_batch < batch):
        raise ValueError("invalid batch normalization")
    total = None
    for name, weight in alpha.items():
        losses, active = anchor_losses[name], eligible[name]
        if losses.shape != active.shape or losses.shape[0] != batch or active.dtype != torch.bool:
            raise ValueError("loss/eligibility shape mismatch")
        # The masked multiplication keeps inactive zero terms connected to autograd.
        per_sample = (losses * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        term = weight * per_sample.sum()
        total = term if total is None else total + term
    denominator = batch if global_batch is None else global_batch / world_size
    return total / denominator
