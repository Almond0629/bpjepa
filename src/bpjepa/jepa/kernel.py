"""Shared, sample-normalized weighted temporal InfoNCE.

Providers receive a checked private copy of the candidate metadata. Tensor
mutation is rejected, and cannot corrupt the original forward's candidates.
No teacher forward, persistent key cache, or production trainer hook lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from bpjepa.jepa.support import (
    GroupSupport, candidates, group_support, pool_student, pool_teacher,
    temporal_sample_mean,
)
from bpjepa.samplers.vjepa_multimask import MaskGroup


EMPTY_CAUSES = ("nonempty", "no_complete_timestamps", "below_v_min",
                "invalid_positive", "no_valid_candidates", "padding")
LOW_MASS_REPORTING_THRESHOLD = 1e-3  # Reporting only; never changes active/eligible.
GROUPS = ("small", "large")


@dataclass(frozen=True)
class KernelSettings:
    offsets: tuple[int, ...]
    tau: float
    alpha: tuple[float, float]
    v_min: int
    eta_q: float
    reduction: str

    @classmethod
    def from_fields(cls, fields: dict) -> KernelSettings:
        """Build from already-resolved shared fields (a resolved pretraining config)."""
        result = cls(tuple(fields["offsets"]), fields["tau"],
                     tuple(fields["alpha"][g] for g in GROUPS), fields["v_min"],
                     fields["eta_q"], fields["reduction"])
        result.validate()
        return result

    def validate(self) -> None:
        if self.eta_q != 0:
            raise ValueError("this release requires eta_q == 0")
        if self.reduction != "per_sample_group_v1":
            raise ValueError("unsupported temporal reduction")
        if (not self.offsets or any(type(d) is not int or d == 0 for d in self.offsets)
                or len(set(self.offsets)) != len(self.offsets) or self.v_min < 1):
            raise ValueError("invalid offsets or v_min")
        if not 0 < self.tau < float("inf"):
            raise ValueError("tau must be finite and positive")
        if (len(self.alpha) != 2 or any(not 0 <= a <= 1 for a in self.alpha)
                or abs(sum(self.alpha) - 1) > 1e-12):
            raise ValueError("alpha must be nonnegative and sum to one")


@dataclass(frozen=True)
class CandidateBundle:
    """Owned metadata in [sample, group, anchor, offset] order.

    indices[..., :] = (sample ID, group ID, anchor timestamp, key timestamp).
    Out-of-clip key timestamps use -1. All tensor fields participate in the
    provider nonmutation guard. Eligibility precedes, and never uses, weights.
    """
    indices: torch.Tensor
    valid: torch.Tensor
    in_bounds: torch.Tensor
    eligible: torch.Tensor
    complete: torch.Tensor
    partial: torch.Tensor
    visible: torch.Tensor
    support_size: torch.Tensor
    pair_valid: torch.Tensor
    key_valid: torch.Tensor
    pair_reasons: torch.Tensor
    key_reasons: torch.Tensor
    empty_causes: torch.Tensor
    sample_padding: torch.Tensor
    settings: KernelSettings

    def copy(self) -> CandidateBundle:
        return CandidateBundle(**{f.name: getattr(self, f.name).clone()
                                  if isinstance(getattr(self, f.name), torch.Tensor)
                                  else getattr(self, f.name) for f in fields(self)})


@dataclass(frozen=True)
class TemporalWeights:
    values: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class TemporalLoss:
    loss: torch.Tensor
    per_anchor: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def build_candidate_bundle(groups: tuple[MaskGroup, ...],
                           temporal: int, height: int, width: int,
                           settings: KernelSettings, *, pair_valid: torch.Tensor,
                           key_valid: torch.Tensor, pair_reasons: torch.Tensor,
                           key_reasons: torch.Tensor, sample_ids: torch.Tensor | None = None,
                           sample_padding: torch.Tensor | None = None) -> CandidateBundle:
    """Join support to verified clip-local validity, without evidence imports.

    The caller supplies pair validity/reason flags [B,T,T] from full-span J(t,u)
    and key validity/reason flags [B,T] from internal transition 2t. Both must
    come from the same verified grid. Full-span pair validity must be symmetric.
    All true pairs (including ones outside
    the configured offsets) must imply valid endpoint keys. The bundle owns
    copies of these arrays; no grid construction or Python span loop occurs
    in the training path. Diagonal pair entries must be false/zero.
    """
    settings.validate()
    if tuple(g.name for g in groups) != GROUPS:
        raise ValueError("small and large groups must be supplied in order")
    supports = tuple(group_support(g, temporal, height, width) for g in groups)
    batch = supports[0].complete.shape[0]
    device = supports[0].complete.device
    if any(s.complete.shape != (batch, temporal) or s.complete.device != device
           for s in supports):
        raise ValueError("group batch, device or timestamp mismatch")
    for name, value, shape, dtype in (
            ("pair validity", pair_valid, (batch, temporal, temporal), torch.bool),
            ("key validity", key_valid, (batch, temporal), torch.bool),
            ("pair reasons", pair_reasons, (batch, temporal, temporal), torch.long),
            ("key reasons", key_reasons, (batch, temporal), torch.long)):
        if value.shape != shape or value.dtype != dtype or value.device != device:
            raise ValueError(f"{name} shape/dtype/device mismatch")
    if bool(pair_valid.diagonal(dim1=1, dim2=2).any()):
        raise ValueError("self-pair validity is forbidden")
    if not torch.equal(pair_valid, pair_valid.transpose(1, 2)):
        raise ValueError("full-span pair validity must be symmetric")
    if (bool(pair_reasons.diagonal(dim1=1, dim2=2).any())
            or bool(((pair_reasons < 0) | (pair_reasons > 127)).any())
            or bool(((key_reasons < 0) | (key_reasons > 127)).any())
            or bool((pair_reasons[pair_valid] != 0).any())
            or bool((key_reasons[key_valid] != 0).any())):
        raise ValueError("invalid validity reason flags")
    sample_ids = torch.arange(batch, device=device) if sample_ids is None else sample_ids
    sample_padding = (torch.zeros(batch, dtype=torch.bool, device=device)
                      if sample_padding is None else sample_padding)
    if sample_ids.shape != (batch,) or sample_ids.dtype != torch.long or sample_ids.device != device:
        raise ValueError("invalid sample IDs")
    if (sample_padding.shape != (batch,) or sample_padding.dtype != torch.bool
            or sample_padding.device != device):
        raise ValueError("invalid sample padding")
    pair, key = pair_valid, key_valid
    times = torch.arange(temporal, device=device)
    negative_times = times[:, None] + torch.tensor(settings.offsets, device=device)
    in_bounds = (negative_times >= 0) & (negative_times < temporal)
    safe_times = negative_times.clamp(0, temporal - 1)
    eligibilities, validity, causes = [], [], []
    for support in supports:
        eligible, dense_valid = candidates(support, settings.offsets, settings.v_min,
                                           pair_valid=pair, key_valid=key)
        eligible = eligible & ~sample_padding[:, None]
        validity.append(dense_valid.gather(2, safe_times[None].expand(batch, -1, -1))
                        & in_bounds[None] & eligible[:, :, None])
        eligibilities.append(eligible)
        complete = support.complete.any(dim=1)
        passes_support = support.support_size >= settings.v_min
        positive = (support.complete & key).any(dim=1)
        cause = torch.zeros(batch, dtype=torch.long, device=device)
        cause[~eligible.any(dim=1)] = 4
        cause[~positive] = 3
        cause[~passes_support] = 2
        cause[~complete] = 1
        cause[sample_padding] = EMPTY_CAUSES.index("padding")
        causes.append(cause)
    shape = (batch, len(GROUPS), temporal, len(settings.offsets))
    indices = torch.stack((sample_ids[:, None, None, None].expand(shape),
                           torch.arange(2, device=device)[None, :, None, None].expand(shape),
                           times[None, None, :, None].expand(shape),
                           torch.where(in_bounds, negative_times, -1)[None, None].expand(shape)), dim=-1)
    return CandidateBundle(
        indices.clone(), torch.stack(validity, dim=1), in_bounds[None, None].expand(shape).clone(),
        torch.stack(eligibilities, dim=1), torch.stack([s.complete for s in supports], dim=1),
        torch.stack([s.partial for s in supports], dim=1),
        torch.stack([s.visible for s in supports], dim=1),
        torch.stack([s.support_size for s in supports], dim=1),
        pair[:, None].expand(-1, 2, -1, -1).clone(), key[:, None].expand(-1, 2, -1).clone(),
        pair_reasons[:, None].expand(-1, 2, -1, -1).clone(),
        key_reasons[:, None].expand(-1, 2, -1).clone(), torch.stack(causes, dim=1),
        sample_padding.clone(), settings)


def _provider_weights(condition, bundle: CandidateBundle) -> TemporalWeights:
    view = bundle.copy()
    snapshot = view.copy()
    condition.prepare_clip(view)
    weights = condition.weights(view)
    for field in fields(view):
        current, original = getattr(view, field.name), getattr(snapshot, field.name)
        if isinstance(current, torch.Tensor) and not torch.equal(current, original):
            raise ValueError(f"provider mutated candidate metadata: {field.name}")
    return weights


def _validate_weights(weights: TemporalWeights, bundle: CandidateBundle) -> None:
    if not isinstance(weights, TemporalWeights):
        raise ValueError("provider must return indexed TemporalWeights")
    w, indices = weights.values, weights.indices
    if not isinstance(w, torch.Tensor) or w.shape != bundle.valid.shape:
        raise ValueError("provider weight shape mismatch")
    if not isinstance(indices, torch.Tensor):
        raise ValueError("provider index alignment mismatch")
    if w.device != bundle.valid.device or indices.device != bundle.indices.device:
        raise ValueError("provider weight/index device mismatch")
    if indices.dtype != torch.long or not torch.equal(indices, bundle.indices):
        raise ValueError("provider index alignment mismatch")
    if w.requires_grad:
        raise ValueError("provider weights must not require gradients")
    if not w.is_floating_point() or not bool(torch.isfinite(w).all()):
        raise ValueError("provider weights must be finite floating-point values")
    if bool(((w < 0) | (w > 1)).any()):
        raise ValueError("provider weights must be in [0,1]")
    if bool((w[~bundle.valid] != 0).any()):
        raise ValueError("provider weights must be zero on invalid pairs")


def project_temporal_tokens(student_features: dict[str, torch.Tensor],
                            groups: tuple[MaskGroup, ...], teacher_tokens: torch.Tensor,
                            query_head: nn.Module, ema_head: nn.Module,
                            bundle: CandidateBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool in FP32, then project once per sample/group/timestamp this forward."""
    if tuple(g.name for g in groups) != GROUPS or set(student_features) != set(GROUPS):
        raise ValueError("both groups must be supplied")
    queries, keys = [], []
    with torch.autocast(device_type=teacher_tokens.device.type, enabled=False):
        for g, group in enumerate(groups):
            support = GroupSupport(bundle.visible[:, g], bundle.complete[:, g],
                                   bundle.partial[:, g], bundle.support_size[:, g])
            features = student_features[group.name]
            if features.device != bundle.valid.device or teacher_tokens.device != bundle.valid.device:
                raise ValueError("token/candidate device mismatch")
            if not bool(torch.isfinite(features).all()):
                raise ValueError("MISSING_NONFINITE_FEATURES: nonfinite student tokens")
            student = pool_student(features.float(), group.context_indices, support)
            projected = query_head(student)
            if not bool(torch.isfinite(projected).all()):
                raise ValueError("MISSING_NONFINITE_FEATURES: nonfinite query projection")
            queries.append(projected)
            with torch.no_grad():
                # Empty supports are padding, with a finite zero pool rather than 0/0.
                safe_support = GroupSupport(support.visible, support.complete, support.partial,
                                            support.support_size.clamp_min(1))
                teacher = pool_teacher(teacher_tokens.detach().float(), safe_support)
                if teacher.shape[:2] != bundle.key_valid[:, g].shape:
                    raise ValueError("teacher timestamp mismatch")
                projected_key = ema_head(teacher)
                if not bool(torch.isfinite(projected_key).all()):
                    raise ValueError("MISSING_NONFINITE_FEATURES: nonfinite EMA projection")
                keys.append(projected_key.detach())
        # Explicitly connect every trainable head parameter, including parameters
        # a caller's head might not exercise on an empty input. Never reset state.
        zero = sum((p.sum() * 0 for p in query_head.parameters() if p.requires_grad),
                   torch.zeros((), device=teacher_tokens.device))
        return torch.stack(queries, dim=1) + zero, torch.stack(keys, dim=1)


def shared_diagnostics(bundle: CandidateBundle, weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Detached reporting counts; fractions are numerator/denominator pairs."""
    with torch.no_grad():
        valid, eligible = bundle.valid, bundle.eligible
        w = weights.detach().float()
        mass = w.sum(dim=-1)
        threshold = LOW_MASS_REPORTING_THRESHOLD
        considered = (bundle.in_bounds & bundle.complete[..., None]
                      & (bundle.support_size >= bundle.settings.v_min)[..., None, None]
                      & ~bundle.sample_padding[:, None, None, None])
        u = bundle.indices[..., 3].clamp_min(0)
        grid_valid = bundle.pair_valid.gather(3, u)
        result = {
            "eligible_anchor_counts": eligible.sum(dim=-1),
            "empty_causes": bundle.empty_causes.clone(),
            "invalid_positive_anchor_counts": (bundle.complete & ~bundle.key_valid
                & (bundle.support_size >= bundle.settings.v_min)[..., None]
                & ~bundle.sample_padding[:, None, None]).sum(dim=-1),
            "invalid_candidate_counts": (considered & ~grid_valid).sum(dim=(-1, -2)),
            "padding_candidate_counts": (~bundle.in_bounds
                | bundle.sample_padding[:, None, None, None]).sum(dim=(-1, -2)),
            "ineligible_anchor_counts": (~eligible).sum(dim=-1),
            "geometric_empty_counts": (bundle.empty_causes == 1).long(),
            "padding_empty_counts": (bundle.empty_causes == EMPTY_CAUSES.index("padding")).long(),
            "validity_empty_counts": ((bundle.empty_causes == 3) | (bundle.empty_causes == 4)).long(),
            "sample_padding": bundle.sample_padding.clone(),
            "key_reasons": bundle.key_reasons.clone(),
            "pair_reasons": bundle.pair_reasons.clone(),
            "pair_weight_denominator": valid.sum(dim=(-1, -2)),
            "zero_pair_weight_numerator": (valid & (w == 0)).sum(dim=(-1, -2)),
            "low_positive_pair_weight_numerator": (valid & (w > 0) & (w < threshold)).sum(dim=(-1, -2)),
            "anchor_mass_denominator": eligible.sum(dim=-1),
            "zero_anchor_mass_numerator": (eligible & (mass == 0)).sum(dim=-1),
            "low_positive_anchor_mass_numerator": (eligible & (mass > 0) & (mass < threshold)).sum(dim=-1),
            # Descriptive per-offset/per-timestamp coverage (telemetry).
            "valid_pairs_per_offset": valid.sum(dim=(0, 1, 2)),
            "weight_mass_per_offset": (w * valid).sum(dim=(0, 1, 2)),
            "eligible_per_timestamp": eligible.sum(dim=0),
        }
        return {name: value.detach() for name, value in result.items()}


def weighted_temporal_loss(queries: torch.Tensor, keys: torch.Tensor,
                           bundle: CandidateBundle, condition, *,
                           global_batch: int | None = None, world_size: int = 1,
                           diagnostics: bool = True) -> TemporalLoss:
    """One validated arithmetic path for U and arbitrary synthetic/E/M weights."""
    weights = _provider_weights(condition, bundle)
    _validate_weights(weights, bundle)
    expected = bundle.eligible.shape
    if (queries.ndim != 4 or queries.shape[:3] != expected or keys.shape != queries.shape
            or queries.shape[-1] < 1 or queries.device != bundle.valid.device
            or keys.device != bundle.valid.device):
        raise ValueError("representation shape/device mismatch")
    with torch.autocast(device_type=queries.device.type, enabled=False):
        q_valid = (bundle.complete & bundle.key_valid
                   & (bundle.support_size >= bundle.settings.v_min)[..., None]
                   & ~bundle.sample_padding[:, None, None])
        k_valid = (bundle.key_valid & (bundle.support_size > 0)[..., None]
                   & ~bundle.sample_padding[:, None, None])
        q, k = queries.float(), keys.detach().float()
        if not bool(torch.isfinite(q[q_valid]).all() and torch.isfinite(k[k_valid]).all()):
            raise ValueError("MISSING_NONFINITE_FEATURES: nonfinite valid representations")
        # Sanitize unused/padded slots BEFORE normalization or similarity.
        q = F.normalize(torch.where(q_valid[..., None], q, 0), dim=-1)
        k = F.normalize(torch.where(k_valid[..., None], k, 0), dim=-1)
        if not bool(torch.isfinite(q).all() and torch.isfinite(k).all()):
            raise ValueError("nonfinite normalized representations")
        u = bundle.indices[..., 3].clamp_min(0)
        b = torch.arange(q.shape[0], device=q.device)[:, None, None, None]
        g = torch.arange(q.shape[1], device=q.device)[None, :, None, None]
        negative_keys = k[b, g, u]
        positive = (q * k).sum(dim=-1)
        similarities = (q[..., None, :] * negative_keys).sum(dim=-1)
        rel_logit = (similarities - positive[..., None]) / bundle.settings.tau
        if not bool(torch.isfinite(rel_logit[bundle.valid]).all()):
            raise ValueError("nonfinite valid logits")
        w = weights.values.float()
        active = bundle.valid & (w > 0)
        safe_w = torch.where(active, w, torch.ones_like(w))
        negative_logits = torch.where(active, rel_logit + torch.log(safe_w),
                                      torch.full_like(rel_logit, -torch.inf))
        positive_zero = torch.zeros_like(positive[..., None])
        per_anchor = torch.logsumexp(torch.cat((positive_zero, negative_logits), dim=-1), dim=-1)
        loss = temporal_sample_mean({name: per_anchor[:, g] for g, name in enumerate(GROUPS)},
                                    {name: bundle.eligible[:, g] for g, name in enumerate(GROUPS)},
                                    alpha=dict(zip(GROUPS, bundle.settings.alpha)),
                                    global_batch=global_batch, world_size=world_size)
    return TemporalLoss(loss, per_anchor,
                        shared_diagnostics(bundle, w) if diagnostics else {})
