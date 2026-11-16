"""Boundary prediction objective (L_B) on the small mask group; added to the temporal-contrast path.

Pooled pre-projection student features z_t (the exact pool_student call project_temporal_tokens makes),
AuxHead(P: 192->32, a, b), squared error reduced exactly like the TC loss (temporal_sample_mean), and
lambda_b on the same pre-update ramp counter. Targets and same-shot flags come from a per-row table; training
is a lookup. MMAG (change magnitude) stays implemented because the original code path carries it, but it
was never trained for the paper.
"""
from __future__ import annotations

import torch
from torch import nn

from bpjepa.jepa.temporal_vjepa import StreamLog, TemporalVJEPA, finite_gradient
from bpjepa.jepa.support import GroupSupport, pool_student, temporal_sample_mean

AUX_KINDS = ("EB", "MMAG")
SIDES = (-2, -1, 1, 2)  # side_ok column order


class AuxHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int, target_mean: float):
        super().__init__()
        self.P = nn.Linear(dim, proj_dim, device="cpu", dtype=torch.float32)
        self.a = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(target_mean), dtype=torch.float32))


def build_aux_head(seed: int, target_mean: float, *, dim: int = 192, proj_dim: int = 32, device="cpu"):
    """CPU-forked construction (no main-stream RNG): P trunc-normal 0.02, zero bias; a=1, b=target mean."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("aux_head_seed must be a nonnegative integer")
    if not torch.isfinite(torch.tensor(float(target_mean))):
        raise ValueError("target_mean must be finite")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = AuxHead(dim, proj_dim, target_mean)
        nn.init.trunc_normal_(head.P.weight, std=0.02)
        nn.init.zeros_(head.P.bias)
    return head.to(device)


def _shift(x: torch.Tensor, offset: int) -> torch.Tensor:
    """out[:, t] = x[:, t + offset], zero where t + offset is outside [0, T)."""
    out = torch.zeros_like(x)
    temporal = x.shape[1]
    if offset >= 0:
        out[:, :temporal - offset] = x[:, offset:]
    else:
        out[:, -offset:] = x[:, :temporal + offset]
    return out


def aux_predictions(kind: str, head: AuxHead, z: torch.Tensor, side_ok: torch.Tensor,
                    dz_ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pred [B,T] and head-rule availability [B,T].

    EB: a*||mean_same-shot(Pz_{t-2},Pz_{t-1}) - mean_same-shot(Pz_{t+1},Pz_{t+2})|| + b, >=1 per side.
    MMAG: a*mean_available(||P dz_{t-1}||, ||P dz_t||) + b, dz_t = z_{t+1} - z_t, >=1 available.
    """
    pz = head.P(z)
    if kind == "EB":
        ok = side_ok.float()
        before = (_shift(pz, -2) * ok[..., 0:1] + _shift(pz, -1) * ok[..., 1:2])
        after = (_shift(pz, 1) * ok[..., 2:3] + _shift(pz, 2) * ok[..., 3:4])
        n_before, n_after = ok[..., 0] + ok[..., 1], ok[..., 2] + ok[..., 3]
        available = (n_before > 0) & (n_after > 0)
        diff = before / n_before.clamp_min(1)[..., None] - after / n_after.clamp_min(1)[..., None]
        return head.a * torch.linalg.vector_norm(diff, dim=-1) + head.b, available
    if kind == "MMAG":
        dz = _shift(z, 1) - z                              # dz_t, valid for t <= T-2
        mag = torch.linalg.vector_norm(head.P(dz) - head.P.bias, dim=-1)  # ||P dz||: linear part only
        prev_ok, cur_ok = _shift(dz_ok, -1).bool(), dz_ok.bool()
        total = _shift(mag, -1) * prev_ok + mag * cur_ok
        count = prev_ok.float() + cur_ok.float()
        return head.a * total / count.clamp_min(1) + head.b, count > 0
    raise ValueError("aux kind must be EB or MMAG")


def complete_window(complete: torch.Tensor) -> torch.Tensor:
    """t eligible only if tubelets t-2..t+2 are all complete (evidence anchor rule)."""
    window = torch.ones_like(complete)
    for offset in (-2, -1, 0, 1, 2):
        window &= _shift(complete, offset)
    return window


def aux_loss(pred, available, y, supported, window, padding):
    """U's reduction (temporal_sample_mean, same defaults as the hook's temporal loss):
    squared error averaged over each sample's eligible anchors, then over samples.
    A sample with no eligible anchor contributes exact 0 and is counted. Small
    group only: alpha {small: 1, large: 0} with an all-ineligible large term."""
    eligible = available & supported & window & ~padding[:, None]
    target = torch.where(eligible, y, torch.zeros_like(y))
    squared = torch.where(eligible, pred - target, torch.zeros_like(pred)) ** 2
    none = torch.zeros_like(eligible)
    loss = temporal_sample_mean({"small": squared, "large": torch.zeros_like(squared)},
                                {"small": eligible, "large": none}, alpha={"small": 1.0, "large": 0.0})
    return loss, int(eligible.sum())


class AuxTable:
    """Frozen per-row targets; row lookup only."""
    def __init__(self, kind, y, supported, side_ok, dz_ok, target_mean):
        if kind not in AUX_KINDS:
            raise ValueError("aux kind must be EB or MMAG")
        rows, temporal = y.shape
        if (y.dtype != torch.float32 or supported.shape != y.shape or supported.dtype != torch.bool
                or side_ok.shape != (rows, temporal, 4) or side_ok.dtype != torch.bool
                or dz_ok.shape != y.shape or dz_ok.dtype != torch.bool):
            raise ValueError("malformed aux target table")
        if (not bool(torch.isfinite(y[supported]).all()) or not bool(torch.isnan(y[~supported]).all())):
            raise ValueError("aux targets must be finite where supported and NaN elsewhere")
        self.kind, self.target_mean = kind, float(target_mean)
        self.y, self.supported, self.side_ok, self.dz_ok = y, supported, side_ok, dz_ok

    def lookup(self, row_ids: torch.Tensor, device):
        rows = row_ids.cpu()
        if bool(((rows < 0) | (rows >= self.y.shape[0])).any()):
            raise ValueError("row outside the frozen aux target table")
        return tuple(t[rows].to(device) for t in (self.y, self.supported, self.side_ok, self.dz_ok))


class EBTemporalVJEPA(TemporalVJEPA):
    def __init__(self, base, settings, *, aux_table: AuxTable, aux_head_seed: int,
                 lambda_b_final: float, proj_dim: int = 32, aux_observer=None, **kwargs):
        super().__init__(base, settings, **kwargs)  # query/EMA heads first: base param order unchanged
        if type(lambda_b_final) is not float or not 0 <= lambda_b_final < float("inf"):
            raise ValueError("lambda_b_final must be a finite nonnegative float")
        device = next(self.context_encoder.parameters()).device
        self.aux_head = build_aux_head(aux_head_seed, aux_table.target_mean,
                                       dim=self.context_encoder.embed_dim, proj_dim=proj_dim, device=device)
        self.aux_table, self.lambda_b_final, self.aux_observer = aux_table, lambda_b_final, aux_observer
        for parameter in self.aux_head.parameters():
            parameter.register_hook(finite_gradient)

    def forward(self, batch, *, generator=None):
        result = super().forward(batch, generator=generator)
        bundle = result["temporal_bundle"]
        small = next(g for g in result["mask_groups"] if g.name == "small")
        features = result["group_context_features"][0]
        support = GroupSupport(bundle.visible[:, 0], bundle.complete[:, 0],
                               bundle.partial[:, 0], bundle.support_size[:, 0])
        y, supported, side_ok, dz_ok = self.aux_table.lookup(batch.row_ids, features.device)
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = pool_student(features.float(), small.context_indices.to(features.device), support)
            pred, available = aux_predictions(self.aux_table.kind, self.aux_head, z, side_ok, dz_ok)
            loss_b, count = aux_loss(pred, available, y, supported,
                                     complete_window(bundle.complete[:, 0]), bundle.sample_padding)
            # Explicit zero dependency keeps every aux parameter connected on empty ranks.
            loss_b = loss_b + sum(p.sum() * 0 for p in self.aux_head.parameters())
        if not bool(torch.isfinite(loss_b)):
            raise FloatingPointError("nonfinite aux loss")
        update = int(self.temporal_completed_updates)
        coefficient_b = self.lambda_b_final * min(update / self.warmup_updates, 1)
        # Pilot/C8 fixture forwards (and monitoring) set model.observer = None so they never touch the production
        # logs; the aux stream follows the same rule (diagnostics run at multiples of the telemetry interval).
        if self.aux_observer is not None and self.observer is not None:
            self.aux_observer.log_aux(update, loss_b, coefficient_b, count)
        return {**result, "loss": result["loss"] + coefficient_b * loss_b,
                "aux_loss": loss_b, "aux_lambda": coefficient_b, "aux_eligible_rows": count}


class EBStreamLog(StreamLog):
    """StreamLog + aux loss records and a separate aux-head gradient norm."""
    def log_aux(self, update, loss, coefficient, count):
        if self.telemetry_interval is not None and update % self.telemetry_interval == 0:
            self._write("aux", {"update": update, "aux_loss": float(loss.detach()),
                                "aux_lambda": coefficient, "eligible_rows": count})

    def log_gradients(self, update, model):
        """Same record as StreamLog (total = clip_grad_norm_'s norm over model.parameters()),
        but base excludes BOTH heads, and the aux head is reported separately."""
        if update == self.last_gradient:
            return
        self.last_gradient = update
        query = {id(p) for p in model.query_head.parameters()}
        aux = {id(p) for p in model.aux_head.parameters()}
        grads = [(id(p), p.grad) for p in model.parameters() if p.grad is not None]

        def norm(selected):
            return float(torch.nn.utils.get_total_norm(selected, 2.0)) if selected else 0.0
        self._write("gradients", {"update": update, "total": norm([g for _, g in grads]),
                                  "base": norm([g for i, g in grads if i not in query and i not in aux]),
                                  "head": norm([g for i, g in grads if i in query]),
                                  "aux_head": norm([g for i, g in grads if i in aux])})
