"""Weight-provider boundary; U uses the shared weighted kernel unchanged."""
from typing import Protocol

import torch

from bpjepa.jepa.kernel import CandidateBundle, TemporalWeights


class TemporalCondition(Protocol):
    def prepare_clip(self, bundle: CandidateBundle) -> None: ...
    def weights(self, bundle: CandidateBundle) -> TemporalWeights: ...
    def diagnostics(self, bundle: CandidateBundle) -> dict: ...


class UCondition:
    kind = "U"

    def prepare_clip(self, bundle: CandidateBundle) -> None:
        pass

    def weights(self, bundle: CandidateBundle) -> TemporalWeights:
        return TemporalWeights(bundle.valid.float(), bundle.indices.clone())

    def diagnostics(self, bundle: CandidateBundle) -> dict:
        return {}


class PairWeightTableCondition:
    """E/M: lookup of a frozen per-row q table [rows, anchors, signed offsets].

    The table keeps validity as a separate mask (q is NaN where invalid). Every
    runtime-valid pair must be table-valid with finite q in [0, 1]; weights
    outside bundle.valid are exactly zero. No evidence is read at train time.
    """
    def __init__(self, kind: str, q: torch.Tensor, table_valid: torch.Tensor,
                 offsets: tuple[int, ...], *, state_121: torch.Tensor | None = None):
        if kind not in ("E", "M"):
            raise ValueError("table conditions are E or M")
        if (q.ndim != 3 or q.dtype != torch.float32 or table_valid.shape != q.shape
                or table_valid.dtype != torch.bool or q.shape[2] != len(offsets)
                or len(set(offsets)) != len(offsets) or q.requires_grad
                or q.device != table_valid.device or min(q.shape[:2]) < 1
                or not offsets or any(type(d) is not int or d == 0 for d in offsets)):
            raise ValueError("malformed q table")
        finite = q[table_valid]
        if (not bool(torch.isfinite(finite).all()) or bool(((finite < 0) | (finite > 1)).any())
                or not bool(torch.isnan(q[~table_valid]).all())):
            raise ValueError("q table must be finite in [0,1] on valid pairs and NaN elsewhere")
        self.kind, self.offsets = kind, tuple(offsets)
        self._q, self._table_valid = q.detach().cpu().clone(), table_valid.cpu().clone()
        if state_121 is not None and (state_121.shape != q.shape or state_121.dtype != torch.bool
                                     or bool((state_121.cpu() & ~self._table_valid).any())):
            raise ValueError("malformed state attribution table")
        self._state_121 = None if state_121 is None else state_121.cpu().clone()

    def prepare_clip(self, bundle: CandidateBundle) -> None:
        pass

    def weights(self, bundle: CandidateBundle) -> TemporalWeights:
        if any(d not in self.offsets for d in bundle.settings.offsets):
            raise ValueError("q table lacks a configured offset")
        indices = bundle.indices.cpu()
        rows, anchors = indices[..., 0], indices[..., 2]
        if (bool(((rows < 0) | (rows >= self._q.shape[0])).any())
                or bool(((anchors < 0) | (anchors >= self._q.shape[1])).any())):
            raise ValueError("row or anchor outside the frozen q table")
        columns = torch.tensor([self.offsets.index(d) for d in bundle.settings.offsets]).expand_as(rows)
        valid = bundle.valid.cpu()
        if bool((valid & ~self._table_valid[rows, anchors, columns]).any()):
            raise ValueError("runtime-valid pair is invalid in the frozen q table")
        w = torch.where(valid, self._q[rows, anchors, columns], torch.zeros(()))
        return TemporalWeights(w.to(bundle.valid.device), bundle.indices.clone())

    def diagnostics(self, bundle: CandidateBundle) -> dict:
        if self._state_121 is None:
            return {}
        with torch.no_grad():
            indices = bundle.indices.cpu()
            columns = torch.tensor([self.offsets.index(d) for d in bundle.settings.offsets]).expand_as(indices[..., 0])
            state = self._state_121[indices[..., 0], indices[..., 2], columns].to(bundle.valid.device)
            weights = self.weights(bundle).values
            counts = bundle.eligible.sum(dim=-1).clamp_min(1)
            alpha = torch.tensor(bundle.settings.alpha, device=weights.device)[None, :, None, None]
            mass = weights * bundle.valid * alpha / counts[..., None, None] / weights.shape[0]
            overall = mass.sum(dim=(0, 1, 2))
            attributed = (mass * state).sum(dim=(0, 1, 2))
            return {"reduction_weight_mass_per_offset": overall,
                    "state_121_weight_mass_per_offset": attributed,
                    "reduction_weight_mass": overall.sum(),
                    "state_121_weight_mass": attributed.sum()}
