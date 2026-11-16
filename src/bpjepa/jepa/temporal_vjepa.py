"""Temporal-contrast model path shared by all four arms (R, BP-JEPA, TC, BP-JEPA+TC).

Validity is supplied explicitly. The model's state_dict/optimizer/checkpoint path owns the heads.
With lambda_final=0 the query head still exists and receives exactly zero gradient, as in the paper.
"""
from __future__ import annotations

import copy
import hashlib
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import default_collate
from torch.utils.data import Dataset

from bpjepa.jepa.conditions import UCondition
from bpjepa.jepa.kernel import (
    KernelSettings, build_candidate_bundle, project_temporal_tokens,
    weighted_temporal_loss,
)
from bpjepa.jepa.multimask import MultiMaskVJEPA
from bpjepa.jepa.full_data import PaddedDistributedEpochSampler


class PresentationSampler(PaddedDistributedEpochSampler):
    """Carry padding by global presentation position, never by row identity."""
    def __iter__(self):
        for local_position, row in enumerate(super().__iter__(), start=self.start_index):
            global_position = self.rank + local_position * self.num_replicas
            yield row, global_position >= self.dataset_size


class PresentationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, presentation):
        row, padding = presentation
        if type(padding) is not bool:
            raise ValueError("invalid presentation padding")
        clip, row_id, manifest = self.dataset[row]
        return clip, row_id, manifest, padding


@dataclass(frozen=True)
class ClipBatch:
    clips: torch.Tensor
    row_ids: torch.Tensor
    manifest_sha256: str
    sample_padding: torch.Tensor

    @property
    def shape(self):
        return self.clips.shape

    def to(self, device, *, non_blocking=False):
        return replace(self, clips=self.clips.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return replace(self, clips=self.clips.pin_memory())


def collate_metadata(rows):
    clips, ids, manifests, padding = zip(*rows)
    if len(set(manifests)) != 1:
        raise ValueError("batch contains inconsistent manifest identities")
    return ClipBatch(default_collate(clips), default_collate(ids), manifests[0], default_collate(padding))


@dataclass(frozen=True)
class ClipValidity:
    row_id: int
    manifest_sha256: str
    pair_valid: torch.Tensor
    key_valid: torch.Tensor
    pair_reasons: torch.Tensor
    key_reasons: torch.Tensor


def build_heads(head_seed: int, device="cpu"):
    if type(head_seed) is not int or not 0 <= head_seed < 2**63:
        raise ValueError("head_seed is required and must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(head_seed)
        head = nn.Sequential(nn.Linear(192, 768, device="cpu", dtype=torch.float32), nn.GELU(),
                             nn.Linear(768, 128, device="cpu", dtype=torch.float32))
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    ema = copy.deepcopy(head).requires_grad_(False)
    return head.to(device), ema.to(device)


def finite_gradient(gradient):
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("nonfinite temporal gradient")
    return gradient


class TemporalVJEPA(MultiMaskVJEPA):
    def __init__(self, base, settings: KernelSettings, *, head_seed: int,
                 validity_source: Callable, manifest_sha256: str,
                 lambda_final: float, warmup_updates: int, condition=None, observer=None):
        # Reuse initialized modules without running their constructors again.
        nn.Module.__init__(self)
        if not isinstance(base, MultiMaskVJEPA) or base.context_encoder.embed_dim != 192:
            raise ValueError("temporal hook requires the 192-dimensional multi-mask model")
        if not callable(validity_source) or not manifest_sha256:
            raise ValueError("explicit manifest-bound validity source is required")
        if not 0 <= lambda_final < float("inf") or type(warmup_updates) is not int or warmup_updates < 1:
            raise ValueError("invalid temporal coefficient/schedule")
        self.context_encoder = base.context_encoder
        self.target_encoder = base.target_encoder
        self.predictor = base.predictor
        self.sampler = base.sampler
        self._base_forward = type(base).forward
        device = next(base.parameters()).device
        self.query_head, self.ema_head = build_heads(head_seed, device)
        self.register_buffer("temporal_completed_updates", torch.zeros((), dtype=torch.long, device=device))
        self.settings = settings
        self.validity_source = validity_source
        self.manifest_sha256 = manifest_sha256
        self.lambda_final = lambda_final
        self.warmup_updates = warmup_updates
        self.condition = UCondition() if condition is None else condition
        self.observer = observer
        for parameter in self.query_head.parameters():
            parameter.register_hook(finite_gradient)

    def _validity(self, batch, temporal, device):
        if not isinstance(batch, ClipBatch) or batch.manifest_sha256 != self.manifest_sha256:
            raise ValueError("missing or inconsistent batch manifest identity")
        if batch.row_ids.shape != (batch.shape[0],) or batch.row_ids.dtype != torch.long:
            raise ValueError("invalid row IDs")
        records = []
        for row in batch.row_ids.tolist():
            record = self.validity_source(row, batch.manifest_sha256)
            if (not isinstance(record, ClipValidity) or record.row_id != row
                    or record.manifest_sha256 != batch.manifest_sha256):
                raise ValueError("unknown or manifest-inconsistent validity")
            records.append(record)
        arrays = {}
        for name, shape, dtype in (("pair_valid", (temporal, temporal), torch.bool),
                                   ("key_valid", (temporal,), torch.bool),
                                   ("pair_reasons", (temporal, temporal), torch.long),
                                   ("key_reasons", (temporal,), torch.long)):
            values = [getattr(record, name) for record in records]
            if any(not isinstance(v, torch.Tensor) or v.shape != shape or v.dtype != dtype for v in values):
                raise ValueError(f"inconsistent validity array: {name}")
            arrays[name] = torch.stack(values).to(device)
        if (batch.sample_padding.shape != (batch.shape[0],)
                or batch.sample_padding.dtype != torch.bool):
            raise ValueError("invalid presentation padding")
        arrays["sample_padding"] = batch.sample_padding.to(device)
        arrays["sample_ids"] = batch.row_ids.to(device)
        return arrays

    def forward(self, batch, *, generator=None):
        if not isinstance(batch, ClipBatch):
            raise ValueError("enabled temporal hook requires row-ID metadata")
        clips = batch.clips
        temporal = clips.shape[2] // self.context_encoder.tube_size
        validity = self._validity(batch, temporal, clips.device)
        teacher = []
        def capture_teacher(module, inputs, output):
            teacher.append(output.reshape(clips.shape[0], temporal, height * width + 1, -1)
                           [:, :, 1:].reshape(clips.shape[0], temporal * height * width, -1))
        height, width = self.context_encoder.spatial_grid
        handle = self.target_encoder.norm.register_forward_hook(capture_teacher)
        try:
            result = self._base_forward(self, clips, generator=generator)
        finally:
            handle.remove()
        if len(teacher) != 1:
            raise RuntimeError("temporal hook requires exactly one existing teacher pass")
        result["teacher_tokens"] = teacher[0]
        groups = tuple(replace(g, context_indices=g.context_indices.to(clips.device),
                               target_indices=g.target_indices.to(clips.device),
                               target_union=g.target_union.to(clips.device))
                       for g in result["mask_groups"])
        bundle = build_candidate_bundle(groups, temporal, height, width, self.settings, **validity)
        features = dict(zip((g.name for g in groups), result["group_context_features"]))
        queries, keys = project_temporal_tokens(features, groups, result["teacher_tokens"],
                                                self.query_head, self.ema_head, bundle)
        if queries.requires_grad:
            queries.register_hook(finite_gradient)
        for feature in features.values():
            if feature.requires_grad:
                feature.register_hook(finite_gradient)
        temporal_result = weighted_temporal_loss(queries, keys, bundle, self.condition)
        if not bool(torch.isfinite(temporal_result.loss)):
            raise FloatingPointError("nonfinite temporal loss")
        coefficient = self.lambda_final * min(int(self.temporal_completed_updates) / self.warmup_updates, 1)
        if self.observer is not None:  # read-only: no RNG, no tensor mutation
            if getattr(self.observer, "wants_telemetry", lambda update: True)(int(self.temporal_completed_updates)):
                temporal_result.diagnostics.update(self.condition.diagnostics(bundle))
                temporal_result.diagnostics["offsets"] = torch.tensor(self.settings.offsets)
            self.observer(int(self.temporal_completed_updates), batch, groups, temporal_result, coefficient)
        return {**result, "base_loss": result["loss"],
                "loss": result["loss"] + coefficient * temporal_result.loss,
                "temporal_loss": temporal_result.loss, "temporal_queries": queries,
                "temporal_keys": keys, "temporal_bundle": bundle, "temporal_lambda": coefficient}

    @torch.no_grad()
    def update_target(self, momentum):
        super().update_target(momentum)
        for target, query in zip(self.ema_head.parameters(), self.query_head.parameters(), strict=True):
            target.mul_(momentum).add_(query, alpha=1 - momentum)
        self.temporal_completed_updates.add_(1)


def check_temporal_gradients(model):
    for parameter in model.query_head.parameters():
        if parameter.grad is not None:
            finite_gradient(parameter.grad)


class StreamLog:
    """Per-rank JSONL: clip IDs and mask hashes for the first
    `stream_updates` optimizer updates, plus descriptive coverage telemetry
    every `telemetry_interval` updates. Reads detached values only."""
    def __init__(self, output_dir, *, stream_updates: int | None, telemetry_interval: int | None):
        """None disables that log (correctness-stage RunSpecs may leave them unset)."""
        stream_updates = 0 if stream_updates is None else stream_updates
        if type(stream_updates) is not int or stream_updates < 0:
            raise ValueError("stream_updates must be a nonnegative integer")
        if telemetry_interval is not None and (type(telemetry_interval) is not int or telemetry_interval < 1):
            raise ValueError("telemetry_interval must be a positive integer")
        self.output_dir = Path(output_dir)
        self.stream_updates, self.telemetry_interval = stream_updates, telemetry_interval
        self.segment = f"{os.getpid()}-{time.time_ns()}"  # distinguishes replayed updates after resume
        self.last_telemetry = self.last_gradient = None

    def log_gradients(self, update, model):
        """Per optimizer update (1-based), after unscale_ and before clipping:
        the total pre-clip norm over model.parameters() exactly as
        clip_grad_norm_ computes it, plus base-only and query-head-only norms."""
        if update == self.last_gradient:
            return
        self.last_gradient = update
        head = {id(p) for p in model.query_head.parameters()}
        grads = [(id(p) in head, p.grad) for p in model.parameters() if p.grad is not None]

        def norm(selected):
            return float(torch.nn.utils.get_total_norm(selected, 2.0)) if selected else 0.0
        self._write("gradients", {"update": update,
                                  "total": norm([g for _, g in grads]),
                                  "base": norm([g for h, g in grads if not h]),
                                  "head": norm([g for h, g in grads if h])})

    def _write(self, name, record):
        import json
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / f"{name}_rank{rank}.jsonl", "a") as handle:
            handle.write(json.dumps({"rank": rank, "segment": self.segment, **record}, sort_keys=True) + "\n")

    def wants_telemetry(self, update):
        return (self.telemetry_interval is not None and update % self.telemetry_interval == 0
                and update != self.last_telemetry)

    def __call__(self, update, batch, groups, result, coefficient):
        if update < self.stream_updates:
            def digest(t):
                return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            self._write("stream", {
                "update": update, "manifest_sha256": batch.manifest_sha256,
                "row_ids": batch.row_ids.tolist(), "padding": batch.sample_padding.tolist(),
                "masks": {g.name: {"context": digest(g.context_indices), "target": digest(g.target_indices)}
                          for g in groups}})
        if self.wants_telemetry(update):
            self.last_telemetry = update
            d = result.diagnostics
            self._write("telemetry", {"update": update, "lambda": coefficient,
                                      "offsets": d.get("offsets", torch.empty(0, dtype=torch.long)).tolist(),
                                      "temporal_loss": float(result.loss.detach()),
                                      **{k: d[k].tolist() for k in (
                                          "valid_pairs_per_offset", "weight_mass_per_offset",
                                          "eligible_per_timestamp", "empty_causes")}})
            if "state_121_weight_mass" in d:
                self._write("state_121", {"update": update,
                    "attribution": "first maximum-e transition in J; M uses source transition; descriptive",
                    "rank_aggregation": "equal-weight mean across ranks with matched local batches",
                    **{k: v.tolist() for k, v in d.items()
                       if k.startswith(("state_121", "reduction_weight_mass")) or k == "offsets"}})
