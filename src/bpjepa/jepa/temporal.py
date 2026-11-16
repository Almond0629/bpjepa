"""Isolated compact temporal predictor; spatial-per-tubelet encoder unchanged."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from bpjepa.jepa.model import ComedianViTTiny
from bpjepa.jepa.multimask import MultiMaskVJEPA, indexed_context
from bpjepa.samplers.vjepa_multimask import VJEPAMultiMaskSampler

TEMPORAL_ID = "compact_temporal_multimask_v1"
CONTROL_ID = "compact_pooled_multimask_control_v1"


def coordinates(indices, grid):
    """Time-major flattened index -> (tubelet, row, column); never renumber tokens."""
    temporal, height, width = grid
    if min(grid) < 1 or indices.dtype != torch.long or indices.ndim != 2:
        raise ValueError("indices must be a batch of int64 flattened grid indices")
    if indices.numel() == 0 or bool(((indices < 0) | (indices >= temporal * height * width)).any()):
        raise ValueError("empty or out-of-grid indices")
    return torch.stack((indices // (height * width), (indices // width) % height,
                        indices % width), dim=-1)


def positions(indices, grid, dim):
    """Fixed concatenated sin/cos codes, with disjoint channels for t/y/x."""
    if dim < 6 or dim % 6:
        raise ValueError("predictor width must be a positive multiple of six")
    coords = coordinates(indices, grid).float()
    frequencies = 10000.0 ** (-torch.arange(dim // 6, device=indices.device).float() / (dim // 6))
    phase = coords[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class TemporalPredictor(nn.Module):
    def __init__(self, dim, width=96, depth=2, num_heads=3, mlp_ratio=2):
        super().__init__()
        if width < 6 or width % 6 or num_heads < 1 or width % num_heads or depth < 1 or mlp_ratio <= 0:
            raise ValueError("invalid temporal predictor width/depth/heads/MLP ratio")
        self.width = width
        self.context_proj = nn.Linear(dim, width)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, num_heads, int(width * mlp_ratio),
                                       dropout=0, activation="gelu", batch_first=True,
                                       norm_first=True, layer_norm_eps=1e-6)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.output_proj = nn.Linear(width, dim)

    def forward(self, context, context_indices, target_indices, grid):
        if context.ndim != 3 or context.shape[:2] != context_indices.shape:
            raise ValueError("context features must match the retained indices")
        if target_indices.shape[0] != context.shape[0]:
            raise ValueError("target batch must match context batch")
        visible = self.context_proj(context)
        visible = visible + positions(context_indices, grid, self.width).to(visible.dtype)
        queries = self.query + positions(target_indices, grid, self.width).to(visible.dtype)
        tokens = torch.cat((visible, queries), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_proj(self.norm(tokens[:, context.shape[1]:]))


class TemporalMultiMaskVJEPA(MultiMaskVJEPA):
    """One teacher pass, exact visible indices, original normalized loss/group weights.

    No teacher features enter the predictor. Temporal interaction is in the
    predictor only; this is not a full V-JEPA reproduction.
    """
    def __init__(self, encoder=None, sampler=None, *, width=96, depth=2, num_heads=3, mlp_ratio=2):
        super().__init__(encoder, sampler)
        self.predictor = TemporalPredictor(self.context_encoder.embed_dim, width, depth,
                                           num_heads, mlp_ratio)

    def forward(self, clips, *, generator=None):
        if clips.ndim != 5 or clips.shape[2] % self.context_encoder.tube_size:
            raise ValueError("clips must be 5D with frames divisible by tube_size")
        batch = clips.shape[0]
        grid = (clips.shape[2] // self.context_encoder.tube_size, *self.context_encoder.spatial_grid)
        groups = self.sampler.sample(batch, *grid, generator=generator)
        with torch.no_grad():
            teacher = self.target_encoder.forward_tokens(clips)[:, :, 1:].reshape(batch, -1, self.context_encoder.embed_dim)
        dim = teacher.shape[-1]
        contexts, predictions, targets, losses = [], [], [], []
        for group in groups:
            ci, ti = group.context_indices.to(clips.device), group.target_indices.to(clips.device)
            coordinates(ci, grid)
            coordinates(ti, grid)
            context = indexed_context(self.context_encoder, clips, ci)
            target = teacher.gather(1, ti[:, :, None].expand(-1, -1, dim))
            prediction = self.predictor(context, ci, ti, grid)
            losses.append(F.smooth_l1_loss(F.layer_norm(prediction, (dim,)), F.layer_norm(target, (dim,))))
            contexts.append(context)
            predictions.append(prediction)
            targets.append(target)
        return {"loss": torch.stack(losses).mean(), "group_losses": torch.stack(losses),
                "predictions": predictions, "targets": targets,
                "context_features": torch.cat(contexts, dim=1),
                "group_context_features": contexts, "mask_groups": groups}


def build_model(spec):
    """Explicit new namespace: never fall back to a legacy architecture."""
    if set(spec) != {"model_id", "encoder", "predictor", "mask"}:
        raise ValueError("model spec requires exactly model_id/encoder/predictor/mask")
    if spec["model_id"] not in (CONTROL_ID, TEMPORAL_ID):
        raise ValueError("unsupported model_id; legacy resume is not supported")
    if set(spec["encoder"]) != {"image_size", "patch_size", "tube_size", "embed_dim", "depth", "num_heads", "mlp_ratio", "drop_path_rate"}:
        raise ValueError("encoder metadata must specify every architecture field")
    encoder = ComedianViTTiny(**spec["encoder"])
    sampler = VJEPAMultiMaskSampler(spec["mask"])
    if spec["model_id"] == CONTROL_ID:
        if spec["predictor"] != {"hidden_dim": 384}:
            raise ValueError("pooled control requires the unchanged 384-wide predictor")
        return MultiMaskVJEPA(encoder, sampler)
    if set(spec["predictor"]) != {"width", "depth", "num_heads", "mlp_ratio"}:
        raise ValueError("temporal predictor metadata must specify all dimensions")
    return TemporalMultiMaskVJEPA(encoder, sampler, **spec["predictor"])


def load_model(payload, expected_spec=None):
    """Reconstruct only versioned pilot checkpoints; no implicit encoder warm start."""
    if payload.get("format") != "temporal_conditioning_pilot_v1" or "model_spec" not in payload:
        raise ValueError("incompatible checkpoint: legacy resume/warm start is forbidden")
    spec = payload["model_spec"]
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("checkpoint/config architecture mismatch")
    model = build_model(spec)
    model.load_state_dict(payload["model"], strict=True)
    return model
