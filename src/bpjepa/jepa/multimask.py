"""Separate V-JEPA multi-mask geometry candidate on the COMEDIAN backbone."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from bpjepa.jepa.model import ComedianViTTiny, BaseVJEPA
from bpjepa.samplers.vjepa_multimask import VJEPAMultiMaskSampler


def indexed_context(encoder, clips, indices):
    """Encode exactly the retained flattened indices, including partial tubes.

    COMEDIAN attends spatially per tubelet. Bucket tubelets by retained count
    so the existing transformer can run without padding or exposed target tokens.
    """
    patches = encoder.patch_embed(clips)
    batch, temporal, height, width, dim = patches.shape
    spatial = height * width
    patches = patches.reshape(batch * temporal, spatial, dim)
    patches = patches + encoder.pos_embed[:, 1:]
    keep = torch.zeros(batch, temporal * spatial, dtype=torch.bool, device=clips.device)
    keep.scatter_(1, indices, True)
    keep = keep.reshape(batch * temporal, spatial)
    counts = keep.sum(1)
    encoded = patches.new_zeros(batch * temporal, spatial, dim)
    for count in counts.unique().tolist():
        if count == 0:
            continue
        rows = (counts == count).nonzero().flatten()
        columns = keep[rows].nonzero()[:, 1].reshape(len(rows), count)
        selected = patches[rows[:, None], columns]
        cls = (encoder.cls_token + encoder.pos_embed[:, :1]).expand(len(rows), -1, -1)
        tokens = encoder.norm(encoder.blocks(torch.cat((cls, selected), dim=1)))[:, 1:]
        encoded[rows[:, None], columns] = tokens
    return encoded.reshape(batch, temporal * spatial, dim).gather(
        1, indices[:, :, None].expand(-1, -1, dim))


class MultiMaskVJEPA(BaseVJEPA):
    """Two context passes, one shared teacher pass, equally weighted group losses.

    Preserves the baseline predictor and normalized Smooth-L1 objective; this is
    an upstream mask-semantics experiment, not a full V-JEPA architecture port.
    """
    def __init__(self, encoder: ComedianViTTiny | None = None, sampler=None):
        super().__init__(encoder, sampler or VJEPAMultiMaskSampler())

    def forward(self, clips, *, generator=None):
        if clips.ndim != 5 or clips.shape[2] % self.context_encoder.tube_size:
            raise ValueError("clips must be 5D with frames divisible by tube_size")
        batch = clips.shape[0]
        temporal = clips.shape[2] // self.context_encoder.tube_size
        height, width = self.context_encoder.spatial_grid
        groups = self.sampler.sample(batch, temporal, height, width, generator=generator)
        with torch.no_grad():
            teacher = self.target_encoder.forward_tokens(clips)[:, :, 1:].reshape(
                batch, temporal * height * width, -1)
        dim = teacher.shape[-1]
        position = self.context_encoder.pos_embed[:, 1:].expand(batch, -1, -1)
        position = position[:, None].expand(-1, temporal, -1, -1).reshape(batch, -1, dim)
        contexts, predictions, targets, losses = [], [], [], []
        for group in groups:
            ci = group.context_indices.to(clips.device)
            ti = group.target_indices.to(clips.device)
            context = indexed_context(self.context_encoder, clips, ci)
            gather = ti[:, :, None].expand(-1, -1, dim)
            target = teacher.gather(1, gather)
            prediction = self.predictor(context[:, None], position.gather(1, gather)[:, None])[:, 0]
            losses.append(F.smooth_l1_loss(F.layer_norm(prediction, (dim,)),
                                          F.layer_norm(target, (dim,))))
            contexts.append(context)
            predictions.append(prediction)
            targets.append(target)
        return {"loss": torch.stack(losses).mean(), "group_losses": torch.stack(losses),
                "predictions": predictions, "targets": targets,
                "context_features": torch.cat(contexts, dim=1),
                "group_context_features": contexts, "mask_groups": groups}
