"""Latent-prediction V-JEPA-framework model with a COMEDIAN-compatible spatial encoder."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from bpjepa.samplers import RandomBlockSampler, TargetSampler


def drop_path(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0 or not training:
        return x
    keep = 1 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x * x.new_empty(shape).bernoulli_(keep) / keep


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)).mul(self.scale).softmax(dim=-1)
        return self.proj((attention @ value).transpose(1, 2).reshape(batch, tokens, dim))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path_probability: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path_probability = drop_path_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + drop_path(self.attn(self.norm1(x)), self.drop_path_probability, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.drop_path_probability, self.training)


class PatchEmbed3D(nn.Module):
    def __init__(self, image_size: int, patch_size: int, tube_size: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.img_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.tube_size = tube_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            3,
            dim,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size),
        )
        self.norm = nn.Identity()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError("clips must have shape (batch, 3, frames, height, width)")
        if clips.shape[2] % self.tube_size:
            raise ValueError("frame count must be divisible by tube_size")
        if tuple(clips.shape[-2:]) != self.img_size:
            raise ValueError(f"clip spatial size must be {self.img_size}")
        embedded = self.proj(clips)
        return embedded.permute(0, 2, 3, 4, 1).contiguous()


class ComedianViTTiny(nn.Module):
    """ViT whose exported parameter names and pooled output match COMEDIAN's trunk."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        tube_size: int = 2,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(image_size, patch_size, tube_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        probabilities = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio, True, probabilities[index])
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._initialize()

    @property
    def spatial_grid(self) -> tuple[int, int]:
        side = self.image_size // self.patch_size
        return side, side

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(
        self, clips: torch.Tensor, spatial_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        patches = self.patch_embed(clips)
        batch, temporal, height, width, dim = patches.shape
        patches = patches.reshape(batch, temporal, height * width, dim)
        positions = self.pos_embed[:, 1:].expand(batch, -1, -1)
        if spatial_mask is not None:
            if spatial_mask.shape != (batch, height * width):
                raise ValueError("spatial mask shape does not match the patch grid")
            counts = spatial_mask.sum(dim=1)
            if not torch.equal(counts, counts[:1].expand_as(counts)):
                raise ValueError("all samples must retain the same number of context tokens")
            indices = spatial_mask.nonzero(as_tuple=False)[:, 1].reshape(batch, -1)
            gather = indices[:, None, :, None].expand(-1, temporal, -1, dim)
            patches = patches.gather(2, gather)
            positions = positions.gather(1, indices[:, :, None].expand(-1, -1, dim))
        tokens = patches + positions[:, None]
        cls = self.cls_token.expand(batch, temporal, -1, -1) + self.pos_embed[:, :1, None]
        tokens = torch.cat((cls, tokens), dim=2).reshape(batch * temporal, -1, dim)
        return self.norm(self.blocks(tokens)).reshape(batch, temporal, -1, dim)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(clips)
        return tokens[:, :, 0].reshape(-1, self.embed_dim)


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self.context_proj = nn.Linear(dim, hidden_dim)
        self.position_proj = nn.Linear(dim, hidden_dim)
        self.layers = nn.Sequential(
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, context: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        summary = context.mean(dim=(1, 2))[:, None, None]
        return self.layers(self.context_proj(summary) + self.position_proj(target_positions))


class BaseVJEPA(nn.Module):
    def __init__(
        self,
        encoder: ComedianViTTiny | None = None,
        sampler: TargetSampler | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = encoder or ComedianViTTiny()
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = Predictor(self.context_encoder.embed_dim)
        self.sampler = sampler or RandomBlockSampler()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        for target, context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(context, alpha=1 - momentum)

    def forward(
        self, clips: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        if clips.ndim != 5:
            raise ValueError("clips must have shape (batch, channels, frames, height, width)")
        batch, _, frames, _, _ = clips.shape
        if frames % self.context_encoder.tube_size:
            raise ValueError("frame count must be divisible by encoder tube_size")
        temporal = frames // self.context_encoder.tube_size
        height, width = self.context_encoder.spatial_grid
        masks = self.sampler.sample(batch, temporal, height, width, generator=generator)
        expected_context = (batch, temporal * height * width)
        expected_targets = (batch, self.sampler.num_targets, temporal * height * width)
        if masks.context.shape != expected_context or masks.targets.shape != expected_targets:
            raise ValueError(
                "target sampler returned invalid mask shapes: "
                f"{tuple(masks.context.shape)}, {tuple(masks.targets.shape)}"
            )
        masks = type(masks)(masks.context.to(clips.device), masks.targets.to(clips.device))
        spatial_context = masks.context.reshape(batch, temporal, height * width)[:, 0]
        context = self.context_encoder.forward_tokens(clips, spatial_context)[:, :, 1:]
        with torch.no_grad():
            target = self.target_encoder.forward_tokens(clips)[:, :, 1:].reshape(
                batch, temporal * height * width, -1
            )
        position = self.context_encoder.pos_embed[:, 1:].expand(batch, -1, -1)
        position = position[:, None].expand(-1, temporal, -1, -1).reshape(
            batch, temporal * height * width, -1
        )
        target_indices = masks.targets.nonzero(as_tuple=False)[:, 2].reshape(
            batch, self.sampler.num_targets, -1
        )
        dim = target.shape[-1]
        gather = target_indices[:, :, :, None].expand(-1, -1, -1, dim)
        targets = target[:, None].expand(-1, self.sampler.num_targets, -1, -1).gather(2, gather)
        positions = position[:, None].expand(-1, self.sampler.num_targets, -1, -1).gather(2, gather)
        predictions = self.predictor(context, positions)
        normalized_predictions = F.layer_norm(predictions, (dim,))
        normalized_targets = F.layer_norm(targets.detach(), (dim,))
        loss = F.smooth_l1_loss(normalized_predictions, normalized_targets)
        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets.detach(),
            "context_features": context,
            "context_mask": masks.context,
            "target_masks": masks.targets,
        }


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 0.9999) -> float:
    if total_steps < 1 or not 0 <= step <= total_steps:
        raise ValueError("invalid EMA schedule position")
    progress = step / total_steps
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2
