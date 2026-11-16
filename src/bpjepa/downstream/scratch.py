"""Scratch reference: no pretrained state; records the fresh-initialization hashes."""
from __future__ import annotations

from .initialization import _spatial_target_state, aggregate_hash, tensor_sha256


def initialize_scratch(transformer, classifier, config):
    spatial = _spatial_target_state(transformer)
    groups = {
        "spatial": spatial,
        "temporal": {k: v for k, v in transformer.state_dict().items() if k not in spatial},
        "classifier": classifier.state_dict(),
    }
    if len(transformer.spatial_transformer.blocks) != 12 or len(transformer.temporal_transformer.blocks) != 4:
        raise RuntimeError("scratch architecture must have 12 spatial and 4 temporal blocks")
    return {
        "pass": True,
        "model_construction_seed": int(config["seed"]),
        "model_identity": "ViViT-Tiny/ViT-Tiny/12-spatial/4-temporal/17-class",
        "spatial_source": {"type": "scratch", "checkpoint_path": None},
        "initialization_hashes": {
            name: {"hash": aggregate_hash({k: tensor_sha256(v) for k, v in state.items()}),
                   "tensor_hashes": {k: tensor_sha256(v) for k, v in state.items()}}
            for name, state in groups.items()
        },
    }
