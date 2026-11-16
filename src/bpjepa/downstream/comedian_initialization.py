"""Strict COMEDIAN Stage-1 spatial-only initialization for direct 100F fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .initialization import (
    SpatialInitSpottingModel,
    _spatial_target_state,
    _torch_load,
    aggregate_hash,
    sha256,
    source_key_for_target,
    state_hashes,
    tensor_sha256,
    verify_export_activation,
)


def _nonspatial_state(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    spatial = set(_spatial_target_state(transformer))
    return {key: value for key, value in transformer.state_dict().items() if key not in spatial}


def _checkpoint_state(path: Path) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    payload = _torch_load(path, trusted=True)
    if not isinstance(payload, dict):
        raise ValueError("COMEDIAN Stage-1 checkpoint must be a dictionary")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("COMEDIAN Stage-1 checkpoint has no state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("COMEDIAN Stage-1 state_dict must contain string-keyed tensors only")
    return payload, state


def initialize_comedian_spatial_only(
    transformer: torch.nn.Module,
    classifier: torch.nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load exactly the online COMEDIAN SCE trunk into the ViViT spatial subset."""
    cfg = dict(config)
    if cfg.get("strategy") != "spatial_only":
        raise ValueError("COMEDIAN initialization requires strategy=spatial_only; repeat is forbidden")
    if cfg.get("source_type") != "comedian_stage1_sce":
        raise ValueError("COMEDIAN initialization requires source_type=comedian_stage1_sce")

    checkpoint_path = Path(cfg["checkpoint_path"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_hash = sha256(checkpoint_path)
    if cfg.get("checkpoint_sha256") not in (None, actual_hash):
        raise RuntimeError(
            f"SHA-256 mismatch for {checkpoint_path}: expected {cfg['checkpoint_sha256']}, "
            f"got {actual_hash}"
        )

    payload, checkpoint_state = _checkpoint_state(checkpoint_path)
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    expected_epoch = int(cfg["expected_epoch"])
    expected_step = int(cfg["expected_step"])
    if int(epoch) != expected_epoch or int(step) != expected_step:
        raise RuntimeError(
            "COMEDIAN checkpoint metadata mismatch: expected epoch/step "
            f"{expected_epoch}/{expected_step}, got {epoch}/{step}"
        )
    completed_epochs = int(epoch) + 1
    if completed_epochs != int(cfg["expected_completed_epochs"]):
        raise RuntimeError(
            "COMEDIAN completed-epoch mismatch: expected "
            f"{cfg['expected_completed_epochs']}, got {completed_epochs}"
        )

    target_state = _spatial_target_state(transformer)
    mapping = {target_key: source_key_for_target(target_key) for target_key in target_state}
    source_state = {
        key: value for key, value in checkpoint_state.items() if key.startswith("trunk.")
    }
    expected_source = set(mapping.values())
    if set(source_state) != expected_source:
        missing = sorted(expected_source - set(source_state))
        unexpected = sorted(set(source_state) - expected_source)
        raise RuntimeError(
            f"strict COMEDIAN spatial transfer key mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    allowed_ignored = {
        key
        for key in checkpoint_state
        if key in {"queue", "queue_ptr"}
        or key.startswith(
            (
                "momentum_trunk.",
                "projector.",
                "predictor.",
                "momentum_projector.",
                "train_transform.",
            )
        )
    }
    unclassified = sorted(set(checkpoint_state) - set(source_state) - allowed_ignored)
    if unclassified:
        raise RuntimeError(f"unclassified COMEDIAN checkpoint tensors: {unclassified}")

    for target_key, source_key in mapping.items():
        if target_state[target_key].shape != source_state[source_key].shape:
            raise RuntimeError(
                f"shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(source_state[source_key].shape)} != {tuple(target_state[target_key].shape)}"
            )

    spatial_before = {key: tensor_sha256(value) for key, value in target_state.items()}
    nonspatial_before = {
        key: tensor_sha256(value) for key, value in _nonspatial_state(transformer).items()
    }
    classifier_before = state_hashes(classifier)
    with torch.no_grad():
        for target_key, source_key in mapping.items():
            target_state[target_key].copy_(source_state[source_key])

    spatial_after_tensors = _spatial_target_state(transformer)
    spatial_after = {
        key: tensor_sha256(value) for key, value in spatial_after_tensors.items()
    }
    nonspatial_after = {
        key: tensor_sha256(value) for key, value in _nonspatial_state(transformer).items()
    }
    classifier_after = state_hashes(classifier)
    if nonspatial_before != nonspatial_after:
        raise RuntimeError("fresh temporal/non-spatial transformer contamination detected")
    if classifier_before != classifier_after:
        raise RuntimeError("fresh classifier contamination detected")
    mismatched = [
        target_key
        for target_key, source_key in mapping.items()
        if not torch.equal(spatial_after_tensors[target_key].cpu(), source_state[source_key].cpu())
    ]
    if mismatched:
        raise RuntimeError(f"COMEDIAN spatial transfer did not match source tensors: {mismatched}")
    changed = sorted(key for key in spatial_before if spatial_before[key] != spatial_after[key])
    if not changed:
        raise RuntimeError("COMEDIAN spatial initialization did not change any tensor")

    ignored_counts = {
        "momentum_trunk": sum(key.startswith("momentum_trunk.") for key in checkpoint_state),
        "projector": sum(key.startswith("projector.") for key in checkpoint_state),
        "predictor": sum(key.startswith("predictor.") for key in checkpoint_state),
        "momentum_projector": sum(
            key.startswith("momentum_projector.") for key in checkpoint_state
        ),
        "queue": sum(key in {"queue", "queue_ptr"} for key in checkpoint_state),
        "train_transform": sum(key.startswith("train_transform.") for key in checkpoint_state),
    }
    return {
        "schema_version": 1,
        "pass": True,
        "experiment": "comedian_table5_comedian_stage1_100f",
        "spatial_source": {
            "type": "comedian_stage1_sce",
            "run_id": cfg["run_id"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": actual_hash,
            "lightning_epoch": int(epoch),
            "completed_epochs": completed_epochs,
            "global_step": int(step),
            "checkpoint_tensor_count": len(checkpoint_state),
            "online_spatial_tensor_count": len(source_state),
            "ignored_tensor_counts": ignored_counts,
        },
        "spatial_transfer": {
            "strategy": "spatial_only",
            "strict": True,
            "complete": True,
            "shape_valid": True,
            "exact_tensor_equality": True,
            "loaded_tensor_count": len(mapping),
            "changed_tensor_count": len(changed),
            "changed_tensors": changed,
            "before_hash": aggregate_hash(spatial_before),
            "after_hash": aggregate_hash(spatial_after),
        },
        "temporal_transformer": {
            "initialization": "fresh",
            "unchanged": True,
            "before_hash": aggregate_hash(nonspatial_before),
            "after_hash": aggregate_hash(nonspatial_after),
            "tensor_hashes": nonspatial_after,
        },
        "classifier": {
            "initialization": "fresh",
            "unchanged": True,
            "before_hash": aggregate_hash(classifier_before),
            "after_hash": aggregate_hash(classifier_after),
            "tensor_hashes": classifier_after,
        },
        "model_construction_seed": int(cfg["seed"]),
        "momentum_trunk_loaded": False,
        "projector_loaded": False,
        "predictor_loaded": False,
        "queue_loaded": False,
        "stage2_checkpoint_loaded": False,
        "baidu_features_used": False,
    }


class ComedianA6000SpottingModel(SpatialInitSpottingModel):
    """Stage-1 online trunk read directly from the SCE checkpoint (the A6000-stack run of Tab. 1)."""
    spatial_initializer = staticmethod(initialize_comedian_spatial_only)


def initialize_with_verification(transformer, classifier, config):
    """Exact retained COMEDIAN initializer plus source-bound activation verification (RTX 5090 runs of Tab. 2)."""
    cfg = dict(config)
    # Reuse the 150-key strict COMEDIAN trunk-to-ViViT mapping.
    audit = initialize_comedian_spatial_only(transformer, classifier, cfg)
    checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=False)
    export = torch.load(cfg["export_path"], map_location="cpu", weights_only=True)["state_dict"]
    online = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("trunk.")}
    if set(online) != set(export) or not all(torch.equal(online[k], export[k]) for k in online):
        raise RuntimeError("export differs from the online COMEDIAN trunk")
    # Verification model creation must preserve the downstream RNG sequence.
    with torch.random.fork_rng(devices=[]):
        activation = verify_export_activation(cfg["export_path"], cfg["verification_path"],
                                              atol=cfg["activation_atol"], rtol=cfg["activation_rtol"])
    audit.update(activation_equivalence=activation, ema_target_loaded=False, jepa_predictor_loaded=False)
    audit["spatial_source"].update(export_path=cfg["export_path"], verification_path=cfg["verification_path"])
    return audit


class ComedianSpottingModel(SpatialInitSpottingModel):
    """Stage-1 online trunk into the same downstream recipe as the four arms."""
    spatial_initializer = staticmethod(initialize_with_verification)
