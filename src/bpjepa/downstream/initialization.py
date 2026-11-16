"""Strict spatial-only initialization of the spotting model from a pretrained encoder export."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import numpy as np


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str | Path, *, trusted: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only.
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        if not trusted:
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {key: tensor_sha256(value) for key, value in sorted(module.state_dict().items())}


def aggregate_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(hashes.items()):
        digest.update(key.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def _spatial_target_state(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in transformer.state_dict().items()
        if key in {"spatial_cls_token", "pos_embed"}
        or key.startswith("patch_embed.")
        or key.startswith("spatial_transformer.")
    }


def source_key_for_target(target_key: str) -> str:
    if target_key == "spatial_cls_token":
        return "trunk.cls_token"
    if target_key == "pos_embed":
        return "trunk.pos_embed"
    if target_key.startswith("patch_embed."):
        return f"trunk.{target_key}"
    prefix = "spatial_transformer."
    if target_key.startswith(prefix):
        return f"trunk.{target_key[len(prefix):]}"
    raise KeyError(f"not a spatial target tensor: {target_key}")


def _export_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = _torch_load(path)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("Pretraining export must be a non-empty state dictionary")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("Pretraining export must contain string-keyed tensors only")
    return state


def verify_checkpoint_contract(
    checkpoint_path: str | Path,
    export_state: Mapping[str, torch.Tensor],
    *,
    expected_epoch: int,
    expected_step: int,
) -> dict[str, Any]:
    payload = _torch_load(checkpoint_path, trusted=True)
    if not isinstance(payload, dict):
        raise ValueError("Pretraining checkpoint must be a dictionary")
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    if int(epoch) != expected_epoch or int(step) != expected_step:
        raise RuntimeError(
            f"Pretraining checkpoint metadata mismatch: expected epoch/step "
            f"{expected_epoch}/{expected_step}, got {epoch}/{step}"
        )
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Pretraining checkpoint has no model state")
    context = {
        f"trunk.{key.removeprefix('context_encoder.')}": value
        for key, value in model_state.items()
        if key.startswith("context_encoder.")
    }
    if set(context) != set(export_state):
        missing = sorted(set(export_state) - set(context))
        extra = sorted(set(context) - set(export_state))
        raise RuntimeError(f"checkpoint/export context keys differ: missing={missing}, extra={extra}")
    unequal = [key for key in context if not torch.equal(context[key].cpu(), export_state[key].cpu())]
    if unequal:
        raise RuntimeError(f"checkpoint context differs from validated export: {unequal}")
    return {
        "epoch": int(epoch),
        "global_step": int(step),
        "context_tensor_count": len(context),
        "ema_target_tensor_count_ignored": sum(
            key.startswith("target_encoder.") for key in model_state
        ),
        "predictor_tensor_count_ignored": sum(key.startswith("predictor.") for key in model_state),
    }


def verify_export_activation(
    export_path: str | Path,
    verification_path: str | Path,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> dict[str, Any]:
    from eztorch.models.trunks import create_vit_tiny

    model = create_vit_tiny(
        img_size=224,
        patch_size=16,
        in_chans=3,
        mlp_ratio=4,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        conv_type="Conv3d",
        tube_size=2,
    ).eval()
    state = {key.removeprefix("trunk."): value for key, value in _export_state(export_path).items()}
    model.load_state_dict(state, strict=True)
    verification = _torch_load(verification_path)
    if not isinstance(verification, dict) or set(verification) != {"input", "output"}:
        raise ValueError("activation verification artifact must contain only input and output")
    with torch.no_grad():
        actual = model(verification["input"])
    expected = verification["output"]
    maximum = float((actual - expected).abs().max())
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    if not passed:
        raise RuntimeError(
            f"Pretraining activation equivalence failed: max_abs_difference={maximum}, "
            f"atol={atol}, rtol={rtol}"
        )
    return {"pass": True, "max_abs_difference": maximum, "atol": atol, "rtol": rtol}


def initialize_spatial_only(
    transformer: torch.nn.Module,
    classifier: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    activation_verifier: Callable[..., dict[str, Any]] = verify_export_activation,
) -> dict[str, Any]:
    """Strictly load the context export into only the ViViT spatial subset."""
    cfg = dict(config)
    if cfg.get("strategy") != "spatial_only":
        raise ValueError("Table-5 initialization requires strategy=spatial_only; repeat is forbidden")
    checkpoint_path = Path(cfg["checkpoint_path"]).resolve()
    export_path = Path(cfg["export_path"]).resolve()
    verification_path = Path(cfg["verification_path"]).resolve()
    expected_hashes = {}
    for path, key in ((checkpoint_path, "checkpoint_sha256"), (export_path, "export_sha256"),
                      (verification_path, "verification_sha256")):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if cfg.get(key) not in (None, actual):
            raise RuntimeError(f"SHA-256 mismatch for {path}: expected {cfg[key]}, got {actual}")
        expected_hashes[path] = actual

    temporal_before = state_hashes(transformer.temporal_transformer)
    classifier_before = state_hashes(classifier)
    spatial_before_tensors = _spatial_target_state(transformer)
    spatial_before = {key: tensor_sha256(value) for key, value in spatial_before_tensors.items()}

    export_state = _export_state(export_path)
    checkpoint = verify_checkpoint_contract(
        checkpoint_path,
        export_state,
        expected_epoch=int(cfg["expected_epoch"]),
        expected_step=int(cfg["expected_step"]),
    )
    target_state = _spatial_target_state(transformer)
    mapping = {target_key: source_key_for_target(target_key) for target_key in target_state}
    expected_source = set(mapping.values())
    source_keys = set(export_state)
    if source_keys != expected_source:
        missing = sorted(expected_source - source_keys)
        unexpected = sorted(source_keys - expected_source)
        raise RuntimeError(f"strict spatial transfer key mismatch: missing={missing}, unexpected={unexpected}")
    for target_key, source_key in mapping.items():
        if target_state[target_key].shape != export_state[source_key].shape:
            raise RuntimeError(
                f"shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(export_state[source_key].shape)} != {tuple(target_state[target_key].shape)}"
            )
    with torch.no_grad():
        for target_key, source_key in mapping.items():
            target_state[target_key].copy_(export_state[source_key])

    temporal_after = state_hashes(transformer.temporal_transformer)
    classifier_after = state_hashes(classifier)
    spatial_after_tensors = _spatial_target_state(transformer)
    spatial_after = {key: tensor_sha256(value) for key, value in spatial_after_tensors.items()}
    if temporal_before != temporal_after:
        raise RuntimeError("temporal transformer contamination detected during spatial initialization")
    if classifier_before != classifier_after:
        raise RuntimeError("classifier contamination detected during spatial initialization")
    mismatched = [
        target_key
        for target_key, source_key in mapping.items()
        if not torch.equal(spatial_after_tensors[target_key].cpu(), export_state[source_key].cpu())
    ]
    if mismatched:
        raise RuntimeError(f"spatial transfer did not match source tensors: {mismatched}")
    changed = sorted(key for key in spatial_before if spatial_before[key] != spatial_after[key])
    if not changed:
        raise RuntimeError("spatial initialization did not change any tensor")

    activation = activation_verifier(
        export_path,
        verification_path,
        atol=float(cfg.get("activation_atol", 2e-5)),
        rtol=float(cfg.get("activation_rtol", 2e-4)),
    )
    return {
        "schema_version": 1,
        "pass": True,
        "experiment": "spatial_only_100f",
        "spatial_source": {
            "run_id": cfg["run_id"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": expected_hashes[checkpoint_path],
            "export_path": str(export_path),
            "export_sha256": expected_hashes[export_path],
            "verification_path": str(verification_path),
            "verification_sha256": expected_hashes[verification_path],
            **checkpoint,
        },
        "spatial_transfer": {
            "strategy": "spatial_only",
            "strict": True,
            "complete": True,
            "shape_valid": True,
            "loaded_tensor_count": len(mapping),
            "changed_tensor_count": len(changed),
            "changed_tensors": changed,
            "before_hash": aggregate_hash(spatial_before),
            "after_hash": aggregate_hash(spatial_after),
        },
        "activation_equivalence": activation,
        "temporal_transformer": {
            "initialization": "fresh",
            "unchanged": True,
            "before_hash": aggregate_hash(temporal_before),
            "after_hash": aggregate_hash(temporal_after),
            "tensor_hashes": temporal_after,
        },
        "classifier": {
            "initialization": "fresh",
            "unchanged": True,
            "before_hash": aggregate_hash(classifier_before),
            "after_hash": aggregate_hash(classifier_after),
            "tensor_hashes": classifier_after,
        },
        "model_construction_seed": int(cfg["seed"]),
        "ema_target_loaded": False,
        "jepa_predictor_loaded": False,
        "stage2_checkpoint_loaded": False,
        "baidu_features_used": False,
    }


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(serialized)
    except FileExistsError:
        if path.read_text() != serialized:
            raise RuntimeError(f"refusing to replace non-identical immutable audit: {path}")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cuda_peak_memory_bytes(device: torch.device | str) -> int:
    """Return CUDA peak allocation only when the recorded device is CUDA."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


try:
    from eztorch.models.soccernet_spotting import SoccerNetSpottingModel
except ImportError:  # Allows the hash helpers to be imported without the downstream dependencies.
    SoccerNetSpottingModel = torch.nn.Module  # type: ignore[assignment,misc]


class SpatialInitSpottingModel(SoccerNetSpottingModel):
    """Eztorch spotting model with strict spatial-only initialization (or scratch), whole model trainable."""

    spatial_initializer = staticmethod(initialize_spatial_only)

    def __init__(self, *args: Any, spatial_init: Mapping[str, Any], **kwargs: Any) -> None:
        if kwargs.get("pretrained_path") is not None or kwargs.get("pretrained_trunk_path") is not None:
            raise ValueError("this route loads only spatial weights; pretrained_path and pretrained_trunk_path must be null")
        super().__init__(*args, **kwargs)
        if not hasattr(self.trunk, "transformer"):
            raise TypeError("requires the Eztorch ViViT token-handler trunk")
        self._spatial_config = dict(spatial_init)
        if self._spatial_config.get("strategy") == "scratch":
            from .scratch import initialize_scratch
            self.initialization_audit = initialize_scratch(self.trunk.transformer, self.head_class, self._spatial_config)
        else:
            self.initialization_audit = self.spatial_initializer(self.trunk.transformer, self.head_class, self._spatial_config)
        frozen = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        if self.freeze_trunk or frozen:
            raise RuntimeError(f"whole model must be trainable from epoch 0; frozen={frozen}")
        self.initialization_audit["trainability"] = {
            "whole_model_trainable": True,
            "trainable_parameter_tensors": sum(1 for _ in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def configure_optimizers(self) -> Any:
        if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
            path = Path(self._spatial_config["initialization_audit_path"])
            if not path.exists():
                write_immutable_json(path, self.initialization_audit)
        return super().configure_optimizers()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._fit_started = time.monotonic()
        self._fit_start_step = int(self.trainer.global_step)
        # Lightning may expose ``self.device`` as CPU by ``on_fit_end``. Keep
        # the actual training device so CUDA accounting and NCCL reduction do
        # not accidentally receive a CPU device during teardown.
        self._runtime_device = torch.device(self.device)
        if self._runtime_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._runtime_device)

    def on_fit_end(self) -> None:
        wall_seconds = time.monotonic() - self._fit_started
        runtime_device = getattr(self, "_runtime_device", torch.device(self.device))
        peak_bytes = cuda_peak_memory_bytes(runtime_device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            peak = torch.tensor(peak_bytes, device=runtime_device, dtype=torch.long)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            peak_bytes = int(peak.item())
        if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
            directory = Path(self._spatial_config["runtime_audit_dir"])
            write_immutable_json(
                directory / f"segment-step-{self._fit_start_step:09d}-{int(self.trainer.global_step):09d}.json",
                {
                    "started_global_step": self._fit_start_step,
                    "completed_global_step": int(self.trainer.global_step),
                    "wall_seconds": wall_seconds,
                    "world_size": int(self.trainer.world_size),
                    "gpu_hours": wall_seconds * int(self.trainer.world_size) / 3600,
                    "peak_memory_mib_max_rank": peak_bytes / (1024**2),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        return super().on_fit_end()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = capture_rng_state()
        checkpoint["table5_rng_state"] = state  # key name kept so older checkpoints stay readable
        states = [state]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(states, state)
        checkpoint["resume_rng_by_rank"] = states
