#!/usr/bin/env python3
"""Epoch-based, resumable trainer for all latent-prediction pretraining runs in the paper.

R, BP-JEPA, TC and BP-JEPA+TC share one code path (`objectives` section of the config); the recipe-study
variants use the plain multi-mask path (`objectives: null`) with their own data and schedule settings.

    torchrun --standalone --nproc_per_node=2 -m bpjepa.pretrain.train --config configs/pretrain/bp_jepa.json \\
        --clip-manifest clips.json --frame-root frames/train --evidence-dir evidence --output-dir runs/bp_jepa
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import platform
import random
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from bpjepa.common import append_jsonl, atomic_json, canonical_digest, refuse_existing, sha256
from bpjepa.jepa.crop_data import CropEpochSampler, SpatialCropDataset
from bpjepa.jepa.full_data import FullSoccerNetClipDataset, PaddedDistributedEpochSampler
from bpjepa.jepa.model import BaseVJEPA, ema_momentum
from bpjepa.jepa.short_data import Centered64Dataset
from bpjepa.jepa.temporal_vjepa import PresentationDataset, PresentationSampler, check_temporal_gradients, collate_metadata
from bpjepa.jepa.transfer import export_comedian
from bpjepa.pretrain import vjepa_recipe
from bpjepa.run_record import write_run_record


CHECKPOINT_FORMAT_VERSION = 3
CRITICAL_METRICS = ("loss", "feature_std", "feature_variance", "ema_distance")
_ACTIVE_RUN: dict[str, Any] = {}
# Settings that may change without changing a run's scientific identity (resume check only).
NONSCIENTIFIC = frozenset({"run_id", "purpose", "data_loader_workers", "checkpoint_max_hours", "provenance"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_identity(config: dict) -> str:
    return canonical_digest({k: v for k, v in config.items() if k not in NONSCIENTIFIC})


def validate_config(config: dict) -> None:
    """Ordinary runtime validation: required fields, consistent arithmetic, supported recipe combinations."""
    condition = config.get("condition")
    if condition != "compact_temporal_multimask_v1":
        raise ValueError(f"unsupported condition: {condition!r}")
    from bpjepa.jepa.temporal import TemporalPredictor
    from bpjepa.samplers.vjepa_multimask import VJEPAMultiMaskSampler
    if set(config["predictor"]) != {"width", "depth", "num_heads", "mlp_ratio"}:
        raise ValueError("temporal predictor requires explicit architecture dimensions")
    with torch.random.fork_rng(devices=[]):
        TemporalPredictor(config["architecture"]["embed_dim"], **config["predictor"])
    VJEPAMultiMaskSampler(config["mask"])
    if config.get("clip_frames") not in (128, 64) or config.get("image_size") != 224:
        raise ValueError("clip_frames must be 128 or 64 and image_size 224")
    positive = ("world_size", "micro_batch_per_gpu", "gradient_accumulation_steps", "effective_global_batch",
                "learning_rate", "checkpoint_max_hours", "epochs", "gradient_clip_norm", "data_loader_workers")
    if any(float(config.get(key, 0)) <= 0 for key in positive):
        raise ValueError(f"these config values must be positive: {positive}")
    effective = (int(config["micro_batch_per_gpu"]) * int(config["world_size"])
                 * int(config["gradient_accumulation_steps"]))
    if effective != int(config["effective_global_batch"]):
        raise ValueError(f"effective global batch mismatch: calculated {effective}")
    validate_lr_schedule(config)
    objectives = config.get("objectives")
    if objectives is not None:
        if config["augmentation"].get("randomness") != "none" or config["clip_frames"] != 128:
            raise ValueError("auxiliary objectives run on the 128-frame temporal multi-mask path without random augmentation")
        kernel = objectives.get("kernel")
        if not isinstance(kernel, dict) or type(objectives.get("head_seed")) is not int:
            raise ValueError("objectives require kernel settings and head_seed")
        if type(objectives.get("lambda_tc")) is not float or objectives["lambda_tc"] < 0:
            raise ValueError("objectives.lambda_tc must be a nonnegative float")
        if type(objectives.get("warmup_updates")) is not int or objectives["warmup_updates"] < 1:
            raise ValueError("objectives.warmup_updates must be a positive integer")
        boundary = objectives.get("boundary")
        if boundary is not None and (boundary.get("kind") != "EB" or type(boundary.get("lambda_b")) is not float
                                     or type(boundary.get("aux_head_seed")) is not int
                                     or boundary.get("min_per_side", 1) != 1):
            raise ValueError("objectives.boundary requires kind=EB, float lambda_b, integer aux_head_seed, min_per_side=1")


def validate_lr_schedule(config: dict) -> None:
    schedule = config.get("scheduler", {})
    if schedule.get("name") == "constant":
        return
    if schedule.get("name") != "linear_warmup_cosine":
        raise ValueError("scheduler.name must be constant or linear_warmup_cosine")
    warmup, total, horizon = (schedule.get(key) for key in
                              ("warmup_updates", "total_updates", "horizon_updates"))
    if not all(type(value) is int for value in (warmup, total, horizon)) or not 0 < warmup < total <= horizon:
        raise ValueError("scheduler requires integer 0 < warmup_updates < total_updates <= horizon_updates")
    start, peak, end = (float(schedule["start_lr"]), float(config["learning_rate"]),
                        float(schedule["end_lr"]))
    if not all(math.isfinite(value) for value in (start, peak, end)) or not 0 < end <= start <= peak:
        raise ValueError("scheduler requires finite 0 < end_lr <= start_lr <= learning_rate")


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """LR at completed-update count u; step once after each optimizer update.

    At u=0 the optimizer uses start_lr. After update u, its LR is f(u),
    ready for the next update. Resume restores both scheduler and optimizer.
    """
    validate_lr_schedule(config)
    schedule = config["scheduler"]
    if schedule["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    peak = float(config["learning_rate"])
    start, end = float(schedule["start_lr"]), float(schedule["end_lr"])
    warmup, horizon = schedule["warmup_updates"], schedule["horizon_updates"]

    def factor(update):
        if update <= warmup:
            return (start + (peak - start) * update / warmup) / peak
        progress = min(1.0, (update - warmup) / (horizon - warmup))
        return (end + (peak - end) * (1 + math.cos(math.pi * progress)) / 2) / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def gradient_norm(module: torch.nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


@torch.no_grad()
def ema_distance(model: BaseVJEPA) -> float:
    squares = [
        (context.detach().float() - target.detach().float()).square().sum()
        for context, target in zip(
            model.context_encoder.parameters(), model.target_encoder.parameters(), strict=True
        )
    ]
    return float(torch.stack(squares).sum().sqrt())


def capture_rng_state(
    mask_generator: torch.Generator, data_generator: torch.Generator | None = None
) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "masking": mask_generator.get_state(),
        "data_loader": data_generator.get_state() if data_generator is not None else None,
    }


def restore_rng_state(
    state: dict,
    mask_generator: torch.Generator,
    data_generator: torch.Generator | None = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    mask_generator.set_state(state["masking"])
    if data_generator is not None and state.get("data_loader") is not None:
        data_generator.set_state(state["data_loader"])


def checkpoint_due(*, epoch_complete: bool, last_save_monotonic: float, now: float,
                   maximum_hours: float) -> bool:
    return epoch_complete or now - last_save_monotonic >= maximum_hours * 3600


def _validate_checkpoint_payload(payload: dict) -> None:
    required = {
        "format_version",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "sample_cursor",
        "global_step",
        "accumulation_position",
        "sampler_states",
        "rng_states",
        "config_sha256",
        "manifest_sha256",
        "resume_history",
        "timing",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"checkpoint is incomplete: {missing}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("unsupported checkpoint format")
    if payload["accumulation_position"] != 0:
        raise RuntimeError("production checkpoints must be written at optimizer boundaries")
    if not payload["model"] or not payload["optimizer"]:
        raise RuntimeError("checkpoint contains empty model/optimizer state")
    timing = payload["timing"]
    for key in ("active_wall_seconds", "gpu_hours", "original_started_at"):
        if key not in timing:
            raise RuntimeError(f"checkpoint timing is incomplete: missing {key}")
    if float(timing["active_wall_seconds"]) < 0 or float(timing["gpu_hours"]) < 0:
        raise RuntimeError("checkpoint timing totals must be non-negative")


def atomic_save_checkpoint(path: Path, payload: dict) -> str:
    """Write a new checkpoint atomically and validate it before and after rename."""
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    _validate_checkpoint_payload(loaded)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_payload(loaded)
    return sha256(path)


def load_checkpoint(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_payload(payload)
    if payload["config_sha256"] != expected_config_sha256:
        raise RuntimeError("resume rejected: frozen config identity mismatch")
    if payload["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError("resume rejected: frozen clip-manifest identity mismatch")
    return payload


def _append_telemetry(path: Path, record: dict) -> bool:
    """Best-effort telemetry which cannot prevent training/checkpointing."""
    try:
        append_jsonl(path, record)
        return True
    except OSError as error:
        print(f"WARNING: telemetry append failed for {path}: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return False


def cumulative_timing(
    *, prior_wall_seconds: float, prior_gpu_hours: float,
    invocation_started_monotonic: float, now_monotonic: float,
    world_size: int, original_started_at: str,
) -> dict:
    invocation_seconds = max(0.0, now_monotonic - invocation_started_monotonic)
    return {
        "active_wall_seconds": prior_wall_seconds + invocation_seconds,
        "gpu_hours": prior_gpu_hours + invocation_seconds * world_size / 3600,
        "original_started_at": original_started_at,
        "current_invocation_wall_seconds": invocation_seconds,
    }


def _dist_info(config: dict, requested_device: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != int(config["world_size"]):
        raise RuntimeError(f"launch world size {world} != frozen world size {config['world_size']}")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if world > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=device)
    else:
        if world != 1:
            raise RuntimeError("CPU dry runs support world_size=1 only")
        device = torch.device("cpu")
    return world, rank, device


def _gather_rank_objects(value: Any, world: int, rank: int) -> list[Any] | None:
    if world == 1:
        return [value]
    gathered = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


def _model_from_config(config: dict) -> BaseVJEPA:
    from bpjepa.jepa.temporal import build_model
    return build_model({"model_id": config["condition"],
                        "encoder": {"image_size": config["image_size"], **config["architecture"]},
                        "predictor": config["predictor"], "mask": config["mask"]})


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _environment(device: torch.device, world: int, rank: int) -> dict:
    result = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "world_size": world,
        "rank": rank,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=properties.name,
            physical_device=os.environ.get("CUDA_VISIBLE_DEVICES", str(device.index)),
        )
    return result


def load_resolved_config(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).read_text())
    for key, value in (("world_size", args.world_size), ("micro_batch_per_gpu", args.micro_batch_per_gpu),
                       ("effective_global_batch", args.effective_global_batch),
                       ("data_loader_workers", args.num_workers)):
        if value is not None:
            config[key] = value
    validate_config(config)
    return config


def build_objective_model(base, config: dict, manifest_sha256: str, args: argparse.Namespace, output: Path):
    """Wrap the base multi-mask model with the TC path and, for BP arms, the boundary head."""
    from bpjepa.jepa.boundary import EBStreamLog, EBTemporalVJEPA
    from bpjepa.jepa.conditions import UCondition
    from bpjepa.jepa.kernel import KernelSettings
    from bpjepa.jepa.temporal_vjepa import StreamLog, TemporalVJEPA
    from bpjepa.pretrain.clip_validity import ClipValiditySource
    objectives = config["objectives"]
    settings = KernelSettings.from_fields(objectives["kernel"])
    evidence = Path(args.evidence_dir)
    validity = ClipValiditySource(evidence / "clip_validity.npz", manifest_sha256, settings.offsets)
    shared = dict(head_seed=objectives["head_seed"], validity_source=validity, manifest_sha256=manifest_sha256,
                  lambda_final=objectives["lambda_tc"], warmup_updates=objectives["warmup_updates"], condition=UCondition())
    logging = dict(stream_updates=objectives.get("stream_log_updates"), telemetry_interval=objectives.get("telemetry_interval_updates"))
    boundary = objectives.get("boundary")
    if boundary is None:
        return TemporalVJEPA(base, settings, observer=StreamLog(output, **logging), **shared)
    from bpjepa.evidence.target_table import load_aux_table
    observer = EBStreamLog(output, **logging)
    return EBTemporalVJEPA(base, settings, aux_table=load_aux_table(evidence / "target_table_boundary.npz", kind=boundary["kind"],
                                                                    clip_manifest_sha256=manifest_sha256),
                           aux_head_seed=boundary["aux_head_seed"], lambda_b_final=boundary["lambda_b"],
                           proj_dim=boundary.get("proj_dim", 32), aux_observer=observer, observer=observer, **shared)


def _dataset_and_sampler_classes(config: dict):
    """Recipe-specific data classes; the V-JEPA recipe and the crop variant use local-RNG spatial crops."""
    policy = config["augmentation"]["policy"]
    if policy == vjepa_recipe.AUGMENTATION["policy"]:
        return vjepa_recipe.RecipeDataset, CropEpochSampler
    if policy == "vjepa_random_resized_crop_only_v1":
        return SpatialCropDataset, CropEpochSampler
    if config["clip_frames"] == 64:
        return Centered64Dataset, PaddedDistributedEpochSampler
    return FullSoccerNetClipDataset, PaddedDistributedEpochSampler


def _run(args: argparse.Namespace) -> int:
    _ACTIVE_RUN.clear()
    manifest_path = args.clip_manifest.resolve()
    config = load_resolved_config(args)
    config_hash = config_identity(config)
    manifest_hash = sha256(manifest_path)
    objectives = config.get("objectives")
    if objectives is not None and args.evidence_dir is None:
        raise ValueError("this arm needs --evidence-dir (clip_validity.npz, and target_table_boundary.npz for BP arms)")
    recipe = config.get("recipe") == "vjepa"
    if recipe:
        vjepa_recipe.validate_config(config)

    world, rank, device = _dist_info(config, args.device)
    seed = int(config["pretraining_seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + rank)
        torch.cuda.reset_peak_memory_stats(device)

    output = args.output_dir.resolve()
    if args.resume is None:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError("resume output directory is missing")
    if world > 1:
        dist.barrier()

    dataset_class, sampler_class = _dataset_and_sampler_classes(config)
    if objectives is not None:
        sampler_class = PresentationSampler
    base_dataset = dataset_class(manifest_path, frame_root=args.frame_root, image_size=config["image_size"],
                                 require_full_corpus=not args.allow_subset_corpus,
                                 **({"with_row_id": True} if objectives is not None else {}))
    dataset = PresentationDataset(base_dataset) if objectives is not None else base_dataset
    sampler = sampler_class(
        len(dataset),
        num_replicas=world,
        rank=rank,
        micro_batch_size=int(config["micro_batch_per_gpu"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        seed=seed,
        shuffle=True,
    )
    loader_workers = int(config["data_loader_workers"])
    data_generator = torch.Generator().manual_seed(seed + 200_003 * rank)
    loader = DataLoader(
        dataset,
        batch_size=int(config["micro_batch_per_gpu"]),
        sampler=sampler,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=loader_workers > 0,
        generator=data_generator,
        **({"collate_fn": collate_metadata} if objectives is not None else {}),
    )

    model = _model_from_config(config)
    if objectives is not None:
        model = build_objective_model(model, config, manifest_hash, args, output)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )
    scheduler = (vjepa_recipe.build_scheduler(optimizer, config, build_lr_scheduler) if recipe
                 else build_lr_scheduler(optimizer, config))
    precision = config["precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    mask_generator = torch.Generator().manual_seed(seed + 100_003 * rank)
    updates_per_epoch = sampler.num_samples // (
        int(config["micro_batch_per_gpu"]) * int(config["gradient_accumulation_steps"])
    )
    total_steps = updates_per_epoch * int(config["epochs"])
    if config["scheduler"].get("total_updates", total_steps) != total_steps:
        raise ValueError("resolved optimizer updates differ from the LR schedule training budget")
    epoch = 0
    sample_cursor = 0
    global_step = 0
    resume_history: list[dict] = []
    prior_wall_seconds = 0.0
    prior_gpu_hours = 0.0
    original_started_at = utc_now()

    _ACTIVE_RUN.update({
        "config": config,
        "output": output,
        "epoch": epoch,
        "sample_cursor": sample_cursor,
        "global_step": global_step,
        "rank": rank,
    })

    if args.resume is not None:
        payload = load_checkpoint(
            args.resume.resolve(),
            expected_config_sha256=config_hash,
            expected_manifest_sha256=manifest_hash,
        )
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        epoch = int(payload["epoch"])
        sample_cursor = int(payload["sample_cursor"])
        global_step = int(payload["global_step"])
        if scheduler.last_epoch != global_step:
            raise RuntimeError("resume rejected: scheduler progress differs from global_step")
        sampler.load_state_dict(payload["sampler_states"][rank])
        restore_rng_state(payload["rng_states"][rank], mask_generator, data_generator)
        resume_history = list(payload["resume_history"])
        prior_wall_seconds = float(payload["timing"]["active_wall_seconds"])
        prior_gpu_hours = float(payload["timing"]["gpu_hours"])
        original_started_at = str(payload["timing"]["original_started_at"])
        resume_history.append(
            {"resumed_at": utc_now(), "checkpoint": str(args.resume.resolve()),
             "checkpoint_sha256": sha256(args.resume), "epoch": epoch,
             "sample_cursor": sample_cursor, "global_step": global_step}
        )
        _ACTIVE_RUN.update(epoch=epoch, sample_cursor=sample_cursor, global_step=global_step)

    ddp_model: torch.nn.Module = model
    if world > 1:
        ddp_model = DistributedDataParallel(model, device_ids=[device.index])

    started_monotonic = time.monotonic()
    _ACTIVE_RUN.update(
        prior_wall_seconds=prior_wall_seconds,
        prior_gpu_hours=prior_gpu_hours,
        original_started_at=original_started_at,
        invocation_started_monotonic=started_monotonic,
        world_size=world,
    )
    last_save = started_monotonic
    if rank == 0:
        write_run_record(output, stage="pretrain", config=config, seed=seed,
                         name=f"run_record_step{global_step:09d}.json",
                         inputs={"clip_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
                                 "frame_root": str(args.frame_root) if args.frame_root else None,
                                 "evidence_dir": str(args.evidence_dir) if args.evidence_dir else None,
                                 "resume_checkpoint": str(args.resume) if args.resume else None},
                         outputs={"output_dir": str(output)})
        launch = {"run_id": config["run_id"], "condition": config["condition"], "pretraining_seed": seed,
                  "config_identity": config_hash, "manifest_sha256": manifest_hash, "output_dir": str(output),
                  "environment": _environment(device, world, rank),
                  "base_samples_per_epoch": len(dataset), "ddp_total_samples_per_epoch": sampler.total_size,
                  "ddp_padding_samples_per_epoch": sampler.padding_size,
                  "optimizer_updates_per_epoch": updates_per_epoch, "planned_optimizer_updates": total_steps,
                  "started_at": utc_now(), "resume": str(args.resume.resolve()) if args.resume else None}
        _append_telemetry(output / "run_events.jsonl",
                          {"event": "resume_started" if args.resume else "training_started", **launch})

    accumulation = int(config["gradient_accumulation_steps"])
    checkpoint_max_hours = float(config["checkpoint_max_hours"])
    optimizer.zero_grad(set_to_none=True)
    stop_requested = False
    resume_mid_epoch = args.resume is not None and sample_cursor > 0
    while epoch < int(config["epochs"]):
        sampler.set_epoch(epoch, start_index=sample_cursor)
        epoch_start = time.monotonic()
        saved_data_state = data_generator.get_state() if resume_mid_epoch else None
        loader_iterator = iter(loader)
        if saved_data_state is not None:
            # Iterator construction draws worker base seeds. Workers perform no random
            # augmentation, so preserve the checkpointed generator position exactly.
            data_generator.set_state(saved_data_state)
            resume_mid_epoch = False
        for batch_index, clips in enumerate(loader_iterator):
            clips = clips.to(device, non_blocking=True)
            within_accum = batch_index % accumulation
            synchronize = within_accum == accumulation - 1
            sync_context = contextlib.nullcontext()
            if world > 1 and not synchronize:
                sync_context = ddp_model.no_sync()  # type: ignore[attr-defined]
            with sync_context, _autocast(device, precision):
                result = ddp_model(clips, generator=mask_generator)
                loss = result["loss"] / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite pretraining loss")
            scaler.scale(loss).backward()
            sample_cursor += int(clips.shape[0])
            if not synchronize:
                continue

            scaler.unscale_(optimizer)
            if objectives is not None:  # after unscale_, before global clipping
                check_temporal_gradients(model)
                model.observer.log_gradients(int(model.temporal_completed_updates) + 1, model)
            context_grad = gradient_norm(model.context_encoder)
            predictor_grad = gradient_norm(model.predictor)
            if not math.isfinite(context_grad) or not math.isfinite(predictor_grad):
                raise FloatingPointError("non-finite gradient norm")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            momentum = (vjepa_recipe.momentum if recipe else ema_momentum)(
                global_step, total_steps, float(config["ema_start"]), float(config["ema_end"])
            )
            model.update_target(momentum)
            optimizer.zero_grad(set_to_none=True)

            now = time.monotonic()
            due_time = checkpoint_due(
                epoch_complete=False,
                last_save_monotonic=last_save,
                now=now,
                maximum_hours=checkpoint_max_hours,
            )
            dry_stop = args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps
            _ACTIVE_RUN.update(epoch=epoch, sample_cursor=sample_cursor, global_step=global_step)

            # A due checkpoint is scientific state; telemetry is best-effort. Save
            # and validate the checkpoint before attempting any metrics append.
            if due_time or dry_stop:
                rank_state = sampler.state_dict()
                rank_state["start_index"] = sample_cursor
                sampler_states = _gather_rank_objects(rank_state, world, rank)
                rng_states = _gather_rank_objects(
                    capture_rng_state(mask_generator, data_generator), world, rank
                )
                if rank == 0:
                    checkpoint = output / "checkpoints" / (
                        f"epoch-{epoch:03d}-sample-{sample_cursor:06d}-step-{global_step:09d}.pt"
                    )
                    payload = {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "sample_cursor": sample_cursor,
                        "global_step": global_step,
                        "accumulation_position": 0,
                        "sampler_states": sampler_states,
                        "rng_states": rng_states,
                        "config_sha256": config_hash,
                        "manifest_sha256": manifest_hash,
                        "resume_history": resume_history,
                        "timing": cumulative_timing(
                            prior_wall_seconds=prior_wall_seconds,
                            prior_gpu_hours=prior_gpu_hours,
                            invocation_started_monotonic=started_monotonic,
                            now_monotonic=now,
                            world_size=world,
                            original_started_at=original_started_at,
                        ),
                    }
                    digest = atomic_save_checkpoint(checkpoint, payload)
                    checkpoint_event = {
                        "timestamp": utc_now(), "checkpoint": str(checkpoint), "sha256": digest,
                        "epoch": epoch, "sample_cursor": sample_cursor, "global_step": global_step,
                        "health": "validated", "reason": "dry_stop" if dry_stop else "six_hour_policy",
                        **payload["timing"],
                    }
                    _append_telemetry(output / "checkpoint_events.jsonl", checkpoint_event)
                last_save = now

            features = result["context_features"].detach().float().reshape(
                -1, model.context_encoder.embed_dim
            )
            timing = cumulative_timing(
                prior_wall_seconds=prior_wall_seconds,
                prior_gpu_hours=prior_gpu_hours,
                invocation_started_monotonic=started_monotonic,
                now_monotonic=now,
                world_size=world,
                original_started_at=original_started_at,
            )
            metric = {
                "timestamp": utc_now(),
                "epoch": epoch,
                "sample_cursor": sample_cursor,
                "global_step": global_step,
                "loss": float(result["loss"].detach()),
                "feature_std": float(features.std(dim=0).mean()),
                "feature_variance": float(features.var(dim=0).mean()),
                "context_gradient_norm": context_grad,
                "predictor_gradient_norm": predictor_grad,
                "ema_distance": ema_distance(model),
                "ema_momentum": momentum,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "throughput_clips_per_second_rank": sample_cursor / max(now - epoch_start, 1e-9),
                "elapsed_wall_seconds": timing["active_wall_seconds"],
                "gpu_hours": timing["gpu_hours"],
                "invocation_wall_seconds": timing["current_invocation_wall_seconds"],
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
                ),
                "effective_global_batch": config["effective_global_batch"],
                "remaining_epochs": int(config["epochs"]) - epoch - 1,
            }
            if not all(math.isfinite(float(metric[key])) for key in CRITICAL_METRICS):
                raise FloatingPointError("non-finite critical training metric")
            if rank == 0:
                _append_telemetry(output / "metrics.jsonl", metric)

            if dry_stop:
                stop_requested = True
                break

        if stop_requested:
            break
        if sample_cursor != sampler.num_samples:
            raise RuntimeError(
                f"epoch traversal incomplete: {sample_cursor} of {sampler.num_samples} rank samples"
            )
        epoch += 1
        sample_cursor = 0
        _ACTIVE_RUN.update(epoch=epoch, sample_cursor=sample_cursor, global_step=global_step)
        sampler.set_epoch(epoch, start_index=0)
        rank_state = sampler.state_dict()
        sampler_states = _gather_rank_objects(rank_state, world, rank)
        rng_states = _gather_rank_objects(
            capture_rng_state(mask_generator, data_generator), world, rank
        )
        if rank == 0:
            checkpoint_now = time.monotonic()
            checkpoint = output / "checkpoints" / f"epoch-{epoch:03d}-step-{global_step:09d}.pt"
            payload = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "sample_cursor": 0,
                "global_step": global_step,
                "accumulation_position": 0,
                "sampler_states": sampler_states,
                "rng_states": rng_states,
                "config_sha256": config_hash,
                "manifest_sha256": manifest_hash,
                "resume_history": resume_history,
                "timing": cumulative_timing(
                    prior_wall_seconds=prior_wall_seconds,
                    prior_gpu_hours=prior_gpu_hours,
                    invocation_started_monotonic=started_monotonic,
                    now_monotonic=checkpoint_now,
                    world_size=world,
                    original_started_at=original_started_at,
                ),
            }
            digest = atomic_save_checkpoint(checkpoint, payload)
            checkpoint_event = {
                "timestamp": utc_now(), "checkpoint": str(checkpoint), "sha256": digest,
                "epoch": epoch, "sample_cursor": 0, "global_step": global_step,
                "health": "validated", "reason": "epoch_complete",
                **payload["timing"],
            }
            _append_telemetry(output / "checkpoint_events.jsonl", checkpoint_event)
        last_save = time.monotonic()

    if rank == 0:
        final_timing = cumulative_timing(
            prior_wall_seconds=prior_wall_seconds,
            prior_gpu_hours=prior_gpu_hours,
            invocation_started_monotonic=started_monotonic,
            now_monotonic=time.monotonic(),
            world_size=world,
            original_started_at=original_started_at,
        )
        completed = epoch == int(config["epochs"])
        final = {
            "event": "training_stopped",
            "timestamp": utc_now(),
            "completed": completed,
            "completed_epochs": epoch,
            "sample_cursor": sample_cursor,
            "global_step": global_step,
            "wall_seconds": final_timing["active_wall_seconds"],
            "gpu_hours": final_timing["gpu_hours"],
            "original_started_at": original_started_at,
            "current_invocation_wall_seconds": final_timing["current_invocation_wall_seconds"],
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            ),
            "resume_count": len(resume_history),
        }
        _append_telemetry(output / "run_events.jsonl", final)
        if completed:
            final_checkpoint = output / "checkpoints" / f"epoch-{epoch:03d}-step-{global_step:09d}.pt"
            export_path = output / "exports" / "comedian_spatial_context.pt"
            first = base_dataset[(0, 0) if sampler_class is CropEpochSampler else 0]
            verification_clip = (first[0] if objectives is not None else first)[None]
            verification = export_comedian(
                copy.deepcopy(model.context_encoder).cpu(), export_path, verification_clip
            )
            export_event = {
                "event": "spatial_export", "timestamp": utc_now(),
                "checkpoint": str(final_checkpoint), "checkpoint_sha256": sha256(final_checkpoint),
                "export": str(export_path), "export_sha256": sha256(export_path),
                "verification": str(verification), "verification_sha256": sha256(verification),
                "wall_seconds": final_timing["active_wall_seconds"],
                "gpu_hours": final_timing["gpu_hours"],
            }
            _append_telemetry(output / "run_events.jsonl", export_event)
            write_run_record(output, stage="pretrain", config=config, seed=seed, name="run_record_final.json",
                             inputs={"clip_manifest": {"path": str(manifest_path), "sha256": manifest_hash}},
                             outputs={"final_checkpoint": {"path": str(final_checkpoint), "sha256": export_event["checkpoint_sha256"]},
                                      "export": {"path": str(export_path), "sha256": export_event["export_sha256"]},
                                      "transfer_verification": {"path": str(verification), "sha256": export_event["verification_sha256"]}})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def run(args: argparse.Namespace) -> int:
    """Run training and make every terminal failure reconstructable."""
    try:
        return _run(args)
    except BaseException as error:
        if int(os.environ.get("RANK", "0")) == 0 and _ACTIVE_RUN.get("config"):
            timing = cumulative_timing(
                prior_wall_seconds=float(_ACTIVE_RUN.get("prior_wall_seconds", 0.0)),
                prior_gpu_hours=float(_ACTIVE_RUN.get("prior_gpu_hours", 0.0)),
                invocation_started_monotonic=float(
                    _ACTIVE_RUN.get("invocation_started_monotonic", time.monotonic())
                ),
                now_monotonic=time.monotonic(),
                world_size=int(_ACTIVE_RUN.get("world_size", 1)),
                original_started_at=str(_ACTIVE_RUN.get("original_started_at", utc_now())),
            )
            failure = {
                "event": "training_failed",
                "timestamp": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc()[-4000:],
                "epoch": int(_ACTIVE_RUN.get("epoch", 0)),
                "sample_cursor": int(_ACTIVE_RUN.get("sample_cursor", 0)),
                "global_step": int(_ACTIVE_RUN.get("global_step", 0)),
                "wall_seconds": timing["active_wall_seconds"],
                "gpu_hours": timing["gpu_hours"],
            }
            output = Path(_ACTIVE_RUN["output"])
            _append_telemetry(output / "run_events.jsonl", failure)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain R, BP-JEPA, TC, BP-JEPA+TC or a recipe-study variant.")
    parser.add_argument("--config", type=Path, required=True, help="configs/pretrain/*.json")
    parser.add_argument("--clip-manifest", type=Path, required=True, help="from bpjepa-clip-manifest")
    parser.add_argument("--frame-root", type=Path, help="overrides the frame root recorded in the manifest")
    parser.add_argument("--evidence-dir", type=Path, help="required by the four arms: clip_validity.npz, target_table_boundary.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="checkpoint written by this trainer; keeps --output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-optimizer-steps", type=int, help="stop early with a checkpoint (dry runs)")
    parser.add_argument("--world-size", type=int, help="override; the paper used 2 (changes data order per rank)")
    parser.add_argument("--micro-batch-per-gpu", type=int, help="override; the paper used 16")
    parser.add_argument("--effective-global-batch", type=int, help="override; the paper used 32")
    parser.add_argument("--num-workers", type=int, help="data-loader workers per rank (no effect on clip or mask order)")
    parser.add_argument("--allow-subset-corpus", action="store_true", help="smoke runs on a small manifest; not the paper corpus")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
