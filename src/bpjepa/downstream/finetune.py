#!/usr/bin/env python3
"""Full fine-tuning (100F): pretrained spatial encoder + fresh temporal transformer and classifier, trained end to end.

    torchrun --standalone --nproc_per_node=2 -m bpjepa.downstream.finetune --config configs/downstream/finetune_100f.yaml \\
        --init jepa --checkpoint runs/bp_jepa/checkpoints/epoch-100-step-000081900.pt \\
        --export runs/bp_jepa/exports/comedian_spatial_context.pt --output-dir runs/bp_jepa_100f \\
        --frames-dir data/frames --label-cache-dir data/label_cache --soccernet-labels-dir data/soccernet/valid
"""
from __future__ import annotations

from bpjepa.downstream import compat  # noqa: F401  (before lightning/eztorch)

import argparse
import copy
import json
import os
from pathlib import Path

import hydra
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

from bpjepa.downstream.config import apply_overrides, load_downstream_config, write_config
from bpjepa.downstream.initialization import SpatialInitSpottingModel, restore_rng_state
from bpjepa.run_record import file_reference, write_run_record
from eztorch.models.soccernet_spotting import SoccerNetSpottingModel
from eztorch.utils.utils import compile_model

MODELS = {"jepa": "bpjepa.downstream.initialization.SpatialInitSpottingModel",
          "scratch": "bpjepa.downstream.initialization.SpatialInitSpottingModel",
          "comedian": "bpjepa.downstream.comedian_initialization.ComedianSpottingModel",
          "comedian-a6000": "bpjepa.downstream.comedian_initialization.ComedianA6000SpottingModel"}


def spatial_init(args, run_dir: Path) -> dict:
    """Paper settings: fixed final checkpoints (JEPA epoch 100 / step 81,900; COMEDIAN epoch index 99 / step 322,300)."""
    fields = {"strategy": "scratch" if args.init == "scratch" else "spatial_only", "seed": args.seed,
              "run_id": run_dir.name, "activation_atol": 2e-05, "activation_rtol": 0.0002,
              "initialization_audit_path": str(run_dir / "initialization_audit.json"),
              "runtime_audit_dir": str(run_dir / "runtime_segments"), "gradient_audit_path": None}
    if args.init == "scratch":
        return fields
    if not args.checkpoint:
        raise ValueError(f"--init {args.init} requires --checkpoint")
    fields["checkpoint_path"] = str(args.checkpoint)
    fields["expected_step"] = args.expected_step
    if args.init == "jepa":
        if not args.export:
            raise ValueError("--init jepa requires --export (written at the end of pretraining or by bpjepa-export jepa)")
        fields.update(expected_epoch=args.expected_epoch, export_path=str(args.export),
                      verification_path=str(args.verification or f"{args.export}.verification.pt"))
    else:
        fields.update(source_type="comedian_stage1_sce", expected_epoch=99 if args.expected_epoch == 100 else args.expected_epoch,
                      expected_completed_epochs=100)
        if args.init == "comedian":
            if not args.export:
                raise ValueError("--init comedian requires --export from bpjepa-export comedian")
            fields.update(export_path=str(args.export),
                          verification_path=str(args.verification or Path(args.export).with_name("comedian_online_spatial.verification.pt")))
    return fields


class ResumeSpottingModel(SoccerNetSpottingModel):
    """Continuation from a completed-epoch checkpoint.

    Exact per-rank RNG continuation needs resume_rng_by_rank, which this release's checkpoints carry. Checkpoints with
    only rank 0's state continue rank 1 from a reproducible distinct stream, not an exact replay.
    """

    def on_load_checkpoint(self, value):
        self._resume_rng = value.get("resume_rng_by_rank")
        self._legacy_rng = value["table5_rng_state"]
        # Callbacks save inside epoch-end/validation-end hooks, before Lightning marks the epoch processed.
        # All updates of the epoch are saved. Complete only this bookkeeping so no batches or validation are replayed.
        loop = value["loops"]["fit_loop"]
        completed = value["epoch"] + 1
        # On restart, Lightning 2.0.9 validates a validation epoch without training it, losing that epoch. The paper's 5/10 cadence never hits this.
        if (completed + 1) % self.trainer.check_val_every_n_epoch == 0:
            raise RuntimeError(f"cannot resume from epoch={value['epoch']}: the next epoch validates, so Lightning would skip its training; use another checkpoint")
        for scope in ["current", "total"]:
            progress = loop["epoch_progress"][scope]
            if progress["ready"] != completed or progress["started"] != completed:
                raise RuntimeError("expected a completed-epoch checkpoint written by the epoch callback")
            progress["processed"] = progress["completed"] = completed
        loop["epoch_loop.state_dict"]["_batches_that_stepped"] = value["global_step"]

    def on_save_checkpoint(self, value):
        from bpjepa.downstream.initialization import capture_rng_state
        state = capture_rng_state()
        states = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(states, state)
        value["table5_rng_state"] = state
        value["resume_rng_by_rank"] = states

    def on_train_start(self):
        super().on_train_start()
        if self._resume_rng:
            restore_rng_state(self._resume_rng[self.global_rank])
        else:
            # Older checkpoints contain rank 0 only; rank 1 gets a reproducible distinct stream, not a claimed exact replay.
            # CUDA generator state size also changed between Torch 2.0 and 2.11.
            restore_rng_state(dict(self._legacy_rng, torch_cuda=[]))
            torch.cuda.manual_seed_all(self._legacy_seed + self.global_rank + self.global_step)
            if self.global_rank:
                from lightning.pytorch import seed_everything
                seed_everything(self._legacy_seed + self.global_rank + self.global_step, workers=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full fine-tuning (100F) of a pretrained spatial encoder.")
    parser.add_argument("--config", type=Path, required=True, help="configs/downstream/finetune_100f.yaml")
    parser.add_argument("--init", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, help="final pretraining checkpoint")
    parser.add_argument("--export", type=Path, help="spatial export")
    parser.add_argument("--verification", type=Path, help="activation check file; default: next to the export")
    parser.add_argument("--expected-epoch", type=int, default=100)
    parser.add_argument("--expected-step", type=int, default=81900, help="COMEDIAN Stage 1: 322300")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--label-cache-dir", type=Path, required=True)
    parser.add_argument("--soccernet-labels-dir", type=Path, required=True, help="validation split: <match>/Labels-v2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="the paper's 100F seed")
    parser.add_argument("--devices", type=int, help="GPUs; the paper used 2 (see README before changing)")
    parser.add_argument("--num-workers", type=int, help="data-loader workers per rank; the paper used 8")
    parser.add_argument("--resume", type=Path, help="epoch=N.ckpt; continues into --output-dir")
    parser.add_argument("--accelerator", default="gpu", help="the paper used gpu; cpu (fp32, one process) is for bounded checks")
    parser.add_argument("--max-steps", type=int, help="stop early without validation (smoke runs); not a paper run")
    parser.add_argument("--override", action="append", metavar="KEY=VALUE", help="set a config value (smoke runs); not a paper run")
    parser.add_argument("--check-only", action="store_true", help="build everything on CPU, write the audit, then stop before fit")
    args = parser.parse_args()
    run_dir = args.output_dir.resolve()
    config = load_downstream_config(args.config, frames_dir=args.frames_dir.resolve(), label_cache_dir=args.label_cache_dir.resolve(),
                                    soccernet_labels_dir=args.soccernet_labels_dir.resolve(), num_workers=args.num_workers, devices=args.devices)
    apply_overrides(config, args.override)
    config["seed"]["seed"] = args.seed
    config["model"]["_target_"] = MODELS[args.init]
    if args.init == "comedian-a6000" or args.init == "comedian":
        args.expected_step = 322300 if args.expected_step == 81900 else args.expected_step
    if args.init == "scratch":
        config["model"]["spatial_init"] = spatial_init(args, run_dir)
    else:
        for name in ("checkpoint", "export", "verification"):
            value = getattr(args, name)
            if value is not None:
                setattr(args, name, value.resolve())
        config["model"]["spatial_init"] = spatial_init(args, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    write_config(config, run_dir / "config/supervised.yaml")
    if int(os.environ.get("RANK", "0")) == 0:
        write_run_record(run_dir, stage="finetune_100f", config=config, seed=args.seed,
                         name="run_record_resume.json" if args.resume else "run_record.json",
                         inputs={"checkpoint": file_reference(args.checkpoint), "export": file_reference(args.export),
                                 "verification": file_reference(args.verification), "resume": file_reference(args.resume)},
                         outputs={"run_dir": str(run_dir)})
    if args.check_only or args.accelerator != "gpu":
        config["trainer"].update(accelerator="cpu" if args.check_only else args.accelerator, devices=1, precision=32, strategy="auto")
    if args.max_steps:
        config["trainer"].update(max_steps=args.max_steps, limit_val_batches=0, num_sanity_val_steps=0)
    cfg = OmegaConf.create(copy.deepcopy(config))
    hydra.utils.instantiate(cfg.seed)  # seed, then callbacks, trainer, datamodule, model: the paper's construction order
    callbacks = [hydra.utils.instantiate(c) for _, c in cfg.callbacks.items()]
    extra = {"num_sanity_val_steps": 0} if args.resume else {}
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, **extra)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    if args.resume:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg.pop("spatial_init")
        model_cfg["_target_"] = "bpjepa.downstream.finetune.ResumeSpottingModel"
        model = hydra.utils.instantiate(OmegaConf.create(model_cfg))
        model._legacy_seed = args.seed
        checkpoint = str(args.resume.resolve())
    else:
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = None
        rank_zero_info(json.dumps(model.initialization_audit.get("spatial_transfer", {}), default=str)[:400])
    model = compile_model(model, **config.get("compile", {}))
    if args.check_only:
        if not args.resume and not Path(config["model"]["spatial_init"]["initialization_audit_path"]).exists():
            from bpjepa.downstream.initialization import write_immutable_json
            write_immutable_json(config["model"]["spatial_init"]["initialization_audit_path"], model.initialization_audit)
        print("CHECK_ONLY_OK", flush=True)
        return
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
