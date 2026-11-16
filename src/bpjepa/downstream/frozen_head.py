#!/usr/bin/env python3
"""Train a fresh temporal transformer and classifier on a frozen-encoder cache. One head seed per run (paper: 42, 203, 666)."""
from __future__ import annotations

from bpjepa.downstream import compat  # noqa: F401

import argparse
import json
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning.pytorch import Callback, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from bpjepa.downstream import frozen as F
from bpjepa.downstream.config import apply_overrides, load_downstream_config
from bpjepa.downstream.frozen_cache import load_comedian_online
from bpjepa.run_record import file_reference, write_run_record


class FrozenCheck(Callback):
    """Runtime validation: spatial weights stay frozen and in eval mode, and only the head trains."""

    def __init__(self, out, source, initial, seed):
        self.out, self.source, self.initial, self.seed = out, source, initial, seed

    def on_train_start(self, trainer, model):
        t = model.trunk.transformer
        assert not t.spatial_transformer.training and not t.patch_embed.training and not t.pos_drop.training
        assert F.frozen_head_hashes(model) == self.initial
        assert all(not p.requires_grad for n, p in t.named_parameters() if n in F.spatial_keys(t.state_dict()))
        record = {"seed": self.seed, "head_hashes": self.initial, "encoder": self.source, "batches": trainer.num_training_batches,
                  "world_size": trainer.world_size, "precision": "32-true", "initial_lr": [g["lr"] for g in trainer.optimizers[0].param_groups]}
        with (self.out / f"initialization_rank{trainer.global_rank}.json").open("w") as f:
            json.dump(record, f, indent=2)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.global_step == 1:
            t = model.trunk.transformer
            assert all(p.grad is None for n, p in t.named_parameters() if n in F.spatial_keys(t.state_dict()))
            assert t.time_embed.grad is not None
            assert any(p.grad is not None for p in t.temporal_transformer.parameters())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="configs/downstream/finetune_100f.yaml (same model, optimizer and schedule)")
    p.add_argument("--cache-registry", type=Path, required=True, help="registry.json from bpjepa-frozen-cache")
    p.add_argument("--encoder", choices=("jepa", "comedian"), required=True)
    p.add_argument("--seed", type=int, required=True, help="paper: 42, 203, 666")
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--label-cache-dir", type=Path, required=True)
    p.add_argument("--soccernet-labels-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-step", type=int, default=81900, help="COMEDIAN Stage 1: 322300")
    p.add_argument("--devices", type=int, help="GPUs; the paper used 2")
    p.add_argument("--num-workers", type=int, default=4, help="the paper used 4")
    p.add_argument("--max-steps", type=int, help="stop early (preflight); not a paper run")
    p.add_argument("--override", action="append", metavar="KEY=VALUE", help="set a config value (smoke runs); not a paper run")
    p.add_argument("--accelerator", default="gpu", help="the paper used gpu; cpu is for bounded checks")
    a = p.parse_args()
    cache = F.PairCache(a.cache_registry)
    source = cache.doc["encoder"]
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank() if distributed else 0
    if rank == 0:
        a.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        torch.distributed.barrier()
    seed_everything(a.seed, workers=True)
    base = load_downstream_config(a.config, frames_dir=a.frames_dir.resolve(), label_cache_dir=a.label_cache_dir.resolve(),
                                  soccernet_labels_dir=a.soccernet_labels_dir.resolve(), devices=a.devices)
    c = F.config_for_cache(base, a.cache_registry.resolve(), a.output_dir.resolve())
    c["trainer"]["precision"] = "32-true"
    if a.accelerator != "gpu":
        c["trainer"].update(accelerator=a.accelerator, devices=1, strategy="auto")
    c["datamodule"]["train"]["loader"]["num_workers"] = a.num_workers
    c["datamodule"]["val"]["loader"]["num_workers"] = a.num_workers
    apply_overrides(c, a.override)
    model = instantiate(OmegaConf.create(c["model"]))
    initial = F.frozen_head_hashes(model)  # same seed -> same head and data order across arms (compare initialization_rank0.json)
    if a.encoder == "comedian":
        load_comedian_online(model.trunk.transformer, source["path"])
    else:
        F.load_pretrained(model.trunk.transformer, source, a.expected_step)
    if a.max_steps:
        c["trainer"].update(max_steps=a.max_steps, limit_val_batches=0, num_sanity_val_steps=0)
    callbacks = [instantiate(OmegaConf.create(v)) for v in c["callbacks"].values()]
    callbacks.append(FrozenCheck(a.output_dir, source, initial, a.seed))
    dm = instantiate(OmegaConf.create(c["datamodule"]))
    logger = TensorBoardLogger(str(a.output_dir / "lightning_logs"), name="", version=0)
    trainer = instantiate(OmegaConf.create(c["trainer"]), callbacks=callbacks, logger=logger, default_root_dir=str(a.output_dir))
    if trainer.is_global_zero:
        write_run_record(a.output_dir, stage="frozen_head", config=c, seed=a.seed,
                         inputs={"cache_registry": file_reference(a.cache_registry), "encoder": source},
                         outputs={"initial_head_hashes": initial})
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
