#!/usr/bin/env python3
"""Spatial-encoder export: JEPA pretraining checkpoint or COMEDIAN Stage-1 checkpoint -> tensor-only state dict.

Only the online encoder is exported. EMA targets, predictors, heads, projectors and queues are never transferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bpjepa.common import atomic_json, sha256

EXPECTED_ONLINE_TENSORS = 150


def export_jepa(checkpoint: Path, output: Path, clip_manifest: Path, frame_root: Path | None, allow_subset: bool) -> dict:
    from bpjepa.jepa.full_data import FullSoccerNetClipDataset
    from bpjepa.jepa.model import ComedianViTTiny
    from bpjepa.jepa.transfer import export_comedian
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "context_encoder."
    state = {k[len(prefix):]: v for k, v in payload["model"].items() if k.startswith(prefix)}
    encoder = ComedianViTTiny(image_size=224, patch_size=16, tube_size=2, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, drop_path_rate=0.1)
    encoder.load_state_dict(state, strict=True)
    clip = FullSoccerNetClipDataset(clip_manifest, frame_root=frame_root, require_full_corpus=not allow_subset)[0][None]
    verification = export_comedian(encoder.eval(), output, clip)
    report = {"status": "PASS", "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
              "epoch": int(payload["epoch"]), "global_step": int(payload["global_step"]),
              "export": {"path": str(output), "sha256": sha256(output)},
              "verification": {"path": str(verification), "sha256": sha256(verification)}}
    atomic_json(output.with_name("export_record.json"), report)
    return report


def export_comedian_online(checkpoint: Path, output_dir: Path, *, expected_epoch: int = 99, expected_step: int | None = 322300) -> dict:
    from eztorch.models.trunks import create_vit_tiny
    from bpjepa.downstream.initialization import verify_export_activation
    source_hash = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if expected_step is not None and (payload.get("epoch") != expected_epoch or payload.get("global_step") != expected_step):
        raise RuntimeError(f"expected epoch {expected_epoch} / step {expected_step}, got {payload.get('epoch')} / {payload.get('global_step')}")
    state = payload.get("state_dict", {})
    if not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise RuntimeError("checkpoint state_dict must hold string-keyed tensors")
    online = {k: v.detach().clone().cpu() for k, v in state.items() if k.startswith("trunk.")}
    if len(online) != EXPECTED_ONLINE_TENSORS:
        raise RuntimeError(f"expected {EXPECTED_ONLINE_TENSORS} online trunk tensors, found {len(online)}")
    allowed = ("momentum_trunk.", "projector.", "predictor.", "momentum_projector.", "train_transform.")
    unclassified = [k for k in state if k not in online and k not in {"queue", "queue_ptr"} and not k.startswith(allowed)]
    if unclassified:
        raise RuntimeError(f"unclassified source tensors: {unclassified}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261006)
        vit = create_vit_tiny(img_size=224, patch_size=16, in_chans=3, mlp_ratio=4, qkv_bias=True, init_values=None,
                              class_token=True, no_embed_class=False, pre_norm=False, drop_rate=0, attn_drop_rate=0,
                              drop_path_rate=0.1, conv_type="Conv3d", tube_size=2).eval()
        vit.load_state_dict({k.removeprefix("trunk."): v for k, v in online.items()}, strict=True)
        sample = torch.randn(1, 3, 2, 224, 224)
        with torch.no_grad():
            activation = vit(sample)
    output_dir.mkdir(parents=True, exist_ok=False)
    export_path = output_dir / "comedian_online_spatial.pt"
    verification_path = output_dir / "comedian_online_spatial.verification.pt"
    torch.save({"state_dict": online}, export_path)
    torch.save({"input": sample, "output": activation}, verification_path)
    with torch.random.fork_rng(devices=[]):
        activation_report = verify_export_activation(export_path, verification_path)
    report = {"status": "PASS", "source_checkpoint": {"path": str(checkpoint), "sha256": source_hash},
              "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
              "export": {"path": str(export_path), "sha256": sha256(export_path)},
              "verification": {"path": str(verification_path), "sha256": sha256(verification_path)}, "activation": activation_report}
    atomic_json(output_dir / "extraction.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kinds = parser.add_subparsers(dest="kind", required=True)
    jepa = kinds.add_parser("jepa", help="context encoder of a pretraining checkpoint")
    jepa.add_argument("--checkpoint", type=Path, required=True)
    jepa.add_argument("--output", type=Path, required=True)
    jepa.add_argument("--clip-manifest", type=Path, required=True)
    jepa.add_argument("--frame-root", type=Path)
    jepa.add_argument("--allow-subset-corpus", action="store_true")
    comedian = kinds.add_parser("comedian", help="online trunk of a COMEDIAN Stage-1 checkpoint")
    comedian.add_argument("--checkpoint", type=Path, required=True)
    comedian.add_argument("--output-dir", type=Path, required=True)
    comedian.add_argument("--expected-epoch", type=int, default=99)
    comedian.add_argument("--expected-step", type=int, default=322300)
    args = parser.parse_args()
    if args.kind == "jepa":
        report = export_jepa(args.checkpoint, args.output, args.clip_manifest, args.frame_root, args.allow_subset_corpus)
    else:
        from bpjepa.downstream import compat  # noqa: F401
        report = export_comedian_online(args.checkpoint, args.output_dir, expected_epoch=args.expected_epoch, expected_step=args.expected_step)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
