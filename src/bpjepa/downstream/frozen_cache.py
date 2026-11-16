#!/usr/bin/env python3
"""Cache the frozen online encoder's 192-d spatial class token for every adjacent frame pair of every half.

Both pair phases, plus the repeated-frame pairs at each half's ends, so any native 64-frame window can be assembled.
"""
from __future__ import annotations

from bpjepa.downstream import compat  # noqa: F401

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bpjepa.downstream import frozen as F
from bpjepa.downstream.config import load_downstream_config
from bpjepa.run_record import file_reference, write_run_record
from eztorch.datasets.decoders.frame_soccernet_video import FrameSoccerNetVideo


def load_comedian_online(transformer, export_path):
    from bpjepa.downstream.initialization import source_key_for_target
    current = transformer.state_dict()
    export = torch.load(export_path, map_location="cpu", weights_only=True)["state_dict"]
    keys = F.spatial_keys(current)
    mapping = {k: source_key_for_target(k) for k in keys}
    if len(mapping) != 150 or set(export) != set(mapping.values()):
        raise RuntimeError("COMEDIAN export is not exactly the 150-tensor spatial trunk")
    mapped = {k: export[s] for k, s in mapping.items()}
    transformer.load_state_dict({**current, **mapped}, strict=True)
    assert all(torch.equal(transformer.state_dict()[k], v) for k, v in mapped.items())
    for name, p in transformer.named_parameters():
        if name in mapped:
            p.requires_grad_(False)
    transformer.freeze_spatial = True
    transformer.freeze_spatial_transformer()
    transformer.pos_drop.eval()
    return {"spatial_tensors": len(mapped), "exact": True}


def build_model(cfg, encoder, source, expected_step, device):
    torch.manual_seed(42)
    c = F.config_for_cache(cfg, "unused_registry.json", "unused_output")
    m = instantiate(OmegaConf.create(c["model"]))
    if encoder == "comedian":
        load_comedian_online(m.trunk.transformer, source["path"])
    else:
        F.load_pretrained(m.trunk.transformer, source, expected_step)
    return m.to(device).eval(), instantiate(OmegaConf.create(cfg["model"]["val_transform"]["_args_"][1]))


def halves(frames_dir: Path, split: str):
    rows = json.loads((frames_dir / (split + ".json")).read_text())
    for game in sorted(rows):
        for half, rec in sorted(rows[game]["halves"].items()):
            if rec["fps"] != 2:
                raise ValueError("native 2 fps frames required")
            yield {"game": game, "half": half, "path": str(frames_dir / split / rec["UrlLocal"]),
                   "frames": int(rec["num_frames"]), "split": split}


def decoder(info):
    return FrameSoccerNetVideo(video_path=Path(info["path"]).parent, half_path=Path(info["path"]),
                               duration=info["frames"] / 2, fps_video=2, fps=2, num_frames=info["frames"],
                               min_clip_duration=0, num_decode=1, num_threads_io=4, num_threads_decode=4)


def preprocess(transform, clip, device):
    batch = {"input": clip["video"][None].to(device), "timestamps": clip["timestamps"][None].to(device)}
    return transform(batch)["input"]


def pair_features(model, pixels):
    pairs = torch.stack((pixels[0, :, :-1], pixels[0, :, 1:]), 2).permute(1, 0, 2, 3, 4).contiguous()
    with torch.no_grad():
        t = model.trunk.transformer
        chunks = []
        for chunk in pairs.split(64):
            z, _ = type(t).forward_spatial(t, chunk)
            chunks.append(z[:, 0].cpu().numpy())
    return np.concatenate(chunks)


def extract(out: Path, model, transform, source, frames_dir: Path, device, limit_halves=None):
    out.mkdir(parents=True, exist_ok=False)
    doc = {"artifact_kind": "frozen_spatial_full_half_cache", "dtype": "float32", "width": 192, "encoder": source, "halves": {},
           "sampling": "every adjacent 2 fps pair plus first/last repeated pairs",
           "preprocessing": "native 100F validation transform: resize 224, antialias true, /255, mean .45, std .225"}
    t0 = time.time()
    selection = [h for split in ["train", "valid"] for h in halves(frames_dir, split)]
    if limit_halves:
        selection = selection[:limit_halves]
    for hi, info in enumerate(selection):
        n = info["frames"]
        video = decoder(info)
        bank = np.empty((n + 1, 192), np.float32)
        filled = np.zeros(n + 1, bool)
        for start in range(0, n - 1, 62):
            end = min(start + 64, n)
            clip = video.get_clip(start / 2, end / 2)
            pix = preprocess(transform, clip, device)
            z = pair_features(model, pix)
            bank[start:start + len(z)] = z
            filled[start:start + len(z)] = True
            if start == 0:
                bank[n - 1] = pair_features(model, torch.cat((pix[:, :, :1], pix[:, :, :1]), dim=2))[0]
                filled[n - 1] = True
            if end == n:
                bank[n] = pair_features(model, torch.cat((pix[:, :, -1:], pix[:, :, -1:]), dim=2))[0]
                filled[n] = True
        assert filled.all(), ("unfilled rows", info, int((~filled).sum()))
        if not np.isfinite(bank).all():
            raise ValueError("nonfinite cache")
        name = f"half_{hi:04d}.npy"
        with (out / name).open("xb") as f:
            np.save(f, bank, allow_pickle=False)
        doc["halves"][info["path"]] = {"file": name, "sha256": F.sha(out / name), "frames": n, "filled_rows": int(filled.sum()),
                                       "game": info["game"], "half": info["half"], "split": info["split"]}
        print("HALF", hi + 1, len(selection), "seconds", round(time.time() - t0), flush=True)
    doc["seconds"] = time.time() - t0
    with (out / "registry.json").open("x") as f:
        json.dump(doc, f, indent=2)
    return out / "registry.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/downstream/finetune_100f.yaml")
    parser.add_argument("--encoder", choices=("jepa", "comedian"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="jepa: final pretraining checkpoint")
    parser.add_argument("--export", type=Path, help="comedian: comedian_online_spatial.pt from bpjepa-export comedian")
    parser.add_argument("--expected-step", type=int, default=81900)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--label-cache-dir", type=Path, required=True)
    parser.add_argument("--soccernet-labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-halves", type=int, help="smoke runs only")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("cache extraction uses exactly one GPU; set CUDA_VISIBLE_DEVICES")
    if (args.encoder == "jepa") != (args.checkpoint is not None) or (args.encoder == "comedian") != (args.export is not None):
        raise ValueError("--encoder jepa needs --checkpoint; --encoder comedian needs --export")
    cfg = load_downstream_config(args.config, frames_dir=args.frames_dir.resolve(), label_cache_dir=args.label_cache_dir.resolve(),
                                 soccernet_labels_dir=args.soccernet_labels_dir.resolve())
    source_path = (args.checkpoint or args.export).resolve()
    source = {"path": str(source_path), "sha256": F.sha(source_path)}
    model, transform = build_model(cfg, args.encoder, source, args.expected_step, device)
    registry = extract(args.output_dir, model, transform, source, args.frames_dir.resolve(), device, args.limit_halves)
    write_run_record(args.output_dir, stage="frozen_cache", config=cfg, seed=42, inputs={"encoder": file_reference(source_path)},
                     outputs={"registry": file_reference(registry)})
    print("CACHE_COMPLETE", registry, flush=True)


if __name__ == "__main__":
    main()
