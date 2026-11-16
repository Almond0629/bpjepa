#!/usr/bin/env python3
"""Pooled small-group features of pretrained encoders on the probe clips: identical clips and masks for all arms.

The arms are loaded together so one job encodes every clip with every encoder. Features are stored as fp16.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from bpjepa.probes import lib as L
from bpjepa.run_record import file_reference, write_run_record

BATCH, DRAWS, T, DIM, SPATIAL = 16, 4, 64, 192, 196


def load_encoder(kind, path, expect_step=None):
    """A pretraining checkpoint's online encoder, or a COMEDIAN Stage-1 online-trunk export."""
    import torch
    from bpjepa.jepa.model import ComedianViTTiny
    enc = ComedianViTTiny(image_size=224, patch_size=16, tube_size=2, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, drop_path_rate=0.1)
    if kind == 'COMEDIAN':
        export = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        enc.load_state_dict({k.removeprefix('trunk.'): v for k, v in export.items()}, strict=True)
    else:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if expect_step is not None and payload['global_step'] != expect_step:
            raise RuntimeError(f'checkpoint global_step {payload["global_step"]} != {expect_step}')
        prefix = 'context_encoder.'
        enc.load_state_dict({k[len(prefix):]: v for k, v in payload['model'].items() if k.startswith(prefix)}, strict=True)
    return enc.eval()


def needed(data):
    """clip index -> (pool position or -1, eval position or -1); draws needed per clip."""
    pool_pos = {int(i): k for k, i in enumerate(data.pool)}
    ev_pos = {int(i): k for k, i in enumerate(data.ev)}
    return pool_pos, ev_pos


def extract(encs, device, data, manifest_path, frame_root=None, allow_subset=False, workers=2, max_batches=None, log=print):
    """encs: {arm name: eval-mode encoder on `device`}; every clip is loaded once and encoded by every encoder with the
    identical masks. -> ({arm: (z_pool [512,64,192] fp16 draw 0, z_eval [4,1440,64,192] fp16)}, fill_pool, fill_eval,
    mask mismatches against the stored Phase 1 masks)."""
    import torch
    from bpjepa.jepa.full_data import FullSoccerNetClipDataset
    from bpjepa.jepa.multimask import indexed_context
    from bpjepa.jepa.support import GroupSupport, group_support, pool_student
    pool_pos, ev_pos = needed(data)
    masks = L.StoredSmallMasks(data.visible, data.complete)
    total = len(data.manifest_rows) // BATCH
    out = {name: (np.zeros((len(data.pool), T, DIM), np.float16), np.zeros((DRAWS, len(data.ev), T, DIM), np.float16))
           for name in encs}
    fill_pool = np.zeros(len(data.pool), bool)
    fill_eval = np.zeros((DRAWS, len(data.ev)), bool)
    batches, flat = [], []
    for b in range(total):
        pos = [i for i in range(b * BATCH, (b + 1) * BATCH) if i in pool_pos or i in ev_pos]
        if pos and (max_batches is None or len(batches) < max_batches):
            batches.append((b, pos))
            flat.extend(int(data.manifest_rows[i]) for i in pos)
    sizes = np.cumsum([0] + [len(p) for _, p in batches])
    dataset = torch.utils.data.Subset(FullSoccerNetClipDataset(manifest_path, frame_root=frame_root, require_full_corpus=not allow_subset), flat)
    loader = iter(torch.utils.data.DataLoader(
        dataset, batch_sampler=[list(range(sizes[k], sizes[k + 1])) for k in range(len(batches))],
        num_workers=workers, pin_memory=device.type == 'cuda'))
    by_batch = {b: p for b, p in batches}
    last = batches[-1][0] if batches else -1
    mask_mismatch = 0
    done = 0
    for b in range(last + 1):
        if b not in by_batch:
            continue
        smalls = [masks.small(d, b) for d in range(DRAWS)]
        pos = by_batch[b]
        clips = next(loader).to(device, non_blocking=True)
        for d in range(DRAWS):
            sel = [k for k, i in enumerate(pos) if d == 0 or i in ev_pos]
            if not sel:
                continue
            small = smalls[d]
            assert small.name == 'small'
            full = group_support(small, T, 14, 14)
            rows = torch.tensor([pos[k] - b * BATCH for k in sel])
            support = GroupSupport(full.visible[rows], full.complete[rows], full.partial[rows], full.support_size[rows])
            ci = small.context_indices[rows].to(device)
            sel_clips = clips[torch.tensor(sel, device=device)]
            for name, enc in encs.items():
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    feats = indexed_context(enc, sel_clips, ci)
                with torch.no_grad():
                    pooled = pool_student(feats.float(), ci, support.to(device))
                assert torch.isfinite(pooled).all()
                pooled = pooled.cpu().numpy().astype(np.float16)
                z_pool, z_eval = out[name]
                for j, k in enumerate(sel):
                    i = pos[k]
                    if d == 0 and i in pool_pos:
                        z_pool[pool_pos[i]] = pooled[j]
                    if i in ev_pos:
                        z_eval[d, ev_pos[i]] = pooled[j]
            for j, k in enumerate(sel):
                i = pos[k]
                mask_mismatch += int(not np.array_equal(support.visible[j].numpy(), data.visible[d, i])
                                     or not np.array_equal(support.complete[j].numpy(), data.complete[d, i]))
                if d == 0 and i in pool_pos:
                    fill_pool[pool_pos[i]] = True
                if i in ev_pos:
                    fill_eval[d, ev_pos[i]] = True
        done += 1
        if done % 20 == 0:
            log(f'extract batch {done}/{len(batches)} (global {b + 1}/{total}) {time.strftime("%H:%M:%S")}')
    return out, fill_pool, fill_eval, mask_mismatch


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip-table', type=Path, required=True, help='probe_clips.npz from bpjepa-probes-prepare')
    ap.add_argument('--clip-manifest', type=Path, required=True)
    ap.add_argument('--frame-root', type=Path)
    ap.add_argument('--arm', action='append', required=True, metavar='NAME=PATH',
                    help='R0, U, EB, EB0 = final pretraining checkpoint; COMEDIAN = online-trunk export')
    ap.add_argument('--expected-step', type=int, default=81900)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-batches', type=int, help='smoke runs only')
    ap.add_argument('--allow-subset-corpus', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)
    data = L.load_clip_table(args.clip_table)
    arms = dict(item.split('=', 1) for item in args.arm)
    encs = {name: load_encoder(name, path, None if name == 'COMEDIAN' else args.expected_step).to(device) for name, path in arms.items()}
    out, fill_pool, fill_eval, mismatch = extract(encs, device, data, args.clip_manifest, args.frame_root, args.allow_subset_corpus,
                                                  workers=args.workers, max_batches=args.max_batches)
    if mismatch:
        raise RuntimeError(f'{mismatch} mask rows differ from the clip table')
    if args.max_batches is None and not (fill_pool.all() and fill_eval.all()):
        raise RuntimeError('some probe clips were not encoded')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, (z_pool, z_eval) in out.items():
        np.savez(args.output_dir / f'features_{name}.npz', z_pool=z_pool, z_eval=z_eval)
    write_run_record(args.output_dir, stage='probe_features', config=vars_config(args), seed=int(L.SEED),
                     inputs={name: file_reference(path) for name, path in arms.items()},
                     outputs={name: str(args.output_dir / f'features_{name}.npz') for name in out})


def vars_config(args):
    return {k: str(v) for k, v in vars(args).items()}


if __name__ == '__main__':
    raise SystemExit(main())
