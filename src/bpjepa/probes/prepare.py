#!/usr/bin/env python3
"""Build probe_clips.npz (clip table, targets, mask supports, zoom) from the evidence directory."""
import argparse
from pathlib import Path

from bpjepa.probes import lib as L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clip-manifest', type=Path, required=True)
    ap.add_argument('--roles', type=Path, required=True, help='configs/evidence/roles.json')
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--probe-masks', type=Path, required=True, help='configs/probes/probe_masks.npz')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    print(L.prepare_clip_table(args.output, manifest_path=args.clip_manifest, roles_path=args.roles, evidence_dir=args.evidence_dir, probe_masks=args.probe_masks))


if __name__ == '__main__':
    raise SystemExit(main())
