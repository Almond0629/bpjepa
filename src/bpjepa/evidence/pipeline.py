"""Label-free boundary evidence, stage by stage. Every stage reads the previous stage's files from --evidence-dir.

  bpjepa-evidence measure       per-transition cut/wipe/alignment/mask/motion (CPU, parallel over halves)
  bpjepa-evidence raw-change    DINOv3 raw per-token change C and final floor-4 validity (one GPU)
  bpjepa-evidence state-keys    nuisance state per transition (alignment, motion bin, overlap bin)
  bpjepa-evidence references    calibration references: full calibration set and two cross-fit folds
  bpjepa-evidence targets       per-transition boundary evidence for every half
  bpjepa-evidence tables        target_table_boundary.npz and clip_validity.npz used by pretraining
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

# The evidence runs fixed these before NumPy/OpenCV/CUDA initialize.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np  # noqa: E402

from bpjepa.common import sha256  # noqa: E402
from bpjepa.run_record import write_run_record  # noqa: E402


def half_key(half: str) -> str:
    return hashlib.sha256(half.encode("utf-8")).hexdigest()


def manifest_halves(manifest_path: Path) -> dict:
    """half_id -> {frame_dir, frame_count (last clip end)} from the training clip manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    halves = {h: {"frame_dir": None, "frame_count": 0} for h in manifest["half_ids"]}
    for e in manifest["entries"]:
        rec = halves[e["half_id"]]
        rec["frame_dir"] = e["frame_dir"]
        rec["frame_count"] = max(rec["frame_count"], e["end_frame_exclusive"])
    return halves


def _write_npz(path: Path, metadata: dict | None, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        if metadata is not None:
            arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_npz(path: Path) -> tuple[dict | None, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item())) if "metadata" in z.files else None
        return meta, {k: z[k] for k in z.files if k != "metadata"}


def measurement_path(evidence: Path, half: str) -> Path:
    return Path(evidence) / "measurements" / f"{half_key(half)}.npz"


def raw_path(evidence: Path, half: str) -> Path:
    return Path(evidence) / "raw_change" / f"{half_key(half)}.npz"


def state_path(evidence: Path, half: str) -> Path:
    return Path(evidence) / "state_keys" / f"{half_key(half)}.npz"


def target_path(evidence: Path, half: str) -> Path:
    return Path(evidence) / "targets" / f"{half_key(half)}.npz"


def load_measurements(evidence: Path, half: str) -> tuple[dict, dict]:
    meta, arrays = _load_npz(measurement_path(evidence, half))
    packed = arrays.pop("mask_packed")
    arrays["mask"] = np.unpackbits(packed, axis=1, bitorder="little")[:, :196].astype(bool).reshape(-1, 14, 14)
    return meta, arrays


def _measure_one(task: dict) -> str:
    import cv2
    from bpjepa.evidence.measurements import measure_half
    cv2.setNumThreads(1)
    path = Path(task["path"])
    if path.exists():
        return task["half"]
    metadata, arrays = measure_half(Path(task["frame_dir"]), task["frame_count"], task["half"])
    arrays = dict(arrays)
    arrays["mask_packed"] = np.packbits(arrays.pop("mask").reshape(len(arrays["n_valid"]), 196), axis=1, bitorder="little")
    _write_npz(path, metadata, **arrays)
    return task["half"]


def cmd_measure(args) -> None:
    halves = manifest_halves(args.clip_manifest)
    tasks = [{"half": h, "frame_dir": str(args.frame_root / r["frame_dir"]), "frame_count": r["frame_count"],
              "path": str(measurement_path(args.evidence_dir, h))} for h, r in halves.items()]
    if args.limit_halves:
        tasks = tasks[:args.limit_halves]
    if args.workers == 1:
        done = list(map(_measure_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            done = list(executor.map(_measure_one, tasks))
    write_run_record(args.evidence_dir, stage="evidence_measure", config=vars_json(args), seed=None, name="run_record_measure.json",
                     outputs={"halves": len(done)})
    print(json.dumps({"stage": "measure", "halves": len(done)}))


def cmd_raw_change(args) -> None:
    import torch
    from bpjepa.evidence.dinov3_runtime import SETTINGS, V2Teacher, configure
    from bpjepa.evidence.raw_change import half_arrays
    from bpjepa.evidence.teacher import TeacherConfig
    environment = configure()
    teacher_fields = json.loads(Path(args.teacher_config).read_text())
    teacher_fields["checkpoint_path"] = str(args.dinov3_checkpoint)
    config = TeacherConfig.from_mapping(teacher_fields)
    teacher = None
    halves = manifest_halves(args.clip_manifest)
    names = list(halves)[:args.limit_halves] if args.limit_halves else list(halves)
    for number, half in enumerate(names, 1):
        path = raw_path(args.evidence_dir, half)
        if path.exists():
            continue
        metadata, measurements = load_measurements(args.evidence_dir, half)
        if metadata["half_id"] != half or metadata["frame_count"] != halves[half]["frame_count"]:
            raise RuntimeError(f"measurements do not match the clip manifest: {half}")
        started = time.perf_counter()
        if teacher is None:
            teacher = V2Teacher(config, args.dinov3_repo)
        arrays = half_arrays(metadata, measurements, args.frame_root / halves[half]["frame_dir"], teacher, SETTINGS["teacher_batch_size"])
        _write_npz(path, {"half_id": half, "environment": environment}, **arrays)
        print(json.dumps({"stage": "raw_change", "half": number, "of": len(names), "seconds": time.perf_counter() - started,
                          "valid": int((arrays["states"] == 1).sum())}), flush=True)
    write_run_record(args.evidence_dir, stage="evidence_raw_change", config=vars_json(args), seed=None, name="run_record_raw_change.json",
                     inputs={"dinov3_checkpoint": {"path": str(args.dinov3_checkpoint), "sha256": sha256(args.dinov3_checkpoint)}},
                     outputs={"environment": environment, "halves": len(names)})
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_state_keys(args) -> None:
    from bpjepa.evidence.regime import state_keys
    halves = list(manifest_halves(args.clip_manifest))
    halves = halves[:args.limit_halves] if args.limit_halves else halves
    for half in halves:
        _, m = load_measurements(args.evidence_dir, half)
        _write_npz(state_path(args.evidence_dir, half), {"half_id": half, "state_fields": ["alignment", "motion", "overlap"]},
                   state_keys=state_keys(m["alignment_status"], m["motion_bin"], m["n_valid"]))
    print(json.dumps({"stage": "state_keys", "halves": len(halves)}))


def load_half_for_regime(evidence: Path, half: str):
    """raw C [n,196] float64, token mask, final valid, reasons, nuisance keys."""
    _, m = load_measurements(evidence, half)
    _, r = _load_npz(raw_path(evidence, half))
    raw = r["raw_c"].reshape(-1, 196).astype(np.float64)
    mask = m["mask"].reshape(-1, 196)
    valid = r["states"] == 1
    assert np.isfinite(raw[mask & valid[:, None]]).all()
    assert np.all((raw[mask & valid[:, None]] >= 0) & (raw[mask & valid[:, None]] <= 2))
    _, s = _load_npz(state_path(evidence, half))
    return raw, mask, valid, r["reasons"], s["state_keys"]


def cmd_references(args) -> None:
    from bpjepa.evidence.regime import construct_folds, fit_references
    policy = json.loads(Path(args.calibration_policy).read_text())
    calibration = json.loads(Path(args.roles).read_text())["calibration_half_ids"]
    loader = lambda half: load_half_for_regime(args.evidence_dir, half)  # noqa: E731
    root = args.evidence_dir / "references"
    fit_references(loader, calibration, policy["reference_support"], root / "full", log=print)
    folds = construct_folds([h.rsplit("::", 1)[0] for h in calibration], policy["folds"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "folds.json").write_text(json.dumps(folds, indent=2))
    for direction in (0, 1):
        members = [h for h in calibration if h.rsplit("::", 1)[0] in folds[direction]]
        fit_references(loader, members, policy["folds"]["token_support_each_training_and_heldout_pool"],
                       root / f"fold_{direction}", log=print)
    write_run_record(args.evidence_dir, stage="evidence_references", config=vars_json(args), seed=None, name="run_record_references.json")


def role_a_reference_set(half: str, folds: list) -> str:
    """Fold d references score the matches held out from it, i.e. those of fold 1-d."""
    match = half.rsplit("::", 1)[0]
    held = [d for d in (0, 1) if match in folds[1 - d]]
    if len(held) != 1:
        raise RuntimeError(f"calibration match not in exactly one fold: {match}")
    return f"fold_{held[0]}"


def cmd_targets(args) -> None:
    from bpjepa.evidence.regime import boundary_half, half_arrays, load_references
    roles = json.loads(Path(args.roles).read_text())
    calibration = set(roles["calibration_half_ids"])
    root = args.evidence_dir / "references"
    folds = json.loads((root / "folds.json").read_text())
    refs = {name: load_references(root / name) for name in ("full", "fold_0", "fold_1")}
    halves = list(manifest_halves(args.clip_manifest))
    halves = halves[:args.limit_halves] if args.limit_halves else halves
    for half in halves:
        which = role_a_reference_set(half, folds) if half in calibration else "full"
        score, reliable, _, _ = boundary_half(*load_half_for_regime(args.evidence_dir, half), refs[which])
        _write_npz(target_path(args.evidence_dir, half), {"half_id": half, "target": "boundary", "reference_set": which},
                   score64=np.asarray(score, np.float64), **half_arrays(score, reliable))
    print(json.dumps({"stage": "targets", "halves": len(halves)}))


def cmd_tables(args) -> None:
    from bpjepa.evidence.target_table import build_target_table
    from bpjepa.pretrain.clip_validity import build_clip_validity
    halves = list(manifest_halves(args.clip_manifest))
    final, targets = {}, {}
    for half in halves:
        _, r = _load_npz(raw_path(args.evidence_dir, half))
        _, t = _load_npz(target_path(args.evidence_dir, half))
        final[half] = {"states": r["states"], "reasons": r["reasons"]}
        targets[half] = {k: t[k] for k in ("value", "supported")}
    report = {"clip_validity": build_clip_validity(args.evidence_dir / "clip_validity.npz", manifest_path=args.clip_manifest, final_validity=final),
              "target_table": build_target_table(args.evidence_dir / "target_table_boundary.npz", manifest_path=args.clip_manifest,
                                                 half_targets=targets, final_validity=final)}
    write_run_record(args.evidence_dir, stage="evidence_tables", config=vars_json(args), seed=None, name="run_record_tables.json", outputs=report)
    print(json.dumps(report))


def vars_json(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    def stage(name, func, help_text, frames=False, roles=False):
        p = stages.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument("--clip-manifest", type=Path, required=True)
        p.add_argument("--evidence-dir", type=Path, required=True)
        p.add_argument("--limit-halves", type=int, help="process only the first N halves (smoke runs); honoured by measure, raw-change, state-keys and targets, ignored by references and tables")
        if frames:
            p.add_argument("--frame-root", type=Path, required=True)
        if roles:
            p.add_argument("--roles", type=Path, required=True, help="configs/evidence/roles.json")
        return p

    stage("measure", cmd_measure, "per-transition measurements (CPU)", frames=True).add_argument("--workers", type=int, default=1)
    raw = stage("raw-change", cmd_raw_change, "DINOv3 raw change C (one GPU)", frames=True)
    raw.add_argument("--dinov3-repo", type=Path, required=True, help="facebookresearch/dinov3 checkout, see docs/ARTIFACTS.md")
    raw.add_argument("--dinov3-checkpoint", type=Path, required=True)
    raw.add_argument("--teacher-config", type=Path, required=True, help="configs/evidence/teacher.json")
    stage("state-keys", cmd_state_keys, "nuisance state per transition")
    stage("references", cmd_references, "calibration references", roles=True).add_argument("--calibration-policy", type=Path, required=True, help="configs/evidence/calibration.json")
    stage("targets", cmd_targets, "per-half boundary evidence", roles=True)
    stage("tables", cmd_tables, "clip-level tables for pretraining")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
