"""Clip-level boundary-target table: per-half targets + final-validity CUT bits -> per-row y, supported, side_ok, dz_ok."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bpjepa.common import sha256
from bpjepa.evidence.transition_grid import Reason
from bpjepa.jepa.boundary import AuxTable

TEMPORAL, TRANSITIONS = 64, 127
TABLE_KIND = "bpjepa_boundary_target_table"
KIND_FOR_TARGET = {"boundary": "EB"}


def clip_targets(value: np.ndarray, supported: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """t=2..60: max over supported {2t, 2t+1}; t=61: 2t only; other t unsupported."""
    if value.shape != (TRANSITIONS,) or supported.shape != (TRANSITIONS,):
        raise ValueError("clip targets must cover 127 transitions")
    y = np.full(TEMPORAL, np.nan, np.float32)
    ok = np.zeros(TEMPORAL, bool)
    for t in range(2, 62):
        candidates = [j for j in ((2 * t, 2 * t + 1) if t <= 60 else (2 * t,)) if supported[j]]
        if candidates:
            y[t], ok[t] = max(value[j] for j in candidates), True
    return y, ok


def clip_same_shot(reasons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dz_ok[t]: no CUT in transitions 2t..2t+2 (t<=62). side_ok[t] = same-shot t-2,t-1,t+1,t+2."""
    cut = (np.asarray(reasons, np.int64) & int(Reason.CUT)) != 0
    if cut.shape != (TRANSITIONS,):
        raise ValueError("clip reasons must cover 127 transitions")
    dz = np.zeros(TEMPORAL, bool)
    for t in range(TEMPORAL - 1):
        dz[t] = not cut[2 * t:2 * t + 3].any()
    side = np.zeros((TEMPORAL, 4), bool)
    for t in range(TEMPORAL):
        side[t, 0] = t >= 2 and dz[t - 2] and dz[t - 1]
        side[t, 1] = t >= 1 and dz[t - 1]
        side[t, 2] = t <= TEMPORAL - 2 and dz[t]
        side[t, 3] = t <= TEMPORAL - 3 and dz[t] and dz[t + 1]
    return side, dz


def build_target_table(output: Path, *, manifest_path: Path, half_targets: dict[str, dict[str, np.ndarray]],
                       final_validity: dict[str, dict[str, np.ndarray]], target: str = "boundary") -> dict:
    """half_targets: half -> {value, supported}; final_validity: half -> {states, reasons}."""
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest["entries"]
    rows = len(entries)
    y = np.full((rows, TEMPORAL), np.nan, np.float32)
    supported = np.zeros((rows, TEMPORAL), bool)
    side_ok = np.zeros((rows, TEMPORAL, 4), bool)
    dz_ok = np.zeros((rows, TEMPORAL), bool)
    for row, entry in enumerate(entries):
        half = entry["half_id"]
        value, ok = half_targets[half]["value"], half_targets[half]["supported"]
        reasons = final_validity[half]["reasons"]
        n = len(reasons)
        if (value.dtype != np.float32 or ok.dtype != bool or value.shape != (n,) or ok.shape != (n,)
                or not np.isfinite(value[ok]).all() or not np.isnan(value[~ok]).all()):
            raise RuntimeError(f"malformed half target: {half}")
        span = slice(entry["start_frame"], entry["end_frame_exclusive"] - 1)
        y[row], supported[row] = clip_targets(value[span], ok[span])
        side_ok[row], dz_ok[row] = clip_same_shot(reasons[span])
    if not supported.any():
        raise RuntimeError("target table has no supported targets")
    target_mean = float(np.mean(y[supported], dtype=np.float64))
    metadata = {"artifact_kind": TABLE_KIND, "schema_version": 1, "target": target, "kind": KIND_FOR_TARGET[target],
                "target_mean": target_mean, "clip_manifest_sha256": sha256(manifest_path), "side_order": [-2, -1, 1, 2],
                "mapping": "t=2..60 max supported {2t,2t+1}; t=61 {2t}; else unsupported",
                "same_shot": "dz_t excluded iff CUT in clip transitions 2t..2t+2"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, y=y, supported=supported, side_ok=side_ok, dz_ok=dz_ok, metadata=np.array(json.dumps(metadata)))
    return {"path": str(output), "sha256": sha256(output), "rows": rows, "target_mean": target_mean}


def load_aux_table(path: Path, *, kind: str, clip_manifest_sha256: str) -> AuxTable:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        if meta.get("artifact_kind") != TABLE_KIND or meta["kind"] != kind:
            raise ValueError(f"{path} is not a {kind} boundary-target table")
        if meta["clip_manifest_sha256"] != clip_manifest_sha256:
            raise ValueError(f"{path} was built from a different clip manifest (rebuild with bpjepa-evidence target-table)")
        return AuxTable(kind, *(torch.from_numpy(z[k]) for k in ("y", "supported", "side_ok", "dz_ok")), meta["target_mean"])
