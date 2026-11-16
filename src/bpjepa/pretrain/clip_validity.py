"""Per-clip transition validity for the TC path: same-time keys, pair validity and reason flags.

clip_validity.npz is written by `bpjepa-evidence clip-validity` from the final per-half validity (floor 4).
Tables are derived on demand with the TransitionGrid rules used for the paper's runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bpjepa.common import sha256
from bpjepa.evidence.transition_grid import TransitionGrid
from bpjepa.jepa.temporal_vjepa import ClipValidity

TEMPORAL = 64
TRANSITIONS = 2 * TEMPORAL - 1
_T = np.arange(TEMPORAL)
_LO = 2 * np.minimum.outer(_T, _T)  # J(t,u) = [2 min(t,u), 2 max(t,u)], inclusive
_HI = 2 * np.maximum.outer(_T, _T)


def pair_reason_table(reasons: np.ndarray) -> np.ndarray:
    """[T,T] OR of transition reason bits over J(t,u); zero diagonal."""
    reasons = np.asarray(reasons, dtype=np.int64)
    if reasons.shape != (TRANSITIONS,):
        raise ValueError("clip reasons must cover 127 transitions")
    out = np.zeros((TEMPORAL, TEMPORAL), np.int64)
    for bit in range(7):
        count = np.r_[0, np.cumsum((reasons >> bit) & 1)]
        out |= ((count[_HI + 1] - count[_LO]) > 0).astype(np.int64) << bit
    np.fill_diagonal(out, 0)
    return out


class ClipValiditySource:
    """Manifest-bound runtime validity, looked up by clip row."""
    test_only = False

    def __init__(self, path: Path, manifest_sha256: str, offsets):
        self.offsets = tuple(offsets)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            if meta.get("artifact_kind") != "bpjepa_clip_validity":
                raise ValueError(f"{path} is not a clip-validity table")
            if meta["clip_manifest_sha256"] != manifest_sha256:
                raise ValueError(f"{path} was built from a different clip manifest (rebuild with bpjepa-evidence clip-validity)")
            self.states, self.reasons = z["states"], z["reasons"]
            self.half_start, self.half_stop = z["half_start"], z["half_stop"]
            self.row_half, self.row_start, self.row_end = z["row_half"], z["row_start"], z["row_end"]
        self.manifest_sha256 = manifest_sha256
        self.identity = {"table_sha256": sha256(path), "manifest_sha256": manifest_sha256, "offsets": list(self.offsets)}

    def __call__(self, row_id: int, manifest_sha256: str) -> ClipValidity:
        if manifest_sha256 != self.manifest_sha256 or not 0 <= row_id < len(self.row_half):
            raise RuntimeError("clip row or manifest identity mismatch")
        half = int(self.row_half[row_id])
        span = slice(int(self.half_start[half]), int(self.half_stop[half]))
        start, end = int(self.row_start[row_id]), int(self.row_end[row_id])
        grid = TransitionGrid(self.states[span], self.reasons[span]).clip(start, end - start)
        keys, key_reasons = grid.key_validity()
        pairs, _ = grid.pair_table(self.offsets)
        return ClipValidity(row_id, manifest_sha256, torch.from_numpy(pairs), torch.from_numpy(keys),
                            torch.from_numpy(pair_reason_table(grid.reasons)),
                            torch.from_numpy(key_reasons.astype(np.int64)))


def build_clip_validity(output: Path, *, manifest_path: Path, final_validity: dict[str, dict[str, np.ndarray]]) -> dict:
    """final_validity: half_id -> {states, reasons}. Concatenated per-half, indexed per manifest row."""
    manifest = json.loads(Path(manifest_path).read_text())
    halves = list(manifest["half_ids"])
    missing = [h for h in halves if h not in final_validity]
    if missing:
        raise ValueError(f"final validity is missing {len(missing)} halves of the clip manifest, e.g. {missing[0]}")
    states, reasons, starts, stops, cursor = [], [], [], [], 0
    index = {}
    for i, half in enumerate(halves):
        s, r = np.asarray(final_validity[half]["states"], np.int8), np.asarray(final_validity[half]["reasons"], np.uint8)
        states.append(s), reasons.append(r), starts.append(cursor)
        cursor += len(s)
        stops.append(cursor)
        index[half] = i
    entries = manifest["entries"]
    row_half = np.array([index[e["half_id"]] for e in entries], np.int32)
    row_start = np.array([e["start_frame"] for e in entries], np.int32)
    row_end = np.array([e["end_frame_exclusive"] for e in entries], np.int32)
    for i, e in enumerate(entries):
        if e["sample_id"] != i or row_end[i] - 1 > stops[row_half[i]] - starts[row_half[i]]:
            raise ValueError(f"manifest row {i} falls outside its half")
    meta = {"artifact_kind": "bpjepa_clip_validity", "schema_version": 1, "clip_manifest_sha256": sha256(manifest_path),
            "half_ids": halves}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, states=np.concatenate(states), reasons=np.concatenate(reasons),
                        half_start=np.array(starts, np.int64), half_stop=np.array(stops, np.int64),
                        row_half=row_half, row_start=row_start, row_end=row_end, metadata=np.array(json.dumps(meta)))
    return {"path": str(output), "sha256": sha256(output), "rows": len(entries), "halves": len(halves)}
