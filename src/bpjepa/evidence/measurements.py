"""Fused byte-hash/decode and label-free per-transition measurements for one half (cut, wipe, alignment, mask, motion)."""
from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .change import MotionConfig, estimate_alignment, measure_cut_wipe
from .motion_proxy import phase_measurement
from .warp import mask_only
from .frames import decode_verified_jpeg, ordered_half_frames
from .validity_policy import derive_validity


ECC_CODES = {"unavailable": 0, "accepted": 1, "low_confidence": 2,
             "degenerate": 3, "error": 4}
ALIGNMENT_CODES = {"failure": 0, "ecc": 1, "phase_correlation": 2}


def measure_half(frame_dir: Path, frame_count: int, half_id: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Return per-transition measurements and floor-4 states, without writing files."""
    paths = ordered_half_frames(frame_dir, frame_count)
    transitions = frame_count - 1
    arrays: dict[str, np.ndarray] = {
        "frame_decode": np.zeros(frame_count, dtype=bool),
        "frame_width": np.zeros(frame_count, dtype=np.int64),
        "frame_height": np.zeros(frame_count, dtype=np.int64),
        "hist_corr": np.full(transitions, np.nan, dtype=np.float64),
        "changed_fraction": np.full(transitions, np.nan, dtype=np.float64),
        "cut": np.zeros(transitions, dtype=bool),
        "wipe": np.zeros(transitions, dtype=bool),
        "ecc_status": np.zeros(transitions, dtype=np.int64),
        "ecc_confidence": np.full(transitions, np.nan, dtype=np.float64),
        "phase_response": np.full(transitions, np.nan, dtype=np.float64),
        "alignment_status": np.zeros(transitions, dtype=np.int64),
        "affine": np.full((transitions, 2, 3), np.nan, dtype=np.float64),
        "affine_nonfinite": np.zeros(transitions, dtype=bool),
        "affine_nonpositive_det": np.zeros(transitions, dtype=bool),
        "affine_condition_exceeded": np.zeros(transitions, dtype=bool),
        "affine_condition_failed": np.zeros(transitions, dtype=bool),
        "mask": np.zeros((transitions, 14, 14), dtype=bool),
        "n_valid": np.full(transitions, -1, dtype=np.int64),
        "motion_displacement": np.full(transitions, np.nan, dtype=np.float64),
        "motion_response": np.full(transitions, np.nan, dtype=np.float64),
        "motion_bin": np.full(transitions, 3, dtype=np.int64),
        "processing_unknown": np.zeros(transitions, dtype=bool),
    }
    frame_records = []
    errors = []
    previous = None
    config = MotionConfig()
    for frame_id, path in enumerate(paths):
        data = path.read_bytes()
        record = {"frame_id": frame_id, "filename": path.name,
                  "sha256": hashlib.sha256(data).hexdigest()}
        try:
            with Image.open(BytesIO(data)) as header:
                width, height = header.size
        except Exception:
            width = height = 0
        arrays["frame_width"][frame_id] = width
        arrays["frame_height"][frame_id] = height
        if width and height and (width, height) != (398, 224):
            raise RuntimeError(f"JPEG header dimension mismatch: {path}: {width}x{height}")
        current = None
        if width and height:
            try:
                image = decode_verified_jpeg(data)
                current = np.ascontiguousarray(np.asarray(image, dtype=np.float32).mean(2) / 255)
            except Exception:
                pass
        arrays["frame_decode"][frame_id] = current is not None
        record.update(width=width, height=height, decoded=current is not None)
        frame_records.append(record)
        if frame_id and previous is not None and current is not None:
            j = frame_id - 1
            try:
                hist, changed, cut, wipe = measure_cut_wipe(previous, current, config)
                arrays["hist_corr"][j], arrays["changed_fraction"][j] = hist, changed
                arrays["cut"][j], arrays["wipe"][j] = cut, wipe
                phase = phase_measurement(previous, current)
                arrays["phase_response"][j] = phase.response
                arrays["motion_response"][j] = phase.response
                arrays["motion_displacement"][j] = phase.displacement
                arrays["motion_bin"][j] = phase.bin
                aligned = estimate_alignment(previous, current, config, phase_result=phase.result)
                arrays["ecc_status"][j] = ECC_CODES[aligned.ecc_status]
                if aligned.ecc_confidence is not None:
                    arrays["ecc_confidence"][j] = aligned.ecc_confidence
                for name, flag in zip(("affine_nonfinite", "affine_nonpositive_det",
                                       "affine_condition_exceeded", "affine_condition_failed"),
                                      aligned.affine_degeneracy, strict=True):
                    arrays[name][j] = flag
                if aligned.status != "failure":
                    try:
                        mask, n_valid = mask_only(aligned, motion_grid=(224, 398))
                    except ValueError:
                        arrays["affine_nonfinite"][j] = True
                    else:
                        arrays["alignment_status"][j] = ALIGNMENT_CODES[aligned.status]
                        arrays["affine"][j] = np.asarray(aligned.transform, dtype=np.float64)
                        arrays["mask"][j] = mask
                        arrays["n_valid"][j] = n_valid
            except (cv2.error, ValueError, RuntimeError) as error:
                arrays["processing_unknown"][j] = True
                errors.append({"transition": j, "error": type(error).__name__})
        previous = current
    states, reasons, labels = derive_validity(arrays, 4)
    arrays.update(states=states, reasons=reasons, summary_label=labels)
    metadata = {"schema_version": 1, "half_id": half_id, "frame_count": frame_count, "frames": frame_records,
                "opencv_version": cv2.__version__, "processing_errors": errors, "validity_floor": 4}
    return metadata, arrays
