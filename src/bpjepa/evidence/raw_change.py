"""Streaming raw per-token change C for one half; per-transition measurements and frame records are inputs."""
from pathlib import Path

import numpy as np
import torch

from bpjepa.evidence.dinov3_runtime import measured_raw, verified_frame
from bpjepa.evidence.validity_policy import derive_validity


def half_arrays(metadata, measurements, frame_dir, teacher, batch_size):
    n = len(metadata['frames'])
    raw = np.full((n - 1, 14, 14), np.nan, np.float32)
    finite = np.zeros(n, bool)
    previous = None
    for start in range(0, n, batch_size):
        records = metadata['frames'][start:start + batch_size]
        frames = torch.stack([verified_frame(Path(frame_dir) / r['filename'], r) for r in records])
        _, features, flags = teacher.extract(frames)
        finite[start:start + len(records)] = flags
        for k, current in enumerate(features):
            j = start + k - 1
            if previous is not None:
                raw[j], _ = measured_raw(previous, current, measurements, j)
            previous = current.copy()
    transition_finite = finite[:-1] & finite[1:]
    states, reasons, labels = derive_validity(measurements, 4, feature_nonfinite=~transition_finite)
    return {'raw_c': raw, 'frame_finite': finite, 'transition_finite': transition_finite,
            'states': states, 'reasons': reasons, 'summary_label': labels}
