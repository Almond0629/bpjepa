"""fp32 DINOv3 runtime for the raw per-token change C; no feature cache."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import platform

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .official_dinov3 import load_official_vitb16
from .teacher import TeacherConfig
from .frames import decode_verified_jpeg
from .warp import raw_transition
from .change import AlignmentResult


SETTINGS = {"teacher_batch_size": 8, "remainder_batch": "pad_repeat_last", "precision": "fp32", "autocast": False,
            "sdpa_backend": "MATH", "cublas_workspace_config": ":4096:8", "matmul_precision": "highest"}


def configure():
    """Apply the numerical settings of the evidence runs and return the environment actually used.

    Thread limits and CUBLAS_WORKSPACE_CONFIG are set by the bpjepa-evidence entry before torch/numpy load.
    Exact bit reproduction also depended on the recorded stack (docs/ARTIFACTS.md); other stacks only get recorded.
    """
    if os.environ.get('OMP_NUM_THREADS') != '1' or os.environ.get('MKL_NUM_THREADS') != '1':
        raise RuntimeError('OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 must be set before Python starts')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before Python starts')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('raw-change extraction needs exactly one visible GPU; set CUDA_VISIBLE_DEVICES')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    return {'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(), 'opencv_version': cv2.__version__,
            'python_version': platform.python_version(), 'numpy_version': np.__version__,
            'gpu_model': torch.cuda.get_device_name(0), 'settings': SETTINGS}


def verified_frame(path, record):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise RuntimeError(f'JPEG hash mismatch: {path}')
    image = decode_verified_jpeg(data)
    if image.size != (record['width'], record['height']):
        raise RuntimeError('verified JPEG dimensions changed')
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def preprocess(frames, teacher_config):
    if frames.dtype != torch.float32 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError('fp32 RGB batch required')
    if not torch.isfinite(frames).all() or bool((frames < 0).any()) or bool((frames > 1).any()):
        raise ValueError('finite RGB [0,1] required')
    resized = F.interpolate(frames, size=(448, 448), mode='bicubic',
                            antialias=True, align_corners=False)
    mean = resized.new_tensor(teacher_config.normalization_mean)[None, :, None, None]
    std = resized.new_tensor(teacher_config.normalization_std)[None, :, None, None]
    return (resized - mean) / std


class V2Teacher:
    def __init__(self, teacher_config, vendor_path, settings=SETTINGS):
        self.config = teacher_config
        self.environment = settings
        self.teacher = load_official_vitb16(teacher_config, Path(vendor_path)).to('cuda:0').float()
        self.teacher.eval().requires_grad_(False)
        if any(p.dtype != torch.float32 for p in self.teacher.parameters()):
            raise RuntimeError('teacher is not fp32')

    @torch.inference_mode()
    def extract(self, frames):
        n = len(frames)
        B = self.environment['teacher_batch_size']
        if not 0 < n <= B:
            raise ValueError('batch outside fixed B')
        # Remainders always use the same kernel shape; padding never becomes evidence.
        padded = torch.cat((frames, frames[-1:].expand(B - n, -1, -1, -1)), dim=0) if n < B else frames
        inputs = preprocess(padded.cpu(), self.config).to('cuda:0')
        with torch.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            dense = self.teacher.adapter(self.teacher.model, inputs, self.config.feature_layer)
            if dense.shape != (B, 784, 768) or dense.dtype != torch.float32:
                raise RuntimeError('unexpected teacher output geometry/precision')
            f28 = F.normalize(dense.float(), dim=-1, eps=1e-12).reshape(B, 28, 28, 768)
        f28 = f28.cpu()
        f14 = F.interpolate(f28.permute(0, 3, 1, 2), size=(14, 14), mode='bilinear',
                            align_corners=False).permute(0, 2, 3, 1).contiguous()
        finite = torch.isfinite(f28).all(dim=(1, 2, 3)) & torch.isfinite(f14).all(dim=(1, 2, 3))
        return f28[:n].contiguous().numpy(), f14[:n].numpy(), finite[:n].numpy()


def measured_raw(previous, current, arrays, j):
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        return np.full((14, 14), np.nan, np.float32), False
    status = int(arrays['alignment_status'][j])
    if status == 0:
        return np.full((14, 14), np.nan, np.float32), True
    alignment = AlignmentResult('ecc' if status == 1 else 'phase_correlation',
                                tuple(tuple(float(v) for v in row) for row in arrays['affine'][j]), None)
    result = raw_transition(torch.from_numpy(previous), torch.from_numpy(current),
                            alignment, motion_grid=(224, 398))
    if not np.array_equal(result.in_bounds, arrays['mask'][j]):
        raise RuntimeError('frozen V1 mask disagrees with frozen warp semantics')
    values = np.asarray(result.raw_c, dtype=np.float32)
    if not np.isfinite(values[arrays['mask'][j]]).all():
        raise RuntimeError('nonfinite raw C on finite-feature accepted transform')
    return values, True
