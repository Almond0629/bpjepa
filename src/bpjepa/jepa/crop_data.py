"""Clip-consistent spatial crops with worker-independent, resumable randomness."""
from __future__ import annotations

import math
import random

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision

from bpjepa.jepa.full_data import FullSoccerNetClipDataset, PaddedDistributedEpochSampler


CROP_AUGMENTATION = {
    "policy": "vjepa_random_resized_crop_only_v1",
    "randomness": "local_rng_by_pretraining_seed_epoch_sample_id",
    "scale": [0.3, 1.0],
    "aspect_ratio": [0.75, 1.35],
    "interpolation": "PIL_bilinear",
    "shared_across_frames_and_jepa_branches": True,
    "horizontal_flip": False,
    "color_jitter": False,
    "gaussian_blur": False,
    "auto_augment": False,
    "random_erasing": False,
    "normalization": "imagenet",
}


def crop_box(width: int, height: int, *, seed: int, epoch: int, sample_id: int):
    """V-JEPA area/uniform and aspect/log-uniform sampling, with center fallback.

    The local RNG is independent of worker scheduling, prefetch, global model RNG,
    and rank. Padded duplicates get the same crop; each epoch gets a new draw.
    """
    if min(width, height) < 1 or min(epoch, sample_id) < 0:
        raise ValueError("invalid crop dimensions, epoch, or sample ID")
    rng = random.Random(f"vjepa-crop-v1:{seed}:{epoch}:{sample_id}")
    for _ in range(10):
        area = rng.uniform(0.3, 1.0) * width * height
        ratio = math.exp(rng.uniform(math.log(0.75), math.log(1.35)))
        w, h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if 0 < w <= width and 0 < h <= height:
            return rng.randint(0, height - h), rng.randint(0, width - w), h, w
    ratio = width / height
    if ratio < 0.75:
        w, h = width, round(width / 0.75)
    elif ratio > 1.35:
        w, h = round(height * 1.35), height
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


class CropEpochSampler(PaddedDistributedEpochSampler):
    """Send epoch with each sample key, including to persistent loader workers."""

    def __iter__(self):
        epoch = self.epoch
        return ((epoch, index) for index in super().__iter__())


class SpatialCropDataset(FullSoccerNetClipDataset):
    """Keep manifest, frame order, geometry and normalization; replace resize only."""

    def __getitem__(self, key) -> torch.Tensor:
        epoch, index = key
        entry = self.entries[index]
        directory = self.frame_root / entry["frame_dir"]
        tensors = []
        size = box = None
        for frame in range(entry["start_frame"] + 1, entry["end_frame_exclusive"] + 1):
            with Image.open(directory / f"{frame:08d}.jpg") as image:
                image = image.convert("RGB")
                if box is None:
                    size = image.size
                    # The paper recipe uses pretraining_seed=0.
                    box = crop_box(*size, seed=0, epoch=epoch, sample_id=entry["sample_id"])
                if image.size != size:
                    raise ValueError("frame dimensions changed within a clip")
                image = vision.resized_crop(
                    image, *box, [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                tensors.append(vision.normalize(
                    vision.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ))
        clip = torch.stack(tensors, dim=1)
        if clip.shape != (3, 128, self.image_size, self.image_size):
            raise RuntimeError(f"invalid production clip shape: {tuple(clip.shape)}")
        return clip
