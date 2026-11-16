"""The recipe closer to V-JEPA (Tab. 1): crops, flip, scheduled LR, linear EMA, cosine weight decay, no drop path."""
import math
import random

import torch

from bpjepa.jepa.crop_data import CROP_AUGMENTATION, SpatialCropDataset

HORIZON = 102375
AUGMENTATION = {**CROP_AUGMENTATION, 'policy': 'vjepa_style_crop_flip_v1',
                'horizontal_flip': True, 'horizontal_flip_probability': 0.5,
                'motion_shift': False}
LR = dict(name='linear_warmup_cosine', start_lr=3.2e-5, end_lr=1e-6,
          warmup_updates=10920, total_updates=81900, horizon_updates=HORIZON)
EMA = dict(name='linear', horizon_updates=HORIZON, indexing='zero_based_optimizer_update')
WD = dict(name='cosine', start=0.04, end=0.4, horizon_updates=HORIZON,
          exclude_bias_and_1d=True, indexing='one_based_optimizer_update')


def validate_config(config):
    expected = dict(augmentation=AUGMENTATION, scheduler=LR, ema_schedule=EMA,
                    weight_decay_schedule=WD, ema_start=0.998, ema_end=1.0,
                    weight_decay=0.04, learning_rate=1e-4, condition='compact_temporal_multimask_v1')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'recipe requires {key}={value!r}')
    if config['architecture'].get('drop_path_rate') != 0.0:
        raise ValueError('recipe requires zero drop-path')


class RecipeDataset(SpatialCropDataset):
    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        clip = super().__getitem__((epoch, index))
        sample_id = self.entries[index]['sample_id']
        if random.Random(f'vjepa-flip-v1:0:{epoch}:{sample_id}').random() < 0.5:
            clip = clip.flip(-1)
        return clip


def weight_decay(update):
    return 0.4 + (0.04 - 0.4) * (1 + math.cos(math.pi * update / HORIZON)) / 2


def momentum(step, total_steps, start, end):
    # Trainer calls after incrementing global_step. Match reference's first m=0.998.
    return start + (end - start) * (step - 1) / HORIZON


class RecipeScheduler(torch.optim.lr_scheduler.LambdaLR):
    def step(self, epoch=None):
        super().step(epoch)
        for group in self.optimizer.param_groups:
            group['weight_decay'] = 0.0 if group['WD_exclude'] else weight_decay(self.last_epoch + 1)


def build_scheduler(optimizer, config, base_builder):
    # Split before scheduler construction, retaining optimizer identity for checkpoints.
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise ValueError('recipe scheduler must be constructed on a fresh optimizer')
    original = optimizer.param_groups[0]
    parameters = original['params']
    original['params'] = [p for p in parameters if p.ndim > 1]
    original['WD_exclude'] = False
    optimizer.add_param_group({'params': [p for p in parameters if p.ndim <= 1],
                               'weight_decay': 0.0, 'WD_exclude': True})
    # All biases in this fixed architecture are 1D; covered by the exclusion above.
    reference = base_builder(optimizer, config)
    factors = reference.lr_lambdas
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']
    return RecipeScheduler(optimizer, factors)
