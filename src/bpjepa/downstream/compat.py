"""Runtime compatibility for the downstream stack: Lightning 2.0.9 code on modern PyTorch.

These are the settings the paper's RTX 5090 runs applied; they do not affect training arithmetic.
Import this module before lightning/eztorch.
"""
import os

import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
try:
    # Optimizer.step lazily imports Dynamo in modern Torch; import before DataLoader/pin-memory threads start.
    import torch._dynamo  # noqa: F401
except ImportError:
    pass

# PyTorch 2.11 removed Sampler's previously ignored data_source argument.
if torch.utils.data.Sampler.__init__ is object.__init__:
    torch.utils.data.Sampler.__init__ = lambda self, data_source=None: None
