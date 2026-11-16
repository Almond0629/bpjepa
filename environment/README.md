# Software stacks

The paper's runs used three stacks. Exact bit reproduction depends on them; other versions run but only get recorded in `run_record*.json`.

| Stage | Stack | torch | Other pinned packages | Hardware of the paper's runs |
|---|---|---|---|---|
| Pretraining of the four arms, DINOv3 raw change | P | 2.11.0+cu128 | numpy 2.2.6, opencv-python-headless 5.0.0.93, pillow 12.2.0, torchvision 0.26.0 | RTX 5090 |
| 100F, frozen cache/heads, COMEDIAN Stage 1 rerun on RTX 5090 | P+D | 2.11.0+cu128 from P | numpy 1.26.4, Lightning 2.0.9.post0, torchaug 0.3.3, hydra, etc. from D | 2x RTX 5090 |
| Recipe study (A6000 rows of Tab. 1), frozen head seeds 203/666, COMEDIAN full pipeline, analyses and probes on CPU | D | 2.0.1 (cu118) | `requirements-downstream.txt` | 2x RTX A6000; analyses on CPU |

Historically P+D was a process-level combination: torch and torchvision from P, then NumPy bound first from D and the remaining packages appended from D (NumPy 2 breaks Lightning 2.0.9, which uses `np.Inf`, so the `downstream` extra requires `numpy<2`). In one environment that means: install `requirements-downstream.txt`, then torch and torchvision 2.11/0.26. `bpjepa.downstream.compat` applies the remaining shims. This install path, and the `downstream` extra on its own, were solved from scratch and run.

Probe masks were drawn under torch 2.0.1, whose CPU generator gives different integers from torch 2.11 for the same seed. They are stored in `configs/probes/probe_masks.npz`, so the probes do not depend on the torch version.

OpenCV: the evidence measurements of the paper used OpenCV 5.0.0, so run them in stack P. Stack D pins opencv-python 4.11.0.86, because the version recorded with D (4.13.0.92) requires NumPy 2; no downstream code imports cv2. `setuptools<70` is required because Lightning 2.0.9 imports `pkg_resources`.

```bash
conda create -n bpjepa-p python=3.10 && conda activate bpjepa-p
pip install -r environment/requirements-pretrain-evidence.txt && pip install --no-deps .

conda create -n bpjepa-d python=3.10 && conda activate bpjepa-d
pip install -r environment/requirements-downstream.txt
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128   # or 2.0.1/0.15.2 for the A6000 rows
pip install --no-deps .
```

## Exact tested versions

For the versions that were actually tested, install from these locks instead of solving again. Only python 3.10 on Linux with the RTX 5090 stack was tested.

| Stack | Command | File |
|---|---|---|
| P | `pip install -r environment/requirements-pretrain-evidence.txt`, then `pip install --no-deps .` | Already exact |
| D, torch 2.11 | `pip install -r environment/constraints-downstream.txt --extra-index-url https://download.pytorch.org/whl/cu128`, then `pip install --no-deps .` | `constraints-downstream.txt` |
| Extras alone | `pip install -c environment/constraints-extra.txt ".[downstream,evidence]"` | `constraints-extra.txt` |

Use `-r`, not `-c`, for D: `requirements-downstream.txt` also serves torch 2.0.1, so it pins `triton` 2.0.0, while the torch 2.11 stack needs 3.6.0.

`pip check` is not enough for this stack, because some packages install files at the same paths and the last one installed wins, while `pip check` still passes:

- **`lightning-fabric` and `pytorch-lightning`:** the `pytorch-lightning` 2.0.9.post0 wheel bundles its own `lightning_fabric`. The recorded `lightning-fabric` 2.3.3 overwrote it and broke `import pytorch_lightning`, so that pin was removed. The release's training code uses `lightning.fabric` and does not load either top-level package.
- **`nvidia-*-cu11` and `nvidia-*-cu12`:** `requirements-downstream.txt` pins the CUDA 11 libraries for torch 2.0.1. Installing them with the CUDA 12 libraries in one step made `import torch` fail (`undefined symbol: ncclMemFree`). The path above that installs `requirements-downstream.txt` and then torch 2.11 works only because the CUDA 12 libraries are installed last, so `constraints-downstream.txt` omits CUDA 11.

Other versions, torch 2.0.1 and other hardware were not tested with these files, and no wider compatibility is claimed.
