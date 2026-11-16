# Third-party code and data

This repository's own code and documentation are released under the MIT License (`LICENSE`). The components below keep their own terms.

| Component | Where | License and status |
|---|---|---|
| Eztorch / COMEDIAN code (Denize et al.), upstream `juliendenize/eztorch` commit d8354719d9a4c79419294cb3a2298fca28bbe8c3 with this project's local changes | `src/eztorch/`, `third_party/eztorch_run/` | CeCILL 2.1, text in `third_party/EZTORCH_LICENSE`. Upstream's license file, README license section and package metadata all say CeCILL 2.1; only its README badge says CeCILL-C. Vendored with attribution; changes are listed below. `configs/comedian/` and `configs/downstream/` are run configurations in Eztorch's format and are treated the same way. |
| V-JEPA framework ideas; `bpjepa/samplers/vjepa_multimask.py` follows `facebookresearch/jepa` `src/masks/multiblock3d.py` | `src/bpjepa/samplers/` | Upstream is CC BY-NC 4.0. The sampler was written for this project and contains no upstream code (compared on 2026-11-16 against upstream `main`): it reproduces the published multi-block masking procedure and upstream's order of random draws, with different structure, interface and random-number handling. Released under MIT with this attribution. |
| DINOv3 code and weights (`facebookresearch/dinov3` commit 6876159a11b4df116f30f667f8c9888617df0751) | not included | DINOv3 License; obtain separately. |
| SoccerNet data, labels, videos, Baidu features | not included | SoccerNet terms; obtain separately. Match names appear in `configs/evidence/roles.json` and `configs/evidence/identities.json`. |
| Python dependencies | `environment/` | Their own licenses; not vendored. |

## CeCILL 2.1 and the MIT-licensed code

CeCILL 2.1 is a copyleft license compatible with the GNU GPL. The code in `src/bpjepa/downstream/` imports Eztorch, so anyone who redistributes that pipeline together with the Eztorch code must do so under CeCILL 2.1 terms. The MIT license on this project's own files allows that. The pretraining, evidence, probe and analysis code does not import Eztorch.

## Changes to the vendored Eztorch code

Compared with upstream commit d8354719d9a4c79419294cb3a2298fca28bbe8c3 on 2026-11-16. Apart from line endings, `src/eztorch/` equals upstream `eztorch/` except for three files:

- `datasets/soccernet_utils/features.py`: memory-bounded PCA and a sampled PCA that writes directly to a zip file.
- `models/soccernet_spotting.py`: one-line fix when loading a checkpoint (`state_dict = state_dict["state_dict"]`).
- `configs/run/pretrain/sce/vit/vit_tiny_spatio_temporal_soccernet.yaml`: checkpointing, worker count, 100 epochs, 2 devices.

`third_party/eztorch_run/` holds upstream `run/` without eight scripts this project does not use. Three scripts changed: `pretrain.py` and `supervised.py` import `bpjepa.downstream.compat` first, and `datasets/pca_soccernet.py` calls the sampled PCA. Upstream `docs/` and `tests/` are not included.
