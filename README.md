# BP-JEPA: label-free boundary prediction for action spotting

Code for **"Teaching V-JEPA When Events Happen: Label-Free Boundary Prediction for Action Spotting"**, LVR @ WACV 2027.

## What BP-JEPA does

BP-JEPA adds one auxiliary objective to masked latent prediction. A label-free per-second *boundary evidence* target `y_t` is computed offline: the change between consecutive frames in camera-compensated DINOv3 features, turned into a same-shot before/after regime statistic, calibrated against nuisance-matched references, and pooled. During pretraining a light head predicts `y_t` from the difference between pooled context features before and after each second, and the loss becomes `L_JEPA + lambda_b(i) L_B` with `lambda_b = 1.6` ramped over 10 epochs. The head is then discarded. No event annotations enter the target or the pretraining losses.

**Relation to V-JEPA.** This is the V-JEPA *framework* adapted to a spatial encoder, not V-JEPA. The encoder attends only within each one-second tubelet; a compact temporal predictor (2 blocks, width 96) combines seconds. The official spatiotemporal V-JEPA encoder is not used, and the recipe differs (supplementary S1).

**Relation to COMEDIAN.** The encoder is COMEDIAN's spatial ViViT-T, so pretrained weights load directly into COMEDIAN's spotting model. COMEDIAN Stage 1 is the self-supervised reference. Stage 2 distillation is not part of the encoder comparisons. The Stages 1-3 reproduction is a separate pipeline check that needs supervised Baidu features.

| Paper name | Config | Result-file key (historical name) |
|---|---|---|
| R, latent-prediction baseline | `configs/pretrain/r.json` | `R0` |
| BP-JEPA, the method | `configs/pretrain/bp_jepa.json` | `EB0` |
| TC, temporal-contrast ablation | `configs/pretrain/tc.json` | `U` |
| BP-JEPA+TC | `configs/pretrain/bp_jepa_tc.json` | `EB` |
| E, evidence-weighted TC | audited and never trained | `E` |

## Installation

Python 3.10. The paper's runs used several stacks; [`environment/README.md`](environment/README.md) lists them and how to rebuild them.

```bash
pip install -r environment/requirements-pretrain-evidence.txt   # or the downstream lock, see environment/README.md
pip install --no-deps .
```

Commands below use `configs/` from this checkout; only the Python packages and the vendored Eztorch configs are in the wheel.

## Data and artifact prerequisites

Nothing large is in this repository. You need:

- SoccerNet-v2 action-spotting videos and `Labels-v2.json`, under the SoccerNet terms; the validation labels are used for evaluation only.
- 2 fps, 224p frames and `train.json`/`valid.json`, plus cached labels (commands in [`REPRODUCE.md`](REPRODUCE.md)).
- DINOv3 ViT-B/16 LVD-1689M weights and code, for the evidence only.
- Baidu features, only for the COMEDIAN Stages 1-3 pipeline check.

[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) gives purpose, size, provenance, checksum and how to get each one. Pretrained weights and evidence tables are not published.

## Building the tables

Tabs. 2-4 are built from the outputs of the analysis commands (`RESULTS` holds `results_100f`, `results_frozen` and `results_probes`, see [`REPRODUCE.md`](REPRODUCE.md)):

```bash
bpjepa-paper check  --results RESULTS
bpjepa-paper tables --results RESULTS --output-dir out
```

The stored results of the paper's runs are not included in this repository.

## Reproducing the paper

Every stage starts directly from its configuration. [`REPRODUCE.md`](REPRODUCE.md) has the commands for every experiment; [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) maps each table, figure and claim to code, configuration, inputs and result files.

| Stage | Command |
|---|---|
| Clip manifest | `bpjepa-clip-manifest` |
| Evidence | `bpjepa-evidence {measure,raw-change,state-keys,references,targets,tables}` |
| Pretraining, with `--resume` | `bpjepa-pretrain` |
| Export | `bpjepa-export {jepa,comedian}` |
| Full fine-tuning, with `--resume` ([limits](REPRODUCE.md#resuming-100f)) | `bpjepa-finetune` |
| Frozen encoder | `bpjepa-frozen-cache`, `bpjepa-frozen-head` |
| Spotting statistics | `bpjepa-analyze-spotting` |
| Representation evaluations | `bpjepa-probes-prepare`, `bpjepa-probes-extract`, `bpjepa-probes-analyze` |
| Evidence audit functions, self-test | `bpjepa-audit test` |
| Evidence figure | `bpjepa-fig2` |
| COMEDIAN | `third_party/eztorch_run/pretrain.py`, `supervised.py` with `configs/comedian/` |

GPUs and distribution are ordinary launch settings: choose GPUs with `CUDA_VISIBLE_DEVICES`, processes with `torchrun --nproc_per_node`, and worker counts with `--num-workers`. The paper used two GPUs; a different world size changes per-rank data order and needs `--world-size`, `--micro-batch-per-gpu` and `--effective-global-batch` to stay consistent. Each run records its resolved configuration, seed, software versions, hardware, input hashes and outputs in `run_record*.json`.

## Compute and storage

**Measured on the paper's runs** (two GPUs unless noted; other users' jobs were sometimes present on these machines):

| Stage | Hardware | Measurement |
|---|---|---|
| Pretraining, R | 2x RTX 5090, frames on a RAM disk | median 0.755 s/update over 81,900 updates |
| Pretraining, recipe study | 2x RTX A6000 | median 1.43 s/update, peak 16.7 GiB per GPU (128 frames) |
| COMEDIAN Stage 1 rerun | 2x RTX 5090 | median 0.204 s/update over 322,300 updates |
| 100F, COMEDIAN arm | 2x RTX 5090 | fit 20,553 s (5.7 h) |
| Frozen cache, 800 halves | 1x RTX 5090 | 21.5 min for the COMEDIAN cache |
| Frozen head, one seed | 2x RTX 5090 | about 74-77 min |
| Frozen head, one seed | 2x RTX A6000 | about 3.3 h, extrapolated from 10-epoch validation cycles |
| Pretraining checkpoint | | 97 MB |

**Estimates.** Multiplying the medians gives about 17 h per pretraining arm and 18 h for COMEDIAN Stage 1 on RTX 5090 with fast frame I/O. Network storage was about 4-6 times slower for these small JPEGs. Storage figures are in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

## Evaluation conventions

- Primary metric: validation tight Avg-mAP (tolerances 1-5 s), official SoccerNet evaluation. Also mAP at 1 s and 5 s, and loose Avg-mAP (5-60 s). The paper reports validation only; these configurations have no test path.
- Fixed final checkpoint: pretraining epoch 100 (81,900 updates); 100F and frozen heads at final epoch index 99 (step 93,700).
- 100F uses seed 42. Frozen heads use seeds 42, 203 and 666, and head initialization and data order are identical across arms within a seed.
- Inference: 32-s windows with 16-s overlap, scores every 0.5 s, max merge, predictions within 6 s of a window edge ignored unless near the half boundary.
- **NMS:** `NMS_args.window: 10` means 10 **steps** on the 0.5-s grid: a score must be the maximum within 5 steps (+-2.5 s) on either side. It is not a 10-second window. COMEDIAN's soft NMS in the pipeline check is different (window 20 steps).
- Spotting statistics: paired whole-match bootstrap, B = 10,000 PCG64(20260927) draws over the 100 validation matches, shared across arms and seeds; the frozen mean is recomputed within each draw and so is the interaction. Representation evaluations use their own 2,000 draws over the 60 evaluation matches.
- Frozen evaluation uses the pretrained **online spatial encoder**, never weights from fine-tuning, with a fresh temporal head on cached features from both frame-pair phases.

## Known limitations

Please read [`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md). In brief:

- One pretraining run per arm; intervals resample validation matches only, so they do not cover variation across pretraining or downstream runs. The BP-JEPA contrasts were analysed after seeing its point estimates.
- Validation only.
- The packaging preserves the code paths; it does **not** establish exact reproduction on other hardware or software.
- The evidence audit cannot be re-run from this repository: its functions and their self-test are included, but the stages that produced the paper's audit numbers and their inputs are not.

## License and citation

This repository's own code and documentation are released under the [MIT License](LICENSE). The bundled Eztorch / COMEDIAN code (`src/eztorch/`, `third_party/eztorch_run/`) and the run configurations derived from it (`configs/comedian/`, `configs/downstream/`) stay under CeCILL 2.1, which also governs the downstream pipeline when it is distributed together with that code. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details and [`CITATION.cff`](CITATION.cff) for citation metadata.
