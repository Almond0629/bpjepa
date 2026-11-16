# Paper-to-code inventory

Result-file keys are the historical names: `R0`=R, `EB0`=BP-JEPA, `U`=TC, `EB`=BP-JEPA+TC. The "Result artifact" column names the files the analysis commands write; the stored results of the paper's runs are not included in this repository.

## Main paper

| Paper item | Values | Code | Configuration | Inputs | Result artifact |
|---|---|---|---|---|---|
| Tab. 1 recipe study, A6000 rows | 56.81, 54.55, 55.78, 56.69, 46.75, 57.45 | `pretrain/train.py`, `jepa/crop_data.py`, `jepa/short_data.py`, `downstream/finetune.py`; COMEDIAN via `third_party/eztorch_run/pretrain.py` | `configs/pretrain/recipe_{constant,scheduled_lr,crop,64f}.json`, `configs/downstream/finetune_100f.yaml`, `configs/comedian/stage1_pretrain_a6000.yaml` | frames, labels | not included (published values); COMEDIAN value from a 4-decimal NMS-10 evaluation log |
| Tab. 1, 5090 rows | 55.94, 53.00 | + `pretrain/vjepa_recipe.py` | `configs/pretrain/{r,recipe_vjepa}.json` | same | not included (published values) |
| Tab. 2 main results, 100F | R, BP-JEPA, COMEDIAN; four tolerances; contrasts | `downstream/finetune.py`, `downstream/initialization.py`, `downstream/comedian_initialization.py`, `analysis/analyze_spotting.py` | `finetune_100f.yaml`, seed 42 | checkpoints, exports, frames | `results_100f/*` (contrasts, draws, `tolerances.json`) |
| Tab. 2 frozen | 47.41, 49.12, 43.50 | `downstream/frozen_cache.py`, `frozen_head.py`, `frozen.py`, `analysis/analyze_spotting.py` | same yaml; seeds 42, 203, 666 | online encoder, cache | `results_frozen/*` |
| Tab. 3, 2x2 ablation | contrasts, interaction | `analysis/contrasts.py` | | Tab. 2 inputs, plus TC and BP-JEPA+TC | `results_100f/`, `results_frozen/` |
| Tab. 4, probes | boundary R^2, zoom coupling, effective rank | `probes/{prepare,extract,analyze,lib}.py` | `configs/evidence/roles.json`, `configs/probes/probe_masks.npz` | checkpoints, evidence, frames | `results_probes/results.json` |
| Per-class claims (14/17, 16/17) | | `analyze_spotting.py` (`per_class`) | | | `results_*/results.json` |
| Sec. 3.3, evidence | | `evidence/{measurements,change,warp,motion_proxy,validity_policy,dinov3_runtime,raw_change,regime,pipeline}.py` | `configs/evidence/{calibration,teacher,roles}.json` | frames, DINOv3 | `configs/evidence/identities.json` |
| Sec. 3.4, BP objective | Eq. 1, ramp, `lambda_b` | `jepa/boundary.py`, `evidence/target_table.py` | `bp_jepa.json` | target table | |
| Sec. 3.5, TC | Eq. 2 | `jepa/kernel.py`, `support.py`, `conditions.py`, `temporal_vjepa.py` | `tc.json` | clip validity | |
| Secs. 5.5, S3, audit | 0.532, 0.668, 0.64, 0.638, 0.636, lifts | `analysis/audit/*` (functions and self-test only) | `calibration.json` | see below | not included |
| Figs. 1, curves, ablation, S1 | | not included | | | |
| Fig. 2, evidence example | | `analysis/fig2_evidence.py` (`bpjepa-fig2`), defaults are the published sequence | `configs/evidence/roles.json` | frames, evidence directory | `fig2_evidence.{png,json}` in the output directory |

## Supplementary

| Item | Code or configuration |
|---|---|
| S1.1 encoder and predictor | `jepa/model.py`, `jepa/temporal.py`, `samplers/vjepa_multimask.py`, `pretrain/*` configs |
| S1.2 support and calibration | `evidence/regime.py`, `validity_policy.py`, `change.py`, `configs/evidence/calibration.json` |
| S1.3 auxiliaries | `jepa/boundary.py`, `jepa/kernel.py` |
| S1.3 coefficient screen for `lambda_TC` | **Not packaged.** Its result fixes `lambda_tc = 0.01` in the configs; the pilot harness depended on the historical orchestration. Values are in the supplementary. |
| S1.4 shared protocol | one trainer, one seed; identity with the paper's update-0 snapshot was checked during packaging |
| S1.4 identity tests | Historical CPU tests are not copied. |
| S1.5 downstream | `finetune_100f.yaml`, `frozen.py`, vendored `eztorch/models/soccernet_spotting.py`, `eztorch/evaluation/nms.py` |
| S1.6 COMEDIAN reference and pipeline check | `configs/comedian/*`, `third_party/eztorch_run/*`, `src/eztorch/` |
| S1.7 probes | `probes/*` |
| S2 choosing `lambda_b` | **Not packaged.** Calibration used intermediate-checkpoint gradients and historical orchestration. `lambda_b = 1.6` is fixed in the configs. Its head-fit routine is kept in `probes/lib.py`. |
| S3 audit | `analysis/audit/*` (functions and self-test only) |

## Stage requirements: COMEDIAN Stage 1 vs the full pipeline

| | Stage 1 reference | Full three-stage pipeline check |
|---|---|---|
| Purpose | encoder comparison, Tabs. 1-2 | check that the pipeline reproduces the published level (68.27 vs 68.0) |
| Labels during pretraining | none | Stage 2 distils a supervised Baidu teacher; Stage 3 uses event labels |
| Extra inputs | none | PCA-512 Baidu features |
| Result | `results_*` | not included |
| NMS | hard, window 10 steps | soft, window 20 steps at 0.5 s |

## Missing or unavailable

| Item | Status |
|---|---|
| Datasets, frames, videos | not redistributed; see `docs/ARTIFACTS.md` |
| Pretrained checkpoints, evidence tables, predictions | not included |
| Audit stages and inputs | Not included, so only the audit self-test runs from this repository (`REPRODUCE.md`, section 10). Two paper values come from an audit input record and not from the audit stages' outputs: the concordance 0.532 for calibrated change magnitude and the "about 43k" audit events (42,919). |
| `lambda_TC` screen and `lambda_b` calibration | see above |
| Run logs of the paper's runs | not included |
