# Reproduction guide

Set these once. `$FRAMES` holds `train.json`, `valid.json`, `train/`, `valid/`.

```bash
FRAMES=data/frames; LABELS=data/label_cache; VAL_LABELS=data/soccernet/valid; EV=data/evidence; RUNS=runs; CLIPS=$FRAMES/clips.json
```

Stack per stage: see `environment/README.md`. Launch training stages with `torchrun --standalone --nproc_per_node=2 -m <module>`, or use the `bpjepa-*` scripts for single-process stages. `CUDA_VISIBLE_DEVICES` selects GPUs.

## 1. Data

```bash
for split in train valid; do
  python third_party/eztorch_run/datasets/extract_soccernet.py --input-folder data/SoccerNet --output-folder $FRAMES --fps 2 --split $split
  python third_party/eztorch_run/datasets/precompute_soccernet_labels.py --radius-label 0.5 --data-path $FRAMES/$split.json --path-prefix $FRAMES/$split --fps 2 --cache-dir $LABELS/$split
done
bpjepa-clip-manifest --source $FRAMES/train.json --frame-root $FRAMES/train --output $CLIPS
```

The manifest command prints `entries_sha256`; the paper's value is in `configs/evidence/identities.json`. A different value means a different row order, so the target tables would not match the paper. The paper's frames came from the 224p videos.

## 2. Label-free evidence (Secs. 3.3, S1.2)

```bash
bpjepa-evidence measure    --clip-manifest $CLIPS --frame-root $FRAMES/train --evidence-dir $EV --workers 32   # CPU
CUDA_VISIBLE_DEVICES=0 bpjepa-evidence raw-change --clip-manifest $CLIPS --frame-root $FRAMES/train --evidence-dir $EV \
    --dinov3-repo third_party_external/dinov3 --dinov3-checkpoint <dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth> --teacher-config configs/evidence/teacher.json
bpjepa-evidence state-keys --clip-manifest $CLIPS --evidence-dir $EV
bpjepa-evidence references --clip-manifest $CLIPS --evidence-dir $EV --roles configs/evidence/roles.json --calibration-policy configs/evidence/calibration.json
bpjepa-evidence targets    --clip-manifest $CLIPS --evidence-dir $EV --roles configs/evidence/roles.json
bpjepa-evidence tables     --clip-manifest $CLIPS --evidence-dir $EV
```

`measure` and `raw-change` skip halves already written, so rerunning continues them. `tables` writes `clip_validity.npz` and `target_table_boundary.npz`. Compare the table's array hashes with `configs/evidence/identities.json`.

## 3. Pretraining the four arms (Tab. 2-4)

```bash
for arm in r bp_jepa tc bp_jepa_tc; do
  torchrun --standalone --nproc_per_node=2 -m bpjepa.pretrain.train --config configs/pretrain/$arm.json \
    --clip-manifest $CLIPS --frame-root $FRAMES/train --evidence-dir $EV --output-dir $RUNS/$arm
done
```

Resume: add `--resume $RUNS/$arm/checkpoints/<file>.pt` with the same `--output-dir`. All four arms share one seed-derived initialization, data order and masks; only the objectives differ. The final epoch writes `checkpoints/epoch-100-step-000081900.pt` and `exports/comedian_spatial_context.pt` (rerun that export with `bpjepa-export jepa` if needed).

## 4. Recipe study (Tab. 1)

```bash
for v in recipe_constant recipe_scheduled_lr recipe_crop recipe_vjepa; do
  torchrun --standalone --nproc_per_node=2 -m bpjepa.pretrain.train --config configs/pretrain/$v.json --clip-manifest $CLIPS --frame-root $FRAMES/train --output-dir $RUNS/$v
done
python - <<'PY'   # 64-frame manifest: the central 64 frames of each parent clip
import json, os
from pathlib import Path
from bpjepa.common import sha256
from bpjepa.jepa.short_data import derive_manifest
clips = Path(os.environ.get("CLIPS", "data/frames/clips.json"))
derived = derive_manifest(json.loads(clips.read_text()), clips, sha256(clips))
clips.with_name("clips64.json").write_text(json.dumps(derived))
PY
torchrun --standalone --nproc_per_node=2 -m bpjepa.pretrain.train --config configs/pretrain/recipe_64f.json --clip-manifest $FRAMES/clips64.json --frame-root $FRAMES/train --output-dir $RUNS/recipe_64f
```

The scratch row uses `bpjepa-finetune --init scratch` with no pretraining. The recipe configs fix their schedule to the paper's update budget, so they refuse a smoke run with a reduced batch or `--allow-subset-corpus` (`ValueError: resolved optimizer updates differ from the LR schedule training budget`).

## 5. COMEDIAN

**Stage 1 reference** (no supervised inputs):

```bash
torchrun --standalone --nproc_per_node=2 third_party/eztorch_run/pretrain.py --config-path $PWD/configs/comedian --config-name stage1_pretrain_5090 \
  paths.frames=$FRAMES paths.run=$RUNS/comedian_s1 paths.run_root=$RUNS
bpjepa-export comedian --checkpoint $RUNS/comedian_s1/pretrain_checkpoints/last.ckpt --output-dir $RUNS/comedian_s1/export
```

`stage1_pretrain_a6000` is the A6000 run used for Tab. 1.

**Full three-stage pipeline check** (additionally needs the Baidu features and event labels; it is not an encoder comparison):

```bash
torchrun --standalone --nproc_per_node=2 third_party/eztorch_run/pretrain.py --config-path $PWD/configs/comedian --config-name stage2_pretrain \
  paths.frames=$FRAMES paths.baidu_features=<pca_512_std_baidu_soccer_embeddings.zip> paths.stage1_checkpoint=<stage1 last.ckpt> paths.run=$RUNS/comedian_s2 paths.run_root=$RUNS
torchrun --standalone --nproc_per_node=2 third_party/eztorch_run/supervised.py --config-path $PWD/configs/comedian --config-name stage3_classifier \
  paths.frames=$FRAMES paths.label_cache=$LABELS paths.soccernet_labels_valid=$VAL_LABELS paths.stage2_checkpoint=<stage2 epoch=99.ckpt> paths.run=$RUNS/comedian_s3c paths.run_root=$RUNS
torchrun --standalone --nproc_per_node=2 third_party/eztorch_run/supervised.py --config-path $PWD/configs/comedian --config-name stage3_finetune \
  paths.frames=$FRAMES paths.label_cache=$LABELS paths.soccernet_labels_valid=$VAL_LABELS paths.stage3_classifier_checkpoint=<classifier last.ckpt> paths.run=$RUNS/comedian_s3f paths.run_root=$RUNS
```

## 6. Full fine-tuning, 100F (Tab. 1-3)

```bash
COMMON="--config configs/downstream/finetune_100f.yaml --frames-dir $FRAMES --label-cache-dir $LABELS --soccernet-labels-dir $VAL_LABELS"
for arm in r bp_jepa tc bp_jepa_tc; do
  torchrun --standalone --nproc_per_node=2 -m bpjepa.downstream.finetune $COMMON --init jepa \
    --checkpoint $RUNS/$arm/checkpoints/epoch-100-step-000081900.pt --export $RUNS/$arm/exports/comedian_spatial_context.pt --output-dir $RUNS/${arm}_100f
done
torchrun --standalone --nproc_per_node=2 -m bpjepa.downstream.finetune $COMMON --init comedian --checkpoint $RUNS/comedian_s1/pretrain_checkpoints/last.ckpt \
  --export $RUNS/comedian_s1/export/comedian_online_spatial.pt --output-dir $RUNS/comedian_100f
torchrun --standalone --nproc_per_node=2 -m bpjepa.downstream.finetune $COMMON --init scratch --output-dir $RUNS/scratch_100f
```

Tab. 1 COMEDIAN on the A6000 stack: `--init comedian-a6000 --checkpoint <stage1 a6000 last.ckpt>`. Recipe variants use the same JEPA command with their own checkpoint. `--init comedian` accepts only a checkpoint of a completed 100-epoch Stage 1 run (`COMEDIAN completed-epoch mismatch` otherwise).

### Resuming 100F

Add `--resume $RUNS/<run>/checkpoints/epoch=N.ckpt` to the original command, with the same config, overrides and `--output-dir`. `--resume` has a limit: some checkpoints are refused.

| Checkpoint | Result |
|---|---|
| Written at the end of an epoch, and the next epoch trains without validating | Accepted |
| `epoch=N` where the next epoch validates, meaning `(N + 2) % check_val_every_n_epoch == 0` | Refused: `cannot resume from epoch=N: the next epoch validates` |
| Saved epoch counters do not match that epoch, as in a file not written by this config's checkpoint callbacks | Refused: `expected a completed-epoch checkpoint written by the epoch callback` |

Lightning 2.0.9 would run validation in the first resumed epoch without training it, which silently loses that epoch and writes partial predictions. The refusal prevents this.

The paper's cadence is not affected. It validates every 10 epochs and checkpoints every 5, so it writes `epoch=4, 9, 14, ...`, and the refused values `8, 18, 28, ...` never occur. Refusals arise only with a changed cadence, such as `--override trainer.check_val_every_n_epoch=...` or `--override callbacks.epoch_checkpoint.every_n_epochs=...`.

If a checkpoint is refused:

1. Training has not started, and the checkpoint is unchanged.
2. Pick the newest earlier `epoch=N.ckpt` in that run that satisfies `(N + 2) % check_val_every_n_epoch != 0`, and resume from it. The epochs after it are trained again, and may overwrite that attempt's later checkpoints and predictions, so copy them first if you need them.
3. If no such checkpoint exists, restart without `--resume` in a new `--output-dir`.

Do not edit the checkpoint or change the validation cadence to get past the check: a different cadence changes which epochs are evaluated.

For test runs, do not combine `--resume` with `--override trainer.max_steps=...`, because that shortens the first resumed epoch. Resuming from a checkpoint that stores only rank 0's RNG state continues rank 1 from a distinct reproducible stream, not an exact replay. `bpjepa-pretrain --resume` uses its own loop and does not have this limit.

## 7. Frozen encoder (Tab. 2, 3)

The cache comes from the pretrained online encoder, never from fine-tuned weights.

```bash
DATA="--frames-dir $FRAMES --label-cache-dir $LABELS --soccernet-labels-dir $VAL_LABELS"
CUDA_VISIBLE_DEVICES=0 bpjepa-frozen-cache --config configs/downstream/finetune_100f.yaml --encoder jepa --checkpoint $RUNS/bp_jepa/checkpoints/epoch-100-step-000081900.pt $DATA --output-dir $RUNS/bp_jepa_cache
for seed in 42 203 666; do
  torchrun --standalone --nproc_per_node=2 -m bpjepa.downstream.frozen_head --config configs/downstream/finetune_100f.yaml --cache-registry $RUNS/bp_jepa_cache/registry.json \
    --encoder jepa --seed $seed $DATA --output-dir $RUNS/bp_jepa_head$seed
done
```

Repeat per arm. COMEDIAN uses `--encoder comedian --export $RUNS/comedian_s1/export/comedian_online_spatial.pt` for the cache and `--encoder comedian` for the head. Compare `head_hashes` in `initialization_rank0.json` across arms: same seed, same head.

## 8. Spotting statistics

```bash
bpjepa-analyze-spotting --setting 100f --soccernet-labels-dir $VAL_LABELS --out results_100f \
  --run R=$RUNS/r_100f --run BP-JEPA=$RUNS/bp_jepa_100f --run TC=$RUNS/tc_100f --run BP-JEPA+TC=$RUNS/bp_jepa_tc_100f --run COMEDIAN=$RUNS/comedian_100f
bpjepa-analyze-spotting --setting frozen --soccernet-labels-dir $VAL_LABELS --out results_frozen \
  --run R:42=$RUNS/r_head42 --run R:203=$RUNS/r_head203 ...   # every arm at every seed
```

## 9. Representation evaluations (Tab. 4)

```bash
bpjepa-probes-prepare --clip-manifest $CLIPS --roles configs/evidence/roles.json --evidence-dir $EV --probe-masks configs/probes/probe_masks.npz --output probe_clips.npz
CUDA_VISIBLE_DEVICES=0 bpjepa-probes-extract --clip-table probe_clips.npz --clip-manifest $CLIPS --frame-root $FRAMES/train --output-dir probe_features \
  --arm R0=$RUNS/r/checkpoints/epoch-100-step-000081900.pt --arm EB0=$RUNS/bp_jepa/checkpoints/epoch-100-step-000081900.pt \
  --arm U=$RUNS/tc/checkpoints/epoch-100-step-000081900.pt --arm EB=$RUNS/bp_jepa_tc/checkpoints/epoch-100-step-000081900.pt
bpjepa-probes-analyze --clip-table probe_clips.npz --features-dir probe_features --arms R0 U EB EB0 --out results_probes
```

`configs/probes/probe_masks.npz` holds the masks of the full 4,800-clip table. Any other clip table stops with `probe masks do not match the clip table`, by design. Keep these official masks unchanged for Tab. 4. A test on a subset needs its own file from `bpjepa.probes.lib.replay_masks`, a clip count divisible by 16, and smaller `N_TRAIN` and `N_INNER` values in `bpjepa.probes.lib`; its output is not a paper result.

## 10. Evidence audit (Sec. 5.5, supplementary Sec. 3)

`src/bpjepa/analysis/audit/` holds the audit's functions (boundary statistics, event and control windows, concordance summaries and the enrichment gate). `bpjepa-audit test` runs their self-test on synthetic data. The stages that produced the paper's audit numbers read intermediate files of the paper's runs and are not included, so the audit cannot be re-run from this repository.

## 11. Tables and Fig. 2

```bash
bpjepa-paper check --results RESULTS
bpjepa-paper tables --results RESULTS --output-dir out
```

`RESULTS` is a directory with the outputs of steps 8 and 9 (`results_100f`, `results_frozen`, `results_probes`, with every arm and seed). A missing result is an error; nothing is filled from published values. The stored results of the paper's runs are not included in this repository. Tab. 1 only prints published values from a `paper_data.json` file, which is not included, and is omitted when that file is absent.

Fig. 2 needs real frames and the evidence directory: `bpjepa-fig2 --evidence-dir $EV --frame-root $FRAMES/train --roles configs/evidence/roles.json --output-dir out/figs`. Its PNG equals the paper's Fig. 2 pixel for pixel only with a Pillow built against libjpeg 9 (conda); with pip wheels (libjpeg-turbo) 6.24% of pixels differ, while `fig2_evidence.json` is identical (`docs/KNOWN_DISCREPANCIES.md`, item 3). Apart from Fig. 2, the code that drew the paper's figures is not included.
