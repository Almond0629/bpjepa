#!/usr/bin/env python3
"""Official SoccerNet tight Avg-mAP plus paired whole-match bootstrap for fixed-final spotting predictions.

10,000 draws, PCG64(20260927), over the sorted 100 validation matches, shared by all arms, seeds and settings.
Per-arm counts are additive, so every draw recomputes AP exactly. Validation only; no test path exists.

  bpjepa-analyze-spotting --setting 100f --run R=runs/r_100f --run BP-JEPA=runs/bp_100f ... --soccernet-labels-dir ... --out results_100f
  bpjepa-analyze-spotting --setting frozen --run R:42=runs/r_head42 ... --soccernet-labels-dir ... --out results_frozen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bpjepa.analysis.bootstrap_batch import evaluate_batch
from bpjepa.analysis.contrasts import ALIASES, DISCLOSURES, contrasts, pct, summarize
from bpjepa.analysis.evaluator_io import load_matches
from bpjepa.analysis.spotting_bootstrap import build_counts, evaluate_counts, flattened
from bpjepa.common import sha256
from bpjepa.run_record import write_run_record

B, BOOTSTRAP_SEED, FINAL_INDEX, N_MATCHES = 10000, 20260927, 99, 100
TOLERANCES = {"at1": np.array([1]), "at5": np.array([5]), "loose": np.arange(12) * 5 + 5}  # seconds; official SoccerNet definitions


def bootstrap_draws(n_matches: int = 100) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0, n_matches, (B, n_matches))


def evaluate_run(tag: str, run: Path, games: list, multiplicities: np.ndarray, labels_dir: Path, out: Path, *, final_index: int = FINAL_INDEX) -> tuple[dict, np.ndarray]:
    from SoccerNet.Evaluation import ActionSpotting as official
    start = time.time()
    predictions = run / "val_preds" / f"{final_index}.zip"
    if not predictions.is_file():
        raise FileNotFoundError(f"fixed-final predictions are missing: {predictions}")
    matches = load_matches(games, predictions, labels_dir)
    cache = build_counts(matches, deltas=np.arange(1, 6), framerate=2)
    halves = [h for values in matches.values() for h in values]
    targets, closests, detections = zip(*halves)
    native = official.average_mAP(targets, detections, closests, framerate=2, deltas=np.arange(1, 6))
    counted = evaluate_counts(cache, np.ones(len(games), dtype=np.int64))
    np.testing.assert_array_equal(flattened(native), flattened(counted))
    tolerances = {name: 100 * float(official.average_mAP(targets, detections, closests, framerate=2, deltas=deltas)[0]) for name, deltas in TOLERANCES.items()}
    np.savez(out / f"{tag}_counts.npz", tp=cache["tp"], predicted=cache["predicted"], gt=cache["gt"], deltas=cache["deltas"])
    del matches, halves, targets, closests, detections
    sampled = np.empty((B, 3))
    degenerate = np.zeros((3, 17), dtype=np.int64)
    for i in range(0, B, 32):
        means, _, gt = evaluate_batch(cache, multiplicities[i:i + 32])
        sampled[i:i + 32] = means
        degenerate += (gt == 0).sum(0)
    np.save(out / f"{tag}_bootstrap_metrics.npy", sampled, allow_pickle=False)
    record = {"run": str(run), "predictions_sha256": sha256(predictions), "tight_avg_mAP": float(native[0]),
              "tight_avg_mAP_percentage": 100 * float(native[0]), "per_class": np.asarray(native[1]).tolist(),
              "visible": float(native[2]), "unshown": float(native[4]), "official_count_parity_bitwise": True,
              "endpoint_marginal95_percentage": (np.array(pct(sampled[:, 0])) * 100).tolist(),
              "degenerate_draws_zero_GT_by_visibility_and_class": degenerate.tolist(), "tolerances": tolerances, "seconds": time.time() - start}
    return record, sampled[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=("100f", "frozen"), required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIR or ARM:SEED=DIR",
                        help="arm names: R, BP-JEPA, TC, BP-JEPA+TC, COMEDIAN (or the result-file keys R0, EB0, U, EB)")
    parser.add_argument("--soccernet-labels-dir", type=Path, required=True, help="validation split: <match>/Labels-v2.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-index", type=int, default=FINAL_INDEX, help="zero-based final epoch of val_preds/<index>.zip")
    args = parser.parse_args()
    from SoccerNet.Evaluation import ActionSpotting as official
    games = sorted(official.getListGames(split="valid", dataset="SoccerNet", task="spotting"))
    if len(games) != N_MATCHES:
        raise RuntimeError(f"expected the {N_MATCHES} validation matches, got {len(games)}")
    draws = bootstrap_draws(len(games))
    multiplicities = np.stack([np.bincount(row, minlength=len(games)) for row in draws])
    args.out.mkdir(parents=True, exist_ok=False)
    records, boot = {}, {}
    if args.setting == "100f":
        for item in args.run:
            arm, path = item.split("=", 1)
            arm = ALIASES[arm]
            records[arm], boot[arm] = evaluate_run(f"{arm}_100f", Path(path), games, multiplicities, args.soccernet_labels_dir, args.out, final_index=args.final_index)
        result = {"artifact_kind": "pretraining_100f_final_comparison", "runs": records,
                  "contrasts": contrasts({a: r["tight_avg_mAP"] for a, r in records.items()}, boot)}
    else:
        point = {}
        for item in args.run:
            key, path = item.split("=", 1)
            arm, seed = key.split(":")
            arm, seed = ALIASES[arm], int(seed)
            records[f"{arm}_seed{seed}"], values = evaluate_run(f"{arm}_seed{seed}", Path(path), games, multiplicities, args.soccernet_labels_dir, args.out, final_index=args.final_index)
            point.setdefault(seed, {})[arm] = records[f"{arm}_seed{seed}"]["tight_avg_mAP"]
            boot.setdefault(seed, {})[arm] = values
        result = {"artifact_kind": "frozen_spatial_spotting_final", "arms": records, "comparisons": summarize(point, boot, sorted(point))}
    (args.out / "tolerances.json").write_text(json.dumps({tag: record.pop("tolerances") for tag, record in records.items()}, indent=2, allow_nan=False))
    result.update(fixed_epoch=args.final_index + 1, B=B, bootstrap_seed=BOOTSTRAP_SEED, bit_generator="PCG64", match_ids=games,
                  draws_sha256_of_array=__import__("hashlib").sha256(draws.tobytes()).hexdigest(),
                  test_access=False, disclosures=DISCLOSURES)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_run_record(args.out, stage=f"analyze_spotting_{args.setting}", config={k: str(v) for k, v in vars(args).items()}, seed=BOOTSTRAP_SEED,
                     outputs={"results_sha256": sha256(args.out / "results.json")})
    print(json.dumps({"results": str(args.out / "results.json")}))


if __name__ == "__main__":
    main()
