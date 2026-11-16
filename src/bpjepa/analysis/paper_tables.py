#!/usr/bin/env python3
"""Build the paper's tables from a results directory (no data, GPU or checkpoints needed).

Tabs. 2-4 read results_100f/, results_frozen/ and results_probes/, the outputs of bpjepa-analyze-spotting and
bpjepa-probes-analyze. Tab. 1 shows the published values in paper_data.json, which is not part of this
repository.

  bpjepa-paper tables --results RESULTS --output-dir out
  bpjepa-paper check  --results RESULTS (recompute contrasts from the saved bootstrap arrays)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpjepa.analysis.contrasts import difference, pct


NO_FALLBACK = "Tables are never filled from published values"
FROZEN_SEEDS = [42, 203, 666]


def _load(root: Path, name: str) -> dict:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}: results_* come from bpjepa-analyze-spotting and bpjepa-probes-analyze; paper_data.json (published values) is not part of this repository. {NO_FALLBACK}.")
    return json.loads(path.read_text())


def _need(mapping: dict, key: str, source: str):
    if key not in mapping:
        raise ValueError(f"{source} has no {key!r}: analyze every arm and seed. {NO_FALLBACK}.")
    return mapping[key]


def _frozen(root: Path) -> dict:
    comparisons = _need(_load(root, "results_frozen/results.json"), "comparisons", "results_frozen/results.json")
    if comparisons["seeds"] != FROZEN_SEEDS:
        raise ValueError(f"results_frozen/results.json covers seeds {comparisons['seeds']}, the tables need {FROZEN_SEEDS}. {NO_FALLBACK}.")
    return comparisons


def table_recipe(root: Path) -> list[str]:
    data = _load(root, "paper_data.json")
    d = data["recipe"]
    rows = [("A6000", "Our recipe (constant LR, 128 frames, resize)", "constant"), ("A6000", "Scheduled LR (warmup + cosine)", "scheduled_lr"),
            ("A6000", "Random resized crop", "crop"), ("A6000", "64-frame clips", "frames64"), ("A6000", "No pretraining (scratch)", "scratch"),
            ("A6000", "COMEDIAN Stage 1", "comedian_a6000")]
    base = d["constant"]["final"]["tight"]
    lines = ["# Tab. 1: pretraining recipe (100F, validation tight Avg-mAP, %; published values from paper_data.json)", "", "| Stack | Spatial pretraining | Tight | Delta |", "|---|---|---:|---:|"]
    for stack, label, key in rows:
        v = d[key]["final"]["tight"]
        lines.append(f"| {stack} | {label} | {v:.2f} | {'---' if key == 'constant' else f'{v - base:+.2f}'} |")
    r = data["curves_5090"]["R"]["final"]["tight"]
    lines += [f"| RTX 5090 | R (our recipe) | {r:.2f} | --- |",
              f"| RTX 5090 | V-JEPA recipe | {d['vjepa_recipe']['final']['tight']:.2f} | {d['vjepa_recipe']['final']['tight'] - r:+.2f} |"]
    return lines


def table_main(root: Path) -> list[str]:
    runs = _load(root, "results_100f/results.json")
    frozen = _frozen(root)
    tolerances = _load(root, "results_100f/tolerances.json")
    lines = ["# Tab. 2: action spotting on SoccerNet-v2 validation (fixed final epoch, %)", "",
             "| Spatial pretraining | 100F tight | @1 s | @5 s | Loose | Delta vs R [95%] | Frozen tight (3 seeds) | Delta vs R [95%] |", "|---|---:|---:|---:|---:|---|---:|---|"]
    names = {"R0": "R", "EB0": "BP-JEPA", "COMEDIAN": "COMEDIAN Stage 1"}
    for arm in ("R0", "EB0", "COMEDIAN"):
        tight = _need(runs["runs"], arm, "results_100f/results.json runs")["tight_avg_mAP_percentage"]
        c = _need(tolerances, arm, "results_100f/tolerances.json")
        at1, at5, loose = (_need(c, m, f"results_100f/tolerances.json {arm}") for m in ("at1", "at5", "loose"))
        contrast = None if arm == "R0" else _need(runs["contrasts"], f"{arm}-R0", "results_100f/results.json contrasts")
        fm = _need(frozen["mean_arms"], arm, "results_frozen/results.json mean_arms")["tight_avg_mAP_percentage"]
        fc = None if arm == "R0" else _need(frozen["mean_contrasts"], f"{arm}-R0", "results_frozen/results.json mean_contrasts")
        lines.append(f"| {names[arm]} | {tight:.2f} | {at1:.2f} | {at5:.2f} | {loose:.2f} | "
                     + (f"{contrast['difference_percentage_points']:+.2f} [{contrast['marginal95'][0]:.2f}, {contrast['marginal95'][1]:.2f}]" if contrast else "---")
                     + f" | {fm:.2f} | " + (f"{fc['mean_percentage_points']:+.2f} [{fc['mean_marginal95'][0]:.2f}, {fc['mean_marginal95'][1]:.2f}]" if fc else "---") + " |")
    return lines


def table_ablation(root: Path) -> list[str]:
    f100 = _load(root, "results_100f/results.json")["contrasts"]
    fr = _frozen(root)["mean_contrasts"]
    rows = [("BP-JEPA", "EB0-R0", 1), ("TC", "U-R0", 1), ("BP-JEPA+TC", "EB-R0", 1), ("BP-JEPA - TC", "EB0-U", 1), ("Interaction", "(EB-EB0)-(U-R0)", 1)]
    lines = ["# Tab. 3: the 2x2 ablation (delta tight vs R, pp [95%])", "", "| Arm | Frozen, 3 seeds | 100F |", "|---|---|---|"]
    for label, key, _ in rows:
        a, b = _need(fr, key, "results_frozen/results.json mean_contrasts"), _need(f100, key, "results_100f/results.json contrasts")
        lines.append(f"| {label} | {a['mean_percentage_points']:+.2f} [{a['mean_marginal95'][0]:.2f}, {a['mean_marginal95'][1]:.2f}] | "
                     f"{b['difference_percentage_points']:+.2f} [{b['marginal95'][0]:.2f}, {b['marginal95'][1]:.2f}] |")
    return lines


def table_probes(root: Path) -> list[str]:
    r = _load(root, "results_probes/results.json")["results"]
    lines = ["# Tab. 4: probes of the final pretrained encoders", "", "| Arm | Boundary R^2 | Zoom coupling | Eff. rank |", "|---|---:|---:|---:|"]
    for label, key in (("R", "R0"), ("BP-JEPA", "EB0"), ("TC", "U"), ("BP-JEPA+TC", "EB")):
        one, two, three = (_need(r[section]["arms"], key, f"results_probes/results.json {section}") for section in ("readout1", "readout2", "readout3"))
        lines.append(f"| {label} | {one['r2']:.3f} | {two['per_half_mean']:.3f} | {three['point']['effective_rank']:.1f} |")
    return lines


def check_contrasts(root: Path) -> dict:
    """Recompute every contrast from the saved 10,000-draw arrays; must equal results.json exactly."""
    checked = 0
    runs = _load(root, "results_100f/results.json")
    boot = {a: np.load(root / "results_100f" / f"{a}_100f_bootstrap_metrics.npy")[:, 0] for a in runs["runs"]}
    point = {a: r["tight_avg_mAP"] for a, r in runs["runs"].items()}
    for name, c in runs["contrasts"].items():
        assert 100 * difference(point, name) == c["difference_percentage_points"], name
        assert pct(difference(boot, name) * 100) == c["marginal95"], name
        checked += 1
    frozen = _load(root, "results_frozen/results.json")
    comp = frozen["comparisons"]
    seeds = comp["seeds"]
    fb = {s: {a: np.load(root / "results_frozen" / f"{a}_seed{s}_bootstrap_metrics.npy")[:, 0] for a in comp["per_seed"][str(s)]["arms"]} for s in seeds}
    fp = {s: {a: frozen["arms"][f"{a}_seed{s}"]["tight_avg_mAP"] for a in comp["per_seed"][str(s)]["arms"] if f"{a}_seed{s}" in frozen["arms"]} for s in seeds}
    for name, c in comp["mean_contrasts"].items():
        mean_boot = np.mean([difference(fb[s], name) * 100 for s in seeds], axis=0)
        assert pct(mean_boot) == c["mean_marginal95"], name
        checked += 1
    _ = fp
    return {"status": "PASS", "contrasts_recomputed_exactly": checked}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("tables", "check"))
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    if a.action == "check":
        print(json.dumps(check_contrasts(a.results)))
    else:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        text = []
        for f in (table_recipe, table_main, table_ablation, table_probes):
            if f is table_recipe and not (a.results / "paper_data.json").is_file():
                print("Tab. 1 omitted: it only regenerates the published values in paper_data.json.")
                continue
            try:
                text += f(a.results) + [""]
            except (FileNotFoundError, ValueError) as error:
                raise SystemExit(f"error: {error}") from None
        (a.output_dir / "paper_tables.md").write_text("\n".join(text))
        print("\n".join(text))


if __name__ == "__main__":
    main()
