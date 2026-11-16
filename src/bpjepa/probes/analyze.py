#!/usr/bin/env python3
"""Representation evaluations of final encoders (descriptive; label-exposed train matches). CPU only.

One boundary head per arm fitted head-only on the 512-clip pool (draw 0), then on the common subset of the 60 evaluation
matches: boundary R^2 and within-half Spearman; zoom coupling (within-half rank-partial Spearman with zoom given the
target); feature std, mean norm, effective rank exp(H), participation ratio. 2,000 match-bootstrap draws shared by all arms.
The flag and contrasts are descriptive; nothing here selects a model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bpjepa.probes import lib as L
from bpjepa.run_record import write_run_record

FLAG_MIN = 0.10
INTERACTION = '(EB-EB0)-(U-R0)'
FINAL_PAIRS = (('EB0', 'R0'), ('U', 'R0'), ('EB', 'R0'), ('EB', 'EB0'), ('EB0', 'U'), ('EB', 'U'), ('COMEDIAN', 'R0'), ('COMEDIAN', 'EB0'))
FINAL = {'baseline': 'R0', 'flagged': ('U', 'EB', 'EB0', 'COMEDIAN'), 'c1': FINAL_PAIRS, 'c2': FINAL_PAIRS, 'c3': FINAL_PAIRS, 'interaction': True}


def interaction_diff(get, boot_get):
    """(EB - EB0) - (U - R0): the point value and the per-draw values (computed within each draw)."""
    point = (get('EB') - get('EB0')) - (get('U') - get('R0'))
    draws = (boot_get('EB') - boot_get('EB0')) - (boot_get('U') - boot_get('R0'))
    return {'diff': float(point), 'ci95': np.quantile(draws, [.025, .975]).tolist()}


def analyze(data, arms, target_mean, zoom, count, spec=FINAL):
    """arms: {name: (z_pool [512,64,192], z_eval [4,1440,64,192])}. -> results dict."""
    heads, fit_info = {}, {}
    for name, (zp, _) in arms.items():
        heads[name], fit_info[name] = L.fit_arm_head(data, zp, target_mean)
    table = L.aux_table_for(data, target_mean)
    rows, common, y_all = L.eval_rows(data)
    keep = np.flatnonzero(common)
    rows_c, y = rows[keep], y_all[keep]
    c_idx = data.ev[rows_c[:, 1]]
    match, half = data.matches[c_idx], data.half_ids[c_idx]
    matches = np.array(sorted(set(data.matches[data.ev])))
    draws = L.boot_draws(len(matches))
    preds = {name: L.predict_rows(heads[name], data, z_eval, table, rows_c) for name, (_, z_eval) in arms.items()}
    assert all(np.isfinite(p).all() for p in preds.values())
    res1, boot1 = L.evaluate_arms(y, preds, target_mean, match, half, draws, matches)
    names = list(arms)
    pairs1 = spec['c1'] if spec['c1'] else tuple((a, b) for i, a in enumerate(names) for b in names[i + 1:])
    contrasts1 = {f'{a}-{b}': L.contrast(res1, boot1, a, b) for a, b in pairs1 if a in res1 and b in res1}
    if spec['interaction']:
        contrasts1[INTERACTION] = {key: interaction_diff(lambda n: res1[n]['r2'] if key == 'r2' else res1[n]['spearman_within_half'],
                                                         lambda n: boot1[n][key]) for key in ('r2', 'spearman')}
    # own-eligible rows (the mean-difference head's own eligibility) equal the common subset for B
    readout1 = {'rows_all_eval_rows': int(len(y_all)), 'rows_common_subset': int(len(y)),
                'rows_own_eligible_meandiff': int(common.sum()), 'halves': int(len(set(half))),
                'constant': target_mean, 'arms': res1, 'contrasts': contrasts1}

    # ---- Readout 2 on the common-subset rows with a defined zoom signal
    zv = zoom[rows_c[:, 1], rows_c[:, 2]]
    ok = np.isfinite(zv)
    half_match = {h: x for h, x in zip(half[ok], np.searchsorted(matches, match[ok]))}
    res2, boot2 = {}, {}
    for name in names + ['target']:
        yhat = y[ok] if name == 'target' else preds[name][ok]
        if name == 'target':
            vals = L.half_stat(lambda a, b: _spearman(a, b), half[ok], y[ok], zv[ok])
        else:
            vals = L.half_stat(L.partial_spearman, half[ok], y[ok], yhat, zv[ok])
        point, boot = L.per_half_stat_boot(vals, half_match, matches, draws)
        res2[name] = {'per_half_mean': point, 'ci95': np.quantile(boot, [.025, .975]).tolist(), 'halves': len(vals)}
        boot2[name] = boot
    contrasts2 = {}
    for a, b in spec['c2']:
        if a in res2 and b in res2:
            diff = boot2[a] - boot2[b]
            contrasts2[f'{a}-{b}'] = {'diff': res2[a]['per_half_mean'] - res2[b]['per_half_mean'],
                                      'ci95': np.quantile(diff, [.025, .975]).tolist()}
    if spec['interaction']:
        contrasts2[INTERACTION] = interaction_diff(lambda n: res2[n]['per_half_mean'], lambda n: boot2[n])
    flags = {}
    base = spec['baseline']
    for a in spec['flagged']:
        if a in res2 and base in res2:
            c_ = contrasts2[f'{a}-{base}']
            flags[a] = bool(res2[a]['per_half_mean'] >= FLAG_MIN and res2[a]['ci95'][0] > 0 and c_['ci95'][0] > 0)
    readout2 = {'rows': int(ok.sum()), 'common_subset_rows': int(len(y)), 'dropped_fraction': float(1 - ok.mean()),
                'halves': int(len(set(half[ok]))), 'zoom_signal': 'mean |log s_j| over valid, ECC-accepted transitions '
                'in start+2(t-2) .. start+2(t+2)+1 (ruling 2026-09-30)', 'arms': res2, 'contrasts': contrasts2,
                'framing_shortcut_flag_descriptive': flags, 'flag_rule': 'partial Spearman >= 0.10 with lower bound > 0 '
                f'and the difference to {base if spec["interaction"] else "frozen R"} has lower bound > 0'}

    # ---- Readout 3 (clarification 2026-10-01): raw-feature scale and geometry on the same common-subset rows and draws
    mi = np.searchsorted(matches, match)
    res3, boot3 = {}, {}
    for name, (_, z_eval) in arms.items():
        point, boot = L.geometry_boot(L.match_sums(z_eval[rows_c[:, 0], rows_c[:, 1], rows_c[:, 2]], mi, len(matches)), draws)
        res3[name] = {'point': point, 'ci95': {k: np.quantile(boot[k], [.025, .975]).tolist() for k in L.GEOMETRY_STATS}}
        boot3[name] = boot
    contrasts3 = {f'{a}-{b}': {k: {'diff': res3[a]['point'][k] - res3[b]['point'][k],
                                   'ci95': np.quantile(boot3[a][k] - boot3[b][k], [.025, .975]).tolist()} for k in L.GEOMETRY_STATS}
                  for a, b in spec['c3'] if a in res3 and b in res3}
    if spec['interaction']:
        contrasts3[INTERACTION] = {k: interaction_diff(lambda n: res3[n]['point'][k], lambda n: boot3[n][k]) for k in L.GEOMETRY_STATS}
    readout3 = {'rows': int(len(y)), 'matches': int(len(matches)), 'dimensions': int(next(iter(arms.values()))[1].shape[-1]),
                'arms': res3, 'contrasts': contrasts3}
    return {'fit': fit_info, 'readout1': readout1, 'readout2': readout2, 'readout3': readout3}


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip-table', type=Path, required=True)
    ap.add_argument('--features-dir', type=Path, required=True, help='features_<ARM>.npz from bpjepa-probes-extract')
    ap.add_argument('--arms', nargs='+', default=['R0', 'U', 'EB', 'EB0', 'COMEDIAN'], help='result-file keys: R0=R, EB0=BP-JEPA, U=TC, EB=BP-JEPA+TC')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = L.load_clip_table(args.clip_table)
    arms = {}
    for name in args.arms:
        with np.load(args.features_dir / f'features_{name}.npz', allow_pickle=False) as z:
            arms[name] = (z['z_pool'], z['z_eval'])
    spec = dict(FINAL)
    spec['interaction'] = all(a in arms for a in ('R0', 'U', 'EB', 'EB0'))
    target_mean = L.pool_target_mean(data)
    results = analyze(data, arms, target_mean, data.zoom, None, spec)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'results.json').write_text(json.dumps({'mode': 'final', 'results': results}, indent=2))
    write_run_record(args.out, stage='probes_analyze', config={k: str(v) for k, v in vars(args).items()}, seed=L.BOOT_SEED)


if __name__ == '__main__':
    raise SystemExit(main())
