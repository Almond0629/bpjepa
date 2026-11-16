"""Regime evidence: same-shot before/after log-ratio per token, conditioned 1-p calibration, top-5% pooling.

Exact arithmetic of the runs behind the paper's target tables. Nothing here reads event annotations.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

from bpjepa.common import canonical_digest, sha256

RHO = .05
T, T_LO, T_HI, SPATIAL = 64, 2, 61, 196
CUT = 1  # CUT reason bit


def topmean(values, mask, rho):
    n = mask.sum(1)
    k = np.maximum(1, np.ceil(rho * n).astype(int))
    sorted_values = np.sort(np.where(mask, values, -np.inf), axis=1)[:, ::-1]
    out = np.where(np.arange(values.shape[1])[None, :] < k[:, None], sorted_values, 0).sum(1) / k
    out[n == 0] = 0
    return out


def log_ratio(pre, post, support):
    score = np.abs(np.log((post + 1e-6) / (pre + 1e-6)))
    assert np.isfinite(score[support]).all()
    return np.where(support, score, 0)


def boundary(raw, mask, valid, reasons):
    """Token statistic |log(m+/m-)| over valid same-shot observations among four transitions on each side."""
    shots = np.cumsum((reasons.astype(np.uint64) & CUT) != 0)
    obs = mask & valid[:, None]
    sums = [np.zeros_like(raw), np.zeros_like(raw)]
    counts = [np.zeros_like(raw, dtype=int), np.zeros_like(raw, dtype=int)]
    n = len(raw)
    for side, offsets in enumerate((range(-4, 0), range(1, 5))):
        for offset in offsets:
            centers = np.arange(max(0, -offset), min(n, n - offset))
            neighbors = centers + offset
            good = obs[neighbors] & (shots[centers] == shots[neighbors])[:, None]
            sums[side][centers] += np.where(good, raw[neighbors], 0)
            counts[side][centers] += good
    support = obs & (counts[0] >= 2) & (counts[1] >= 2)
    usable = valid & (support.sum(1) >= 61)
    support &= usable[:, None]
    pre, post = [s / np.maximum(c, 1) for s, c in zip(sums, counts)]
    return log_ratio(pre, post, support), support, usable


def percentile(ref, values):
    """1-p with p=(1+#{r>=v})/(N+1): ranks / (N+1) over the sorted references (ties use >=)."""
    flat = values.ravel()
    order = np.argsort(flat)
    ranks = np.empty(len(flat), dtype=np.int64)
    ranks[order] = np.searchsorted(ref, flat[order], side='left')
    return (ranks / (len(ref) + 1)).reshape(values.shape)


def pools(state):
    a, m, o = state
    return [('L0', a, m, o), ('L1', a, o), ('L2',)]


def keyname(key):
    return '|'.join(map(str, key))


def state_keys(alignment, motion, n_valid):
    """[alignment, motion bin, overlap bin] per transition. n_valid/196 bins: 0.50, 0.75, 0.90 are inclusive lower edges."""
    overlap = np.searchsorted((.5, .75, .9), n_valid / 196, side='right')
    return np.column_stack((alignment, motion, overlap)).astype(np.int16)


def construct_folds(matches, policy, *, require_size=True):
    ids = sorted(set(matches), key=lambda m: (canonical_digest([policy['seed'], 'calib_fold', m]), m))
    folds = [ids[::2], ids[1::2]]
    if require_size and any(len(x) != policy['matches_per_fold'] for x in folds):
        raise ValueError('calibration fold construction must give two 50-match folds')
    return folds


def fit_references(load_half, half_ids, support_rules, out_dir: Path, *, log=print):
    """Sorted reference statistics per qualified nuisance pool, from half_ids only.

    load_half(half) -> raw, mask, valid, reasons, keys. Pools need min_tokens, min_matches and a bounded largest match share.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    for index, half in enumerate(half_ids):
        raw, mask, valid, reasons, keys = load_half(half)
        values, support, usable = boundary(raw, mask, valid, reasons)
        for state in np.unique(keys[usable], axis=0):
            state = tuple(map(int, state))
            ix = usable & np.all(keys == state, axis=1)
            selected = values[ix][support[ix]]
            parts[state].append(selected)
            counts[state][half.rsplit('::', 1)[0]] += len(selected)
        if (index + 1) % 10 == 0:
            log(f'references {index + 1}/{len(half_ids)}')
    values_by = {s: np.concatenate(v) for s, v in parts.items()}
    parts.clear()
    pool_counts = collections.defaultdict(collections.Counter)
    for state, count in counts.items():
        for pool in pools(state):
            pool_counts[pool].update(count)
    refs, reports = {}, {}
    for pool, count in sorted(pool_counts.items()):
        total = sum(count.values())
        share = max(count.values()) / total
        qualified = (total >= support_rules['min_tokens'] and len(count) >= support_rules['min_matches']
                     and share <= support_rules['max_match_token_share'])
        reports[keyname(pool)] = {'tokens': total, 'matches': len(count), 'max_match_share': share, 'qualified': qualified}
        if qualified:
            values = np.sort(np.concatenate([v for s, v in values_by.items() if pool in pools(s)]))
            path = out_dir / ('reference_' + keyname(pool) + '.npy')
            np.save(path, values, allow_pickle=False)
            refs[keyname(pool)] = {'path': path.name, 'sha256': sha256(path)}
    if not refs:
        raise RuntimeError('no qualified reference pools; check the calibration halves and support rules')
    (out_dir / 'fit.json').write_text(json.dumps({'mapping': '1-p; no zero floor', 'context': 'same shot', 'pools': reports,
                                                  'references': refs, 'half_ids': list(half_ids)}, indent=2, sort_keys=True))
    return load_references(out_dir)


def load_references(directory: Path) -> dict:
    directory = Path(directory)
    refs = {}
    for key, ref in json.loads((directory / 'fit.json').read_text())['references'].items():
        k = key.split('|')
        refs[(k[0], *map(int, k[1:]))] = np.load(directory / ref['path'], mmap_mode='r', allow_pickle=False)
    return refs


def boundary_half(raw, mask, valid, reasons, keys, refs):
    """Per-transition score, reliability and dense per-token 1-p values for one half."""
    values, support, usable = boundary(raw, mask, valid, reasons)
    dense = np.zeros_like(values)
    reliable = usable.copy()
    for state in np.unique(keys[usable], axis=0):
        state = tuple(map(int, state))
        ix = np.flatnonzero(usable & np.all(keys == state, axis=1))
        pool = next((p for p in pools(state) if p in refs), None)
        if pool is None:
            reliable[ix] = False
            continue
        block = dense[ix]
        block[support[ix]] = percentile(refs[pool], values[ix][support[ix]])
        dense[ix] = block
    support &= reliable[:, None]
    score = topmean(dense, support, RHO)
    score[~reliable] = 0
    return score, reliable, dense, support


def half_arrays(score, supported):
    """Per-transition value float32 (NaN where unsupported) and supported bool."""
    supported = np.asarray(supported, bool)
    value = np.where(supported, np.asarray(score, np.float64), np.nan).astype(np.float32)
    assert np.isfinite(value[supported]).all() and np.isnan(value[~supported]).all()
    return {'value': value, 'supported': supported}


def topk_mask(values, support, rho=RHO):
    """Top ceil(rho*n) supported tokens; ties -> lower token index (used by the evidence figure)."""
    n = int(support.sum())
    out = np.zeros(len(values), bool)
    if n == 0:
        return out
    k = max(1, math.ceil(rho * n))
    order = np.lexsort((np.arange(len(values)), -np.where(support, values, -np.inf)))
    out[order[:k]] = True
    return out
