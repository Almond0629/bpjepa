"""Library for the final-checkpoint representation evaluations (boundary R^2, zoom coupling, effective rank).

Clip table: 12 clips of each audit half, in a fixed hash order (4,800 clips, 300 batches of 16). Head-fit pool: the first
512 clips of the 140 non-evaluation matches (mask draw 0). Evaluation: 1,440 clips of the other 60 matches, all 4 mask
draws. Bootstrap: 2,000 match draws, PCG64(20260927), shared by all arms (not the spotting draws). Descriptive only.
"""
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SEED = '20260928'
DRAWS, CLIPS_PER_HALF, BATCH = 4, 12, 16
N_TRAIN, N_INNER = 140, 28
T = 64
BOOT, BOOT_SEED = 2000, 20260927
POOL_TAG, POOL_N = 'eb_phase3_headfit|20260930|', 512
HEADFIT_SEED = 20260929
SPLIT_TAG, INNER_FRACTION = 'eb_phase2_headfit_split|20260929|', 0.2
FIT_LR, FIT_WD, FIT_MAX_EPOCHS, FIT_PATIENCE, FIT_BATCH = 1e-3, 1e-4, 200, 5, 16
AUX_HEAD_SEED = 1
CUT = 1


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------- clip table

def selection(entries, audit_halves):
    """Selected clip manifest rows, in the fixed batch order."""
    audit_halves = set(audit_halves)
    by = collections.defaultdict(list)
    for i, e in enumerate(entries):
        if e['half_id'] in audit_halves:
            by[e['half_id']].append(i)
    chosen = []
    for half in sorted(by):
        chosen += sorted(by[half], key=lambda i: _sha(f"eb_probe|{SEED}|{half}|{entries[i]['start_frame']}"))[:CLIPS_PER_HALF]
    return sorted(chosen, key=lambda i: _sha(f"eb_probe_batch|{SEED}|{entries[i]['half_id']}|{entries[i]['start_frame']}"))


def split(matches):
    """match -> 'fit' | 'inner' | 'eval' (first 140 train, last 28 of those inner, remaining 60 evaluation)."""
    ranked = sorted(set(matches), key=lambda x: _sha(f"eb_probe_split|{SEED}|{x}"))
    role = {x: 'eval' for x in ranked[N_TRAIN:]}
    role.update({x: 'inner' for x in ranked[N_TRAIN - N_INNER:N_TRAIN]})
    role.update({x: 'fit' for x in ranked[:N_TRAIN - N_INNER]})
    return role


def pool_and_eval(half_ids, starts):
    """-> (pool clip indices, eval clip indices, role by match), from the clip table."""
    matches = np.array([h.rsplit('::', 1)[0] for h in half_ids])
    role = split(matches)
    is_eval = np.array([role[x] == 'eval' for x in matches])
    keyed = sorted((hashlib.sha256(f'{POOL_TAG}{half_ids[i]}|{starts[i]}'.encode()).hexdigest(), int(i))
                   for i in np.flatnonzero(~is_eval))
    pool = np.array([i for _, i in keyed[:POOL_N]])
    return pool, np.flatnonzero(is_eval), role


def grid(values, support, start):
    """Per-tubelet value [64] (nan = unsupported) and source transition [64] (-1)."""
    val, src = np.full(T, np.nan), np.full(T, -1)
    for t in range(2, 62):
        cands = [start + 2 * t] + ([start + 2 * t + 1] if t < 61 else [])
        ok = [x for x in cands if x < len(support) and support[x]]
        if ok:
            best = max(ok, key=lambda x: (values[x], -x))
            val[t], src[t] = values[best], best
    return val, src


def log_scale(affine, status):
    """log sqrt|det A| where an alignment was accepted (status 1 or 2) and the affine is finite, else NaN."""
    affine = np.asarray(affine, np.float64)
    det = affine[:, 0, 0] * affine[:, 1, 1] - affine[:, 0, 1] * affine[:, 1, 0]
    ok = np.isin(status, (1, 2)) & np.isfinite(affine).all(axis=(1, 2)) & (np.abs(det) > 0)
    out = np.full(len(affine), np.nan)
    out[ok] = 0.5 * np.log(np.abs(det[ok]))
    return out


def prepare_clip_table(output, *, manifest_path, roles_path, evidence_dir, probe_masks):
    """Write probe_clips.npz: clips, float64 targets, cut bits, stored mask supports, and the zoom signal of evaluation clips."""
    from bpjepa.evidence import pipeline as E
    entries = json.loads(Path(manifest_path).read_text())['entries']
    audit = json.loads(Path(roles_path).read_text())['audit_half_ids']
    order = selection(entries, audit)
    half_ids = np.array([entries[i]['half_id'] for i in order])
    starts = np.array([entries[i]['start_frame'] for i in order])
    n = len(order)
    value = np.zeros((n, T))
    cut = np.zeros((n, 127), bool)
    rows_by = collections.defaultdict(list)
    for c, half in enumerate(half_ids):
        rows_by[str(half)].append(c)
    for half, rows in sorted(rows_by.items()):
        _, tg = E._load_npz(E.target_path(evidence_dir, half))
        _, r = E._load_npz(E.raw_path(evidence_dir, half))
        for c in rows:
            start = int(starts[c])
            value[c] = grid(tg['score64'], tg['supported'], start)[0]
            cut[c] = (r['reasons'][start:start + 127] & CUT) != 0
    zoom = zoom_signal(half_ids, starts, lambda half: _zoom_inputs(evidence_dir, half))
    with np.load(probe_masks, allow_pickle=False) as stored:
        visible, complete = stored['visible'], stored['complete']
    if visible.shape[1] != n:
        raise ValueError('probe masks do not match the clip table')
    np.savez_compressed(output, half_ids=half_ids, starts=starts, manifest_rows=np.array(order), B_value=value, cut=cut,
                        visible=visible, complete=complete, zoom=zoom)
    return output


def _zoom_inputs(evidence_dir, half):
    from bpjepa.evidence import pipeline as E
    _, m = E.load_measurements(evidence_dir, half)
    _, r = E._load_npz(E.raw_path(evidence_dir, half))
    return m['affine'], m['alignment_status'], r['states']


def zoom_signal(half_ids, starts, load):
    """Tubelet zoom of the evaluation clips: mean |log s_j| over valid, ECC-accepted (alignment 1) transitions among clip
    transitions 2(t-2) .. min(2(t+2)+1, 126); NaN if none. load(half) -> affine, alignment_status, final states."""
    _, ev, _ = pool_and_eval(half_ids, starts)
    good = np.zeros((len(ev), 127), bool)
    mag = np.zeros((len(ev), 127))
    rows_by = collections.defaultdict(list)
    for k, c in enumerate(ev):
        rows_by[str(half_ids[c])].append((k, int(starts[c])))
    for half, rows in sorted(rows_by.items()):
        affine, status, states = load(half)
        scale = log_scale(affine, status)
        for k, start in rows:
            g = (status[start:start + 127] == 1) & np.isfinite(scale[start:start + 127])
            mag[k] = np.where(g, np.abs(scale[start:start + 127]), 0.0)
            good[k] = g & (states[start:start + 127] == 1)
    csum = np.concatenate([np.zeros((len(ev), 1)), np.cumsum(np.where(good, mag, 0.0), 1)], 1)
    ccnt = np.concatenate([np.zeros((len(ev), 1)), np.cumsum(good, 1)], 1)
    zoom = np.full((len(ev), T), np.nan)
    for t in range(2, 62):
        lo, hi = 2 * (t - 2), min(2 * (t + 2) + 1, 126)
        count = ccnt[:, hi + 1] - ccnt[:, lo]
        with np.errstate(invalid='ignore', divide='ignore'):
            zoom[:, t] = np.where(count > 0, (csum[:, hi + 1] - csum[:, lo]) / np.maximum(count, 1), np.nan)
    return zoom


def replay_masks(mask_config, n):
    """Regenerate the small-group mask supports from the sampler. Reproduces probe_masks.npz only under torch 2.0.1,
    whose CPU generator produced the stored masks; torch 2.11 draws different integers from the same seeds."""
    import torch
    from bpjepa.jepa.support import group_support
    from bpjepa.samplers.vjepa_multimask import VJEPAMultiMaskSampler
    sampler = VJEPAMultiMaskSampler(mask_config)
    gens = [torch.Generator().manual_seed(int(SEED) + d) for d in range(DRAWS)]
    visible = np.zeros((DRAWS, n, 196), bool)
    complete = np.zeros((DRAWS, n, T), bool)
    for b in range(n // BATCH):
        for d in range(DRAWS):
            small = sampler.sample(BATCH, T, 14, 14, generator=gens[d])[0]
            assert small.name == 'small'
            support = group_support(small, T, 14, 14)
            visible[d, b * BATCH:(b + 1) * BATCH] = support.visible.numpy()
            complete[d, b * BATCH:(b + 1) * BATCH] = support.complete.numpy()
    return visible, complete


class StoredSmallMasks:
    """The evaluation masks, identical for every arm and independent of the torch RNG version (configs/probes/probe_masks.npz)."""

    def __init__(self, visible, complete):
        self.visible, self.complete = visible, complete

    def small(self, draw, batch):
        """Rebuild the MaskGroup of a batch of 16 clips: bitwise the sampler's context, target and union indices."""
        import torch
        from bpjepa.jepa.support import group_support
        from bpjepa.samplers.vjepa_multimask import MaskGroup
        sl = slice(BATCH * batch, BATCH * (batch + 1))
        v = torch.as_tensor(np.array(self.visible[draw, sl], copy=True), dtype=torch.bool)
        union = (~v[:, None, :]).expand(-1, T, -1).reshape(BATCH, -1).clone()
        ci = [(~row).nonzero().flatten() for row in union]
        ti = [row.nonzero().flatten() for row in union]
        group = MaskGroup('small', torch.stack([x[:min(y.numel() for y in ci)] for x in ci]),
                          torch.stack([x[:min(y.numel() for y in ti)] for x in ti]), union)
        support = group_support(group, T, 14, 14)
        if not np.array_equal(support.complete.numpy(), self.complete[draw, sl]):
            raise RuntimeError('reconstructed complete support differs from the stored masks')
        return group


def load_clip_table(path):
    z = dict(np.load(path, allow_pickle=False))
    half_ids, starts = z['half_ids'], z['starts']
    pool, ev, role = pool_and_eval(half_ids, starts)
    matches = np.array([h.rsplit('::', 1)[0] for h in half_ids])
    return SimpleNamespace(t={'B_value': z['B_value'], 'cut': z['cut']}, visible=z['visible'], complete=z['complete'],
                           half_ids=half_ids, starts=starts, manifest_rows=z['manifest_rows'], pool=pool, ev=ev, role=role,
                           matches=matches, zoom=z['zoom'])


def rows_for(value, complete):
    """Eligible (draw, clip, t): target supported, t in allowed range, tubelets t-2..t+2 complete."""
    win = np.zeros_like(complete)
    win[..., 2:T - 2] = np.all([complete[..., 2 + k:T - 2 + k] for k in range(-2, 3)], axis=0)
    ok = win & np.isfinite(value)[None]
    return np.argwhere(ok)


def r2(y, pred, const):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - const) ** 2)


def spearman_by_half(y, pred, half):
    from scipy.stats import spearmanr
    out = {}
    for h in np.unique(half):
        k = half == h
        if k.sum() >= 3 and np.ptp(y[k]) > 0 and np.ptp(pred[k]) > 0:
            out[h] = spearmanr(y[k], pred[k]).correlation
    return out


def boot_draws(n_matches):
    return np.random.Generator(np.random.PCG64(BOOT_SEED)).integers(0, n_matches, (BOOT, n_matches))


def shift(x, offset, axis):
    """out[..., i, ...] = x[..., i + offset, ...], False/0 outside."""
    out = np.zeros_like(x)
    n = x.shape[axis]
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    if offset >= 0:
        src[axis], dst[axis] = slice(offset, n), slice(0, n - offset)
    else:
        src[axis], dst[axis] = slice(0, n + offset), slice(-offset, n)
    out[tuple(dst)] = x[tuple(src)]
    return out


def same_shot_from_cut(cut):
    """[n,127] transition CUT bits -> (side_ok [n,64,4], dz_ok [n,64]); target_table.clip_same_shot, vectorized."""
    dz = np.zeros((len(cut), T), bool)
    dz[:, :63] = ~(cut[:, 0:126:2] | cut[:, 1:127:2] | cut[:, 2:128:2])
    side = np.stack([shift(dz, -2, 1) & shift(dz, -1, 1), shift(dz, -1, 1), dz, dz & shift(dz, 1, 1)], -1)
    side[:, :2, 0] = False
    side[:, :1, 1] = False
    return side, dz


def aux_table_for(data, target_mean):
    """AuxTable over the clip table: y = B_value (NaN unsupported), same-shot flags from the cut bits."""
    import torch
    from bpjepa.jepa.boundary import AuxTable
    value = data.t['B_value']
    supported = np.isfinite(value)
    y = np.where(supported, value, np.nan).astype(np.float32)
    side, dz = same_shot_from_cut(data.t['cut'])
    return AuxTable('EB', torch.from_numpy(y), torch.from_numpy(supported), torch.from_numpy(side),
                    torch.from_numpy(dz), float(target_mean))


def pool_target_mean(data):
    """Mean boundary target over the pool's rows (supported target, complete t-2..t+2 window; draw 0)."""
    rows = rows_for(data.t['B_value'][data.pool], data.complete[[0]][:, data.pool])
    return float(data.t['B_value'][data.pool][rows[:, 1], rows[:, 2]].mean())


# ----------------------------------------------------------------------------- head fit and prediction

def inner_split(half_ids, fraction=INNER_FRACTION):
    """Split clips by match: the round(fraction*n) matches with the smallest split hash are inner."""
    matches = sorted({h.rsplit('::', 1)[0] for h in half_ids})
    ordered = sorted(matches, key=lambda m: hashlib.sha256(f'{SPLIT_TAG}{m}'.encode()).hexdigest())
    inner = set(ordered[:max(1, round(fraction * len(ordered)))])
    is_inner = [h.rsplit('::', 1)[0] in inner for h in half_ids]
    return [k for k, f in enumerate(is_inner) if not f], [k for k, f in enumerate(is_inner) if f], sorted(inner)


def fit_head(kind, head, table, cache, train_idx, val_idx, device):
    """Head-only fit on cached pooled features z (encoder frozen)."""
    import torch
    from bpjepa.jepa.boundary import aux_loss, aux_predictions
    z = cache['z'].to(device)
    window, padding = cache['window'].to(device), cache['padding'].to(device)
    y, supported, side_ok, dz_ok = table.lookup(cache['row_ids'], device)
    params = list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=FIT_LR, weight_decay=FIT_WD)

    def predict(idx):
        pred, avail = aux_predictions(kind, head, z[idx], side_ok[idx], dz_ok[idx])
        return pred, avail & supported[idx] & window[idx] & ~padding[idx][:, None]

    def inner():
        with torch.no_grad():
            idx = torch.as_tensor(val_idx, device=device)
            pred, ok = predict(idx)
            target = y[idx][ok].double()
            error = pred[ok].double() - target
            sse = float((error ** 2).sum())
            sst = float(((target - target.mean()) ** 2).sum()) if target.numel() else float('nan')
            return sse / max(target.numel(), 1), 1 - sse / sst, int(target.numel())

    best_mse, best_r2, n_val = inner()
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    bad, best_epoch, epochs_run = 0, 0, 0
    order = np.random.Generator(np.random.PCG64(HEADFIT_SEED))
    epoch0_mse = best_mse
    for epoch in range(FIT_MAX_EPOCHS):
        epochs_run = epoch + 1
        shuffled = order.permutation(train_idx)
        for start in range(0, len(shuffled), FIT_BATCH):
            idx = torch.as_tensor(shuffled[start:start + FIT_BATCH], device=device)
            pred, avail = aux_predictions(kind, head, z[idx], side_ok[idx], dz_ok[idx])
            loss, _ = aux_loss(pred, avail, y[idx], supported[idx], window[idx], padding[idx])
            loss = loss + sum(p.sum() * 0 for p in params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mse, r2_value, _ = inner()
        if mse < best_mse - 1e-12:
            best_mse, best_r2, best_epoch, bad = mse, r2_value, epoch + 1, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= FIT_PATIENCE:
                break
    head.load_state_dict(best_state)
    return {'epochs_run': epochs_run, 'best_epoch': best_epoch, 'inner_mse': best_mse, 'inner_r2': best_r2,
            'epoch0_inner_mse': epoch0_mse, 'inner_eligible_anchors': n_val,
            'a': float(head.a.detach()), 'b': float(head.b.detach())}


def fit_arm_head(data, z_pool, target_mean, device='cpu'):
    """One head for an arm from its pool features z_pool [512,64,192] (draw 0)."""
    import torch
    from bpjepa.jepa.boundary import build_aux_head, complete_window
    table = aux_table_for(data, target_mean)
    head = build_aux_head(AUX_HEAD_SEED, target_mean, device=device)
    cache = {'z': torch.from_numpy(np.asarray(z_pool, np.float32)),
             'window': complete_window(torch.from_numpy(data.complete[0, data.pool])),
             'padding': torch.zeros(len(data.pool), dtype=torch.bool),
             'row_ids': torch.from_numpy(data.pool.astype(np.int64))}
    train_idx, val_idx, inner = inner_split(data.half_ids[data.pool])
    info = fit_head('EB', head, table, cache, train_idx, val_idx, device)
    info.update(pool_clips=len(data.pool), inner_clips=len(val_idx), inner_matches=len(inner))
    return head, info


def predict_rows(head, data, z_eval, table, rows):
    """Head predictions at rows (d, c_local, t) of the evaluation clips; z_eval [4, n_ev, 64, 192]."""
    import torch
    from bpjepa.jepa.boundary import aux_predictions
    out = np.full(len(rows), np.nan)
    side_ok, dz_ok = table.side_ok[torch.from_numpy(data.ev)], table.dz_ok[torch.from_numpy(data.ev)]
    with torch.no_grad():
        for d in range(z_eval.shape[0]):
            pred, avail = aux_predictions('EB', head, torch.from_numpy(np.asarray(z_eval[d], np.float32)), side_ok, dz_ok)
            sel = np.flatnonzero(rows[:, 0] == d)
            out[sel] = pred.numpy()[rows[sel, 1], rows[sel, 2]]
    return out


def eval_rows(data):
    """Evaluation rows (draw, local clip, t) with a supported target and complete t-2..t+2, and the common subset
    (mean-difference head eligible: same shot on both sides of t, i.e. dz_{t-1} and dz_t not cut)."""
    value = data.t['B_value'][data.ev]
    rows = rows_for(value, data.complete[:, data.ev])
    _, dz = same_shot_from_cut(data.t['cut'][data.ev])
    common = dz[rows[:, 1], rows[:, 2] - 1] & dz[rows[:, 1], rows[:, 2]]
    y = value[rows[:, 1], rows[:, 2]].astype(float)
    return rows, common, y


# ----------------------------------------------------------------------------- statistics

def per_half_stat_boot(values_by_half, half_match, matches, draws):
    """Mean over halves of a per-half statistic, with its match-bootstrap draws.
    values_by_half: dict half -> value; half_match: dict half -> index into `matches`; draws [B, n_matches]."""
    per = np.zeros(len(matches))
    cnt = np.zeros(len(matches))
    for h, v in values_by_half.items():
        per[half_match[h]] += v
        cnt[half_match[h]] += 1
    boot = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
    point = float(np.mean(list(values_by_half.values()))) if values_by_half else None
    return point, boot


def evaluate_arms(y, preds, const, match, half, draws, matches):
    """R^2 and within-half Spearman, returning also the bootstrap draws so contrasts share them.
    -> (results per arm, boot per arm = {'r2': [B], 'spearman': [B]})"""
    mi = np.searchsorted(matches, match)
    sst = np.bincount(mi, (y - const) ** 2, len(matches))
    half_match = {h: x for h, x in zip(half, mi)}
    res, boot = {}, {}
    for name, p in preds.items():
        sse = np.bincount(mi, (y - p) ** 2, len(matches))
        bs = 1 - sse[draws].sum(1) / sst[draws].sum(1)
        rho = spearman_by_half(y, p, half)
        point, sb = per_half_stat_boot(rho, half_match, matches, draws)
        res[name] = {'r2': float(r2(y, p, const)), 'r2_ci95': np.quantile(bs, [.025, .975]).tolist(),
                     'spearman_within_half': point, 'spearman_ci95': np.quantile(sb, [.025, .975]).tolist(),
                     'rows': int(len(y)), 'halves': int(len(rho))}
        boot[name] = {'r2': bs, 'spearman': sb}
    return res, boot


def contrast(res, boot, a, b):
    """a - b for R^2 and within-half Spearman, with the shared-draw 95% interval."""
    out = {}
    for key, point in (('r2', lambda n: res[n]['r2']), ('spearman', lambda n: res[n]['spearman_within_half'])):
        diff = boot[a][key] - boot[b][key]
        out[key] = {'diff': float(point(a) - point(b)), 'ci95': np.quantile(diff, [.025, .975]).tolist()}
    return out


def partial_spearman(y, yhat, zoom):
    """Rank-based partial Spearman: rank y, yhat and zoom, residualize the ranks of yhat and zoom on the ranks of y
    (OLS with intercept), then Spearman of the residuals. None if degenerate."""
    from scipy.stats import rankdata, spearmanr
    if len(y) < 4 or np.ptp(y) == 0 or np.ptp(yhat) == 0 or np.ptp(zoom) == 0:
        return None
    ry, rp, rz = rankdata(y), rankdata(yhat), rankdata(zoom)
    ry = ry - ry.mean()
    denom = float(ry @ ry)
    res_p = rp - rp.mean() - (ry @ (rp - rp.mean())) / denom * ry
    res_z = rz - rz.mean() - (ry @ (rz - rz.mean())) / denom * ry
    if np.ptp(res_p) < 1e-12 or np.ptp(res_z) < 1e-12:
        return None
    return float(spearmanr(res_p, res_z).correlation)


def half_stat(fn, half, *arrays):
    """{half: fn(arrays restricted to the half)} skipping None."""
    out = {}
    for h in np.unique(half):
        k = half == h
        v = fn(*(a[k] for a in arrays))
        if v is not None:
            out[h] = v
    return out


# ----------------------------------------------------------------------------- feature scale and geometry (Readout 3)

GEOMETRY_STATS = ('feature_std', 'mean_norm', 'effective_rank', 'participation_ratio')


def match_sums(z, mi, n_matches):
    """Per-match sufficient statistics of raw feature rows z [N, D]: (n [M], sum z [M, D], sum zz^T [M, D, D], sum |z| [M])."""
    z = np.asarray(z, np.float64)
    n = np.bincount(mi, minlength=n_matches).astype(np.float64)
    norm = np.bincount(mi, np.linalg.norm(z, axis=1), n_matches)
    s, ss = np.zeros((n_matches, z.shape[1])), np.zeros((n_matches, z.shape[1], z.shape[1]))
    for m in range(n_matches):
        x = z[mi == m]
        s[m], ss[m] = x.sum(0), x.T @ x
    return n, s, ss, norm


def feature_geometry(n, s, ss, norm):
    """Feature std (mean over dimensions of the per-dimension std over the pooled rows), mean |z|, effective rank exp(H) and
    participation ratio (sum l)^2 / sum l^2 of the centred pooled covariance (population, 1/n), from pooled sums."""
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    lam = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2), 0, None)
    p = lam[lam > 0] / lam.sum()
    return {'feature_std': float(np.sqrt(np.clip(np.diag(cov), 0, None)).mean()), 'mean_norm': float(norm / n),
            'effective_rank': float(np.exp(-(p * np.log(p)).sum())), 'participation_ratio': float(lam.sum() ** 2 / (lam ** 2).sum())}


def geometry_boot(sums, draws):
    """Point statistics on all matches, and the match-bootstrap draws [B] of every statistic (draws [B, n_matches]).
    A draw's pooled sums are the draw-count-weighted sums of the per-match sums, so no covariance is refit from rows."""
    n, s, ss, norm = sums
    point = feature_geometry(n.sum(), s.sum(0), ss.sum(0), norm.sum())
    weights = np.stack([np.bincount(d, minlength=len(n)) for d in draws]).astype(np.float64)
    boot = {k: np.empty(len(draws)) for k in GEOMETRY_STATS}
    for b, w in enumerate(weights):
        g = feature_geometry(w @ n, w @ s, np.tensordot(w, ss, 1), w @ norm)
        for k in GEOMETRY_STATS:
            boot[k][b] = g[k]
    return point, boot
