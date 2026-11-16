"""Additive SoccerNet threshold counts for whole-match resampling.

Matching uses the installed official compute_class_scores, independently in each
half. Only additive TP/prediction/GT counts are cached; AP is recomputed globally
for each draw, including absent classes, visibility and tolerance integration.
"""
import numpy as np


def build_counts(matches, *, deltas, framerate=2):
    """matches: sorted {game_id: [(target, closest, detection), half2]}.

    Inputs are the dense arrays produced by the validated evaluation path;
    this module does not parse labels or perform NMS/preprocessing.
    """
    from SoccerNet.Evaluation import ActionSpotting as official
    if list(matches) != sorted(matches) or not matches or framerate != 2:
        raise ValueError('canonical nonempty match universe and 2 fps required')
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or not deltas.size or not np.isfinite(deltas).all() or np.any(deltas <= 0):
        raise ValueError('invalid tolerance grid')
    thresholds = np.linspace(0, 1, 200)
    shape = (len(matches), len(deltas), 3, 200, 17)
    tp, predicted = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    gt = np.zeros((len(matches), 3, 17), dtype=np.int64)
    for game, halves in enumerate(matches.values()):
        if len(halves) != 2:
            raise ValueError('each whole match requires both halves')
        for target, closest, detection in halves:
            target, closest, detection = map(np.asarray, (target, closest, detection))
            if target.ndim != 2 or target.shape[1] != 17 or target.shape != closest.shape or target.shape != detection.shape:
                raise ValueError('dense half arrays must have identical [frames,17] shape')
            if not all(np.isfinite(x).all() for x in (target, closest, detection)):
                raise FloatingPointError('nonfinite evaluator input')
            if not np.isin(target, [-1, 0, 1]).all() or not np.isin(closest, [-1, 0, 1]).all():
                raise ValueError('invalid visibility labels')
            for c in range(17):
                visible, unshown = int((target[:, c] > 0).sum()), int((target[:, c] < 0).sum())
                gt[game, :, c] += [visible + unshown, visible, unshown]
                for tolerance, delta in enumerate(deltas):
                    scores, _, _ = official.compute_class_scores(target[:, c], closest[:, c], detection[:, c], delta * framerate)
                    for visibility, selected in enumerate((np.ones(len(scores), bool), scores[:, 2] > .5, scores[:, 2] < -.5)):
                        keep = selected[:, None] & (scores[:, 0, None] >= thresholds)
                        predicted[game, tolerance, visibility, :, c] += keep.sum(0)
                        tp[game, tolerance, visibility, :, c] += (keep * scores[:, 1, None]).sum(0).astype(np.int64)
    return {'match_ids': list(matches), 'deltas': deltas, 'tp': tp, 'predicted': predicted, 'gt': gt,
            'framerate': framerate}


def evaluate_counts(cache, multiplicities):
    multiplicities = np.asarray(multiplicities)
    n = len(cache['match_ids'])
    if multiplicities.shape != (n,) or multiplicities.dtype.kind not in 'iu' or np.any(multiplicities < 0) or multiplicities.sum() == 0:
        raise ValueError('invalid match multiplicities')
    tp = np.tensordot(multiplicities, cache['tp'], axes=(0, 0))
    predictions = np.tensordot(multiplicities, cache['predicted'], axes=(0, 0))
    gt = np.tensordot(multiplicities, cache['gt'], axes=(0, 0))
    precision = np.divide(tp, predictions, out=np.zeros_like(tp, dtype=float), where=predictions != 0)
    recall = np.divide(tp, gt[None, :, None, :], out=np.zeros_like(tp, dtype=float), where=gt[None, :, None, :] != 0)
    ap = sum(np.where(recall >= threshold, precision, 0).max(axis=2) for threshold in np.arange(11)/10) / 11
    if len(cache['deltas']) == 1:
        per_class = ap[0]
        means = per_class.mean(-1)
    else:
        # Match the official arithmetic order: integrate per-tolerance macro AP
        # separately from per-class AP, with sequential trapezoid accumulation.
        per_class = np.zeros_like(ap[0])
        means = np.zeros(3)
        macro = ap.mean(-1)
        for i in range(len(ap) - 1):
            per_class += (ap[i] + ap[i+1]) / 2
            means += (macro[i] + macro[i+1]) / 2
        per_class /= len(ap) - 1
        means /= len(ap) - 1
        means[2] = means[2] * 17 / 13  # preserve official operation order
    return (means[0], per_class[0], means[1], per_class[1], means[2], per_class[2])


def flattened(result):
    return np.concatenate([np.atleast_1d(x) for x in result])
