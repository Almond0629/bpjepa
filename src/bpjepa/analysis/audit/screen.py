"""Same-shot screen functions of the evidence audit, with a synthetic self-test."""
import numpy as np

from bpjepa.analysis.audit import diagnostics as m


def boundary(raw, mask, valid, reasons):
    shots = np.cumsum((reasons.astype(np.uint64) & 1) != 0)
    obs = mask & valid[:, None]
    sums = [np.zeros_like(raw), np.zeros_like(raw)]
    counts = [np.zeros_like(raw, dtype=int), np.zeros_like(raw, dtype=int)]
    n = len(raw)
    for side, offsets in enumerate((range(-4, 0), range(1, 5))):
        for offset in offsets:
            centers = np.arange(max(0, -offset), min(n, n-offset))
            neighbors = centers + offset
            good = obs[neighbors] & (shots[centers] == shots[neighbors])[:, None]
            sums[side][centers] += np.where(good, raw[neighbors], 0)
            counts[side][centers] += good
    support = obs & (counts[0] >= 2) & (counts[1] >= 2)
    usable = valid & (support.sum(1) >= 61)
    support &= usable[:, None]
    pre, post = [s / np.maximum(c, 1) for s, c in zip(sums, counts)]
    return m.boundary(pre, post, support, 'log_ratio'), support, usable


def percentile(ref, values):
    flat = values.ravel()
    order = np.argsort(flat)
    ranks = np.empty(len(flat), dtype=np.int64)
    ranks[order] = np.searchsorted(ref, flat[order], side='left')
    return (ranks / (len(ref)+1)).reshape(values.shape)


def decision(score):
    if score is None:
        return 'INSUFFICIENT'
    if score < .60:
        return 'STOP_E_CHAIN'
    if score >= .62:
        return 'WORTH_FURTHER_TESTING_NOT_E_PASS'
    return 'INCONCLUSIVE_NO_CONTINUATION'


def modal(values):
    values = values[(values >= 0) & (values < 3)].astype(int)
    if not len(values):
        return -1
    bins = np.bincount(values, minlength=3)
    return int(np.flatnonzero(bins == bins.max())[-1])


original_summary = m.subset_summary
def summary(rows, values, keep, confidence=True):
    out = original_summary(rows, values, keep, False)
    if confidence and np.any(keep):
        records = [r for r,k in zip(rows,keep) if k]
        scores = m.wins(values[keep])
        matches, ix = np.unique([r['match'] for r in records], return_inverse=True)
        sums = np.bincount(ix, weights=scores); counts = np.bincount(ix)
        draw = np.random.Generator(np.random.PCG64(20260927)).integers(0,len(matches),(2000,len(matches)))
        bs = sums[draw].sum(1)/counts[draw].sum(1)
        out['bootstrap_95_interval'] = np.quantile(bs,[.025,.975]).tolist()
        out['B4_lower_100th'] = float(np.sort(bs)[99])
        out['B4'] = 'PASS' if len(records)>=500 and len(matches)>=30 and scores.mean()>=.60 and out['B4_lower_100th']>.55 else 'FAIL_OR_INSUFFICIENT'
    return out


def self_test():
    raw=np.ones((20,196)); mask=np.ones_like(raw,dtype=bool)
    valid=np.ones(20,dtype=bool); reasons=np.zeros(20,dtype=np.uint16)
    for a,b in zip(boundary(raw,mask,valid,reasons),m.token_boundary(raw,mask,valid,'log_ratio')):
        np.testing.assert_array_equal(a,b)
    reasons[9]=1; valid[9]=False; raw[10:]=100
    values,_,usable=boundary(raw,mask,valid,reasons)
    assert usable[6] and usable[12] and not usable[8] and not usable[10]
    assert np.all(values==0), 'Cross-shot leakage'
    mask[6,:136]=False
    assert not boundary(raw,mask,valid,reasons)[2][6]
    np.testing.assert_array_equal(percentile(np.array([0.,1.,1.,3.]),np.array([-1.,0.,1.,2.,3.,4.])),np.array([0,0,1,3,3,4])/5)
    assert decision(.5999)=='STOP_E_CHAIN'
    assert decision(.60).startswith('INCONCLUSIVE') and decision(.61999).startswith('INCONCLUSIVE')
    assert decision(.62).startswith('WORTH')
    rng=np.random.default_rng(9)
    rows=[{'match':str(i%37)} for i in range(300)]
    vals=rng.integers(0,10,(300,6)); keep=rng.random(300)>.1
    assert summary(rows,vals,keep)==original_summary(rows,vals,keep)
    print('PASS: synthetic shot isolation, support, percentile ties, bootstrap parity, decision boundaries. No scientific inputs loaded.')


if __name__=='__main__':
    self_test()
