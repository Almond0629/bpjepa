#!/usr/bin/env python3
"""Fig. 2: change magnitude (raw C) versus calibrated regime evidence on one sequence. Defaults are the published example.

Needs the evidence directory (measurements, raw change, state keys, references, targets) and frames. Displayed seconds are
s-4, s-2, s and s+2. Each column shows the transition that sets y_t for that second, on the frame it ends at.
"""
import json
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from bpjepa.evidence import pipeline as E
from bpjepa.evidence.regime import boundary_half, load_references, topk_mask

W, H, G = 398, 224, 14
OFFSETS = (-4, -2, 0, 2)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42})
GRAY = '#c3c2b7'


def clock(t):
    return f'{int(t // 60):d}:{t % 60:04.1f}'


def tokmap(ax, img, vals, ok, cmap, norm, top=None):
    ax.imshow(np.asarray(Image.fromarray(img).convert('L')), cmap='gray', extent=(0, W, H, 0), alpha=0.35)
    rgba = cmap(norm(np.where(ok, vals, 0).reshape(G, G)))
    rgba[..., 3] = np.where(ok, 0.88, 0).reshape(G, G)
    ax.imshow(rgba, extent=(0, W, H, 0), interpolation='nearest')
    if top is not None:
        cw, ch = W / G, H / G
        for k in np.flatnonzero(top):
            r, c = divmod(int(k), G)
            ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=False, ec='#2a78d6', lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)


def per_second(score, reliable):
    """y for second s = max over supported transitions 2s and 2s+1."""
    n = len(score) // 2
    y, src = np.full(n, np.nan), np.full(n, -1)
    for s in range(n):
        ok = [x for x in (2 * s, 2 * s + 1) if x < len(score) and reliable[x]]
        if ok:
            b = max(ok, key=lambda x: (score[x], -x))
            y[s], src[s] = score[b], b
    return y, src


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--frame-root', type=Path, required=True)
    ap.add_argument('--roles', type=Path, required=True, help='configs/evidence/roles.json')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--name', default='fig2_evidence')
    ap.add_argument('--half', default='spain_laliga/2016-2017/2017-01-08 - 22-45 Villarreal 1 - 1 Barcelona::1')
    ap.add_argument('--second', type=int, default=58)
    ap.add_argument('--event-time', type=float, default=58.668)
    ap.add_argument('--label', default='Ball out of play')
    a = ap.parse_args()
    half, s, te, name = a.half, a.second, a.event_time, a.name
    sel = {'half': half, 'event_time': te, 'label': a.label, 'second': s}
    OUT = a.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    roles = json.loads(a.roles.read_text())
    if half in set(roles['calibration_half_ids']):
        raise ValueError('choose an audit half: calibration halves use cross-fitted references')
    raw, mask, valid, reasons, keys = E.load_half_for_regime(a.evidence_dir, half)
    score, reliable, dense, support = boundary_half(raw, mask, valid, reasons, keys, load_references(a.evidence_dir / 'references/full'))
    y, src = per_second(score, reliable)
    match, h = half.rsplit('::', 1)
    frames = a.frame_root / match / f'{h}_224p'
    cols = []
    for off in OFFSETS:
        j = int(src[s + off])
        assert j >= 0
        f = j + 1
        img = np.asarray(Image.open(frames / f'{f + 1:08d}.jpg').convert('RGB'))
        assert img.shape == (H, W, 3)
        top = topk_mask(dense[j], support[j])
        assert np.isclose(dense[j][top].mean(), score[j])
        cols.append(dict(sec=s + off, j=j, f=f, img=img, c=raw[j], cok=mask[j] & np.isfinite(raw[j]),
                         r=dense[j], rok=support[j], top=top))
    allc = np.concatenate([c['c'][c['cok']] for c in cols])
    cmax = math.ceil(np.percentile(allc, 99) / 0.05) * 0.05
    cC, nC = plt.get_cmap('viridis'), Normalize(0, cmax)
    cR, nR = plt.get_cmap('magma'), Normalize(0, 1)

    fig = plt.figure(figsize=(6.875, 2.45))
    gl = fig.add_gridspec(3, 4, left=0.075, right=0.575, top=0.9, bottom=0.03, hspace=0.12, wspace=0.04)
    for i, c in enumerate(cols):
        a0, a1, a2 = (fig.add_subplot(gl[r, i]) for r in range(3))
        a0.imshow(c['img'], extent=(0, W, H, 0)); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(clock(c['f'] / 2), fontsize=7.5, pad=2)
        tokmap(a1, c['img'], c['c'], c['cok'], cC, nC)
        tokmap(a2, c['img'], c['r'], c['rok'], cR, nR, c['top'])
        if i == 0:
            for a, t in ((a0, 'frame'), (a1, 'raw $C$'), (a2, 'regime\nevidence')):
                a.set_ylabel(t, fontsize=7.5, labelpad=2)
    for ypos, (cm, nm, lab) in zip((0.36, 0.05), ((cC, nC, 'raw $C$'), (cR, nR, 'evidence'))):
        cax = fig.add_axes([0.585, ypos + 0.02, 0.008, 0.24])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax)
        cb.set_ticks([0, nm.vmax / 2, nm.vmax]); cb.ax.set_yticklabels([f'{v:g}' for v in (0, nm.vmax / 2, nm.vmax)])
        cb.ax.tick_params(labelsize=6, pad=1.5); cb.set_label(lab, fontsize=6.5, labelpad=3)
    ax = fig.add_axes([0.76, 0.2, 0.215, 0.62])
    lo, hi = s - 8, s + 9
    t = np.arange(lo, hi)
    ax.plot(t + 0.5, y[lo:hi], color='#2a78d6', lw=1.2, marker='o', ms=2)
    for c in cols:
        ax.axvspan(c['sec'], c['sec'] + 1, color='#e1e0d9', lw=0, zorder=0)
    ax.axvline(te, color='#0b0b0b', lw=0.8, ls='--')
    ax.text(te, 1.03, sel['label'], fontsize=7, ha='center', va='bottom', transform=ax.get_xaxis_transform())
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
    ticks = list(range(lo, hi + 1, 4))
    ax.set_xticks(ticks); ax.set_xticklabels([clock(v) for v in ticks], fontsize=6.5)
    ax.set_ylabel('$y_t$', fontsize=8, labelpad=1); ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02); fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    (OUT / f'{name}.json').write_text(json.dumps(dict(selection=sel, offsets=OFFSETS, raw_c_scale_max=cmax, columns=[
        dict(second=c['sec'], transition=c['j'], frame_file=f'{c["f"] + 1:08d}.jpg', frame_time_s=c['f'] / 2,
             y=float(y[c['sec']]), raw_c_median=float(np.median(c['c'][c['cok']])),
             raw_c_p95=float(np.percentile(c['c'][c['cok']], 95)), evidence_median=float(np.median(c['r'][c['rok']])),
             top_rows=sorted(int(q) // G for q in np.flatnonzero(c['top']))) for c in cols]), indent=1))


if __name__ == '__main__':
    main()
