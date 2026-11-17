#!/usr/bin/env python3
"""Is the SOFT DRAWING well behaved when a streamline moves sideways?

Motivation (Jonas's question): if a change in v shifts a streamline a little to the left, a pixel
it passes goes small -> large -> small again. Does the drawn value jump? Does the gradient jump
between zero and non-zero?

Method: take one line, slide it laterally by `delta` in sub-cell steps, and record the drawn value,
the loss and dL/d(delta) as functions of delta. The line is a gentle sinusoid, so its samples cross
cell boundaries at STAGGERED delta - the realistic case. (A perfectly axis-aligned line makes all
samples cross at once and exaggerates every discontinuity by roughly an order of magnitude.)

Two losses, because they behave differently:
  target : L = ||occ(delta) - occ(0)||^2   sharp target, the worst case; used for the capture radius
  smooth : L = sum(occ * G), G a smooth low-frequency field. This is what training actually sees -
           the gradient arriving at sf comes from CNN2 and is a smooth field (see the
           'log10 |dL/dsf|' panel of grad_diag), not a delta.

What it finds (see the summary at the end of a run):
  * the VALUE is clean - amplitude varies by ~0.2% as the line slides across a cell;
  * the GRADIENT is a staircase, jumping by up to ~45% of its own peak per 0.02 cell;
  * there are TWO separate causes, needing different fixes:
      - splat path: the kernel is truncated at +-2*sigma, where a Gaussian is still 13.5% of peak.
        Widening the window to +-4*sigma removes it (p99 jump 45% -> 0.2%).
      - blur path: `_scatter_bilinear` uses piecewise-LINEAR weights, so their derivative is
        piecewise CONSTANT and jumps at every cell boundary. The Gaussian afterwards smooths the
        field in space but not its dependence on sample position, so no window helps. Fixing this
        needs a higher-order (B-spline) deposition kernel.
  * the capture radius is ~4*sigma: at sigma=1 a line displaced by more than ~4-5 cells feels
    almost no restoring gradient.

Usage
    python check_drawing_gradient.py                 # all four checks
    python check_drawing_gradient.py --plot out.png
    python check_drawing_gradient.py --quick
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step2_streamlines.streamlines_helpers import draw_streamlines_soft, SIGMA_BLUR_MIN

N = 96             # grid
NS = 400           # samples along the line
T_END = 27.5
ROW = 48.0         # the line's mean lateral position
DT = torch.float64  # so finite differences are meaningful


def draw(delta, sigma, method="auto", window=None, straight=False):
    """One line at lateral offset `delta`, drawn through the production code path."""
    sy = torch.linspace(6.0, N - 7.0, NS, dtype=DT)
    wig = torch.zeros(NS, dtype=DT) if straight else 3.0 * torch.sin(2 * np.pi * sy / 40.0)
    sx = ROW + wig + delta
    t = torch.linspace(0.0, T_END * 0.6, NS, dtype=DT)
    return draw_streamlines_soft([(sx, sy, t)], (N, N), faded=True, sigma=sigma, window=window,
                                 fade_mode="absolute", t_end=T_END, method=method)


def smooth_field():
    ii, jj = torch.meshgrid(torch.arange(N, dtype=DT), torch.arange(N, dtype=DT), indexing="ij")
    return torch.sin(2 * np.pi * ii / N * 1.5) * torch.cos(2 * np.pi * jj / N * 1.5) + 1.5


def sweep(deltas, sigma, mode, method="auto", window=None, straight=False):
    """L(delta) and dL/d(delta) over the sweep."""
    tgt = draw(torch.tensor(0.0, dtype=DT), sigma, method, window, straight).detach()
    G = smooth_field()
    L, dL = [], []
    for d in deltas:
        dd = torch.tensor(float(d), dtype=DT, requires_grad=True)
        occ = draw(dd, sigma, method, window, straight)
        loss = ((occ - tgt) ** 2).sum() if mode == "target" else (occ * G).sum()
        L.append(float(loss))
        dL.append(float(torch.autograd.grad(loss, dd)[0]))
    return np.array(L), np.array(dL)


def roughness(dL):
    """Step-to-step change in the gradient, as a fraction of its own peak."""
    pk = max(np.abs(dL).max(), 1e-30)
    j = np.abs(np.diff(dL)) / pk
    return np.median(j), np.percentile(j, 99)


def banner(s):
    print("\n" + "=" * 94 + f"\n{s}\n" + "=" * 94)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    step = 0.05 if a.quick else 0.02
    deltas = np.arange(-4.0, 4.0 + 1e-9, step)
    sigmas = (0.7, 1.0, 2.0) if a.quick else (0.7, 1.0, 2.0, 4.0)

    print(f"grid {N}x{N}, one sinusoidal line of {NS} samples, delta swept in {step}-cell steps")
    print(f"current SIGMA_BLUR_MIN = {SIGMA_BLUR_MIN} (sigma >= this uses the blur path)")

    # ---------------------------------------------------------------- 1. the value
    banner("1  IS THE DRAWN VALUE CONTINUOUS AS THE LINE SLIDES?")
    fine = np.arange(-2.0, 2.0 + 1e-9, 0.01)
    for s in sigmas[:3]:
        v = np.array([float(draw(torch.tensor(float(d), dtype=DT), s)[int(ROW), N // 2])
                      for d in fine])
        tot = np.array([float(draw(torch.tensor(float(d), dtype=DT), s).sum())
                        for d in np.arange(0, 1.0, 0.05)])
        print(f"  sigma={s:4.1f}: value at one cell - peak {v.max():.4f}, largest change per "
              f"0.01 cell {100*np.abs(np.diff(v)).max()/v.max():5.2f}% of peak")
        print(f"  {'':11s} total drawn occupancy over one cell of shift varies by "
              f"{100*(tot.max()-tot.min())/tot.mean():.2f}%")
    print("  -> the value is smooth; the rise-and-fall as the line passes a pixel is the kernel")
    print("     profile doing its job, not an artefact.")

    # ---------------------------------------------------------------- 2. the gradient
    banner("2  IS THE GRADIENT SMOOTH?  (loss = smooth upstream field, as in training)")
    print(f"  {'sigma':>6s} {'path':>6s} {'line':>8s}   {'median':>8s} {'p99':>8s}   "
          f"(change in dL/d(delta) per {step} cell, % of its peak)")
    for s in sigmas:
        for straight, tag in ((False, "wiggly"), (True, "aligned")):
            _, dL = sweep(deltas, s, "smooth", straight=straight)
            m, p = roughness(dL)
            path = "blur" if s >= SIGMA_BLUR_MIN else "splat"
            print(f"  {s:6.1f} {path:>6s} {tag:>8s}   {100*m:7.2f}% {100*p:7.1f}%")
    print("  The 'aligned' rows are a degenerate case (all samples cross a cell boundary at the")
    print("  same delta) and are shown only to bound the effect; 'wiggly' is the realistic one.")

    # ---------------------------------------------------------------- 3. the window
    banner("3  WHERE THE STAIRCASE COMES FROM: KERNEL TRUNCATION vs BILINEAR DEPOSITION")
    print("  The default window is 2*ceil(2*sigma)+1, i.e. the Gaussian is cut at +-2 sigma where")
    print("  it is still 13.5% of its peak. Widening it separates the two mechanisms:\n")
    print(f"  {'window':>7s} {'half/sigma':>11s} {'kernel cut at':>14s}   "
          f"{'splat med':>10s} {'splat p99':>10s}   {'blur med':>9s} {'blur p99':>9s}")
    win_rows = []
    for w in ((5, 7, 9) if a.quick else (5, 7, 9, 13, 17)):
        half = w // 2
        cut = float(np.exp(-half ** 2 / 2.0))          # Gaussian at the cut, sigma = 1
        _, ds_ = sweep(deltas, 1.0, "smooth", method="splat", window=w)
        _, db_ = sweep(deltas, 1.0, "smooth", method="blur", window=w)
        ms, ps = roughness(ds_)
        mb, pb = roughness(db_)
        win_rows.append((w, ps, pb))
        print(f"  {w:7d} {half:10.1f}s {100*cut:13.2f}%   {100*ms:9.2f}% {100*ps:9.1f}%   "
              f"{100*mb:8.2f}% {100*pb:8.1f}%")
    print("\n  splat: widening the window removes the staircase almost entirely - its cause IS the")
    print("         truncation, and the exact sub-cell Gaussian underneath is smooth.")
    print("  blur : unchanged at every window. Its cause is _scatter_bilinear: piecewise-linear")
    print("         weights -> piecewise-constant derivative, jumping at each cell boundary. The")
    print("         Gaussian afterwards smooths the field in space, not its dependence on position.")

    # how much does a wider window change the FORWARD value? (existing runs used +-2 sigma)
    ref = draw(torch.tensor(0.3, dtype=DT), 1.0, "splat", 5).detach()
    wide = draw(torch.tensor(0.3, dtype=DT), 1.0, "splat", 9).detach()
    print(f"\n  forward change from window 5 -> 9 at sigma=1: total occupancy "
          f"{float(ref.sum()):.3f} -> {float(wide.sum()):.3f} "
          f"({100*(float(wide.sum())/float(ref.sum())-1):+.2f}%), max cell "
          f"{float((wide-ref).abs().max()):.4f}")
    print("  (the wider window is the LESS truncated, more faithful Gaussian - but it is a change,")
    print("   so a channel drawn this way differs slightly from what earlier runs saw.)")

    # ---------------------------------------------------------------- 4. capture radius
    banner("4  CAPTURE RADIUS: how far can a line be displaced and still feel a gradient?")
    print(f"  {'sigma':>6s} {'|dL|max':>10s}   |dL| as a fraction of peak at a displacement of")
    print(f"  {'':6s} {'':10s}   {'0.5c':>7s} {'1c':>7s} {'2c':>7s} {'4c':>7s} {'6c':>7s}")
    # a wider sweep than the rest: the whole point is to reach displacements where the signal dies
    far = np.arange(-10.0, 10.0 + 1e-9, step * 2)
    cap = {}
    for s in sigmas:
        L, dL = sweep(far, s, "target")
        pk = np.abs(dL).max()
        vals = [np.abs(dL[np.argmin(np.abs(far - d))]) / pk for d in (0.5, 1, 2, 4, 6, 8)]
        cap[s] = (far, dL / pk)
        print(f"  {s:6.1f} {pk:10.3e}   " + " ".join(f"{v:6.1%}" for v in vals))
    print("  -> the radius scales like ~4*sigma. At the production sigma=1.0 a streamline that is")
    print("     more than ~4-5 cells (20-25 m) from where it should be gets almost no signal.")

    # ---------------------------------------------------------------- plot
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        d2 = np.arange(-2.0, 2.0 + 1e-9, step)
        for w, c, lab in ((5, "tab:red", "window 5 (±2σ, current default)"),
                          (9, "tab:blue", "window 9 (±4σ)")):
            _, g = sweep(d2, 1.0, "smooth", method="splat", window=w)
            ax[0].plot(d2, g / np.abs(g).max(), c, lw=1.2, label=lab)
        ax[0].set_xlabel("lateral offset δ [cells]"); ax[0].set_ylabel("dL/dδ  (normalised)")
        ax[0].set_title("splat path, σ=1.0:\ntruncation is the staircase"); ax[0].legend(fontsize=8)

        ws = [r[0] for r in win_rows]
        ax[1].semilogy(ws, [100 * r[1] for r in win_rows], "o-", label="splat")
        ax[1].semilogy(ws, [100 * r[2] for r in win_rows], "s-", label="blur")
        ax[1].set_xlabel("window [cells]"); ax[1].set_ylabel("p99 jump in dL/dδ  [% of peak]")
        ax[1].set_title("σ=1.0: widening the window fixes\nsplat, never blur"); ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

        for s in sigmas:
            dd, gg = cap[s]
            ax[2].plot(dd, np.abs(gg), lw=1.2, label=f"σ={s}")
        ax[2].set_xlabel("displacement from the target [cells]")
        ax[2].set_ylabel("|dL/dδ| / peak")
        ax[2].set_title("capture radius ≈ 4σ"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"\nfigure written to {a.plot}")


if __name__ == "__main__":
    main()
