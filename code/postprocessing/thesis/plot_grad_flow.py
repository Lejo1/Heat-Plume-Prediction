"""Thesis figure: how the temperature-loss gradient reaches CNN1 (sparse thread along the streamlines).

Input: forPaper/figures/grad_flow/graddiag_blur{0,1}/grad_diag_stage2_start.npz, written by the
gradient diagnostic (grad_diag_only: true, grad_diag_save_arrays: true; start state BEST_V + BEST_T, pure
T loss, full adjoint). Panels on a zoomed crop: soft streamline channel, |dL/dsf| at CNN2's input (dense),
|dL/dv| through the streamlines for v_blur 0 and 1, |dL/dv| through CNN2's direct velocity channels
(reference, cut in training by detach_direct_v). Arrays are [w, h] (imshow origin lower), as in the diagnostic.

usage (from code/): python -m postprocessing.thesis.plot_grad_flow [--crop x0 x1 y0 y1]"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from postprocessing.thesis.common import FIGURES, save

SRC = FIGURES / "grad_flow"


def logmap(a, floor=1e-14):
    return np.log10(np.maximum(np.asarray(a, dtype=float), floor))


def main(crop):
    d0 = np.load(SRC / "graddiag_blur0" / "grad_diag_stage2_start.npz")
    d1 = np.load(SRC / "graddiag_blur1" / "grad_diag_stage2_start.npz")
    x0, x1, y0, y1 = crop
    c = lambda a: a[y0:y1, x0:x1]   # arrays are [w, h]: rows = second image axis

    stream = [d0["dLdv_streamline"], d1["dLdv_streamline"]]
    vmax = max(np.log10(s.max()) for s in stream)
    vmin = vmax - 8
    panels = [("soft streamline channel $s + s_o$", c(d0["streamlines"]), "magma", None, None),
              (r"$\log_{10}|\partial L/\partial s|$ at CNN$_2$'s input", c(logmap(d0["dLdsf"])), "magma", None, None),
              (r"$\log_{10}|\partial L/\partial v|$ via streamlines, $v_{blur}=0$", c(logmap(stream[0])), "magma", vmin, vmax),
              (r"$\log_{10}|\partial L/\partial v|$ via streamlines, $v_{blur}=1$", c(logmap(stream[1])), "magma", vmin, vmax),
              (r"$\log_{10}|\partial L/\partial v|$ via direct channels", c(logmap(d0["dLdv_direct"])), "viridis", None, None)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    axes = axes.ravel()
    im = axes[0].imshow(d0["streamlines"], origin="lower", cmap="magma", interpolation="nearest")
    axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=1.2))
    axes[0].set_title("full field (crop marked)", fontsize=9)
    for ax, (title, data, cmap, lo, hi) in zip(axes[1:], panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest",
                       extent=(x0, x1, y0, y1))
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel("cells"); ax.set_ylabel("cells")
    fig.tight_layout()
    save(fig, SRC, "grad_flow")

    for name, d in (("v_blur 0", d0), ("v_blur 1", d1)):
        s, on = d["dLdv_streamline"], d["streamlines"] > 0.05
        print(f"{name}: cells with dL/dv != 0 via streamlines {100 * (s > 0).mean():.1f} %, "
              f"mass on the lines {100 * s[on].sum() / s.sum():.1f} %, lines cover {100 * on.mean():.1f} % of the cells; "
              f"parameter-gradient norm streamline {float(d['param_grad_norm_streamline']):.3e}, "
              f"direct {float(d['param_grad_norm_direct']):.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crop", type=int, nargs=4, default=[900, 1500, 900, 1500], metavar=("x0", "x1", "y0", "y1"))
    main(p.parse_args().crop)
