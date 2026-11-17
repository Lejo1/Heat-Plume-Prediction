"""Thesis figure: signed T error of the start state with the predicted and the true streamlines (sec:m-line-losses).

Start state (BEST_V + BEST_T, soft drawing, no training) on the test datapoint, one crop: T - T_label with two
contours on top, the soft streamline channel s + s_o of the predicted lines (traced on CNN1's velocity, s_pred) and
of the true lines (traced on the simulated velocity with the same tracer, s_label), each at one level. Shows that
the error lies along both line sets, while the gradient of the T loss reaches only the predicted ones.

Input: E3_finetune/central/central_frozenT_r2/test_eval/test_fields_start.npz (eval_scaling_e2e.py --split test
--save_fields_start), crop presets and raw-cell crop as in result_fields.py.

usage (from code/): python -m postprocessing.thesis.line_error_overlay [--crop first_runs|soft_vs_hard|x0 x1 y0 y1]
                    [--level L] [--err_max E] [--out <dir>]"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from postprocessing.thesis.common import FIGURES, save
from postprocessing.thesis.result_fields import CROPS, ERR_PERCENTILE, FILES, crop_of

LINES = {"s_pred": ("predicted streamlines", "black"), "s_label": ("true streamlines", "#00a651")}


def main(crop, tag, level, err_max, out_dir):
    d = np.load(FILES["start"])
    T, label = crop_of(d["T_pred"], crop), crop_of(d["T_label"], crop)
    err = T - label
    e = err_max if err_max is not None else float(np.percentile(np.abs(err), ERR_PERCENTILE))
    print(f"crop {tag}: T MAE {np.abs(err).mean():.4f}, error scale +-{e:.3f} degC, contour level {level}")
    fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
    im = ax.imshow(err.T, origin="lower", cmap="PuOr_r", vmin=-e, vmax=e, interpolation="nearest")
    for key, (_, color) in LINES.items():
        ax.contour(crop_of(d[key], crop).T, levels=[level], colors=color, linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    over = err.max() > e, err.min() < -e
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=r"$T - T_\mathrm{label}$ [$^\circ$C]",
                 extend={(True, True): "both", (True, False): "max", (False, True): "min"}.get(over, "neither"))
    fig.legend(handles=[Line2D([], [], color=c, lw=1.2, label=t) for t, c in LINES.values()],
               loc="outside lower center", ncols=2, fontsize=9, frameon=False)
    save(fig, out_dir, f"line_error_overlay" + ("" if tag == "first_runs" else f"_{tag}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--crop", nargs="+", default=["first_runs"],
                   help=f"preset {list(CROPS)} or raw cells x0 x1 y0 y1 of the 2560^2 field (default first_runs)")
    p.add_argument("--level", type=float, default=0.05, help="contour level of the streamline channels s + s_o")
    p.add_argument("--err_max", type=float, default=None,
                   help=f"colour limit of the signed error (default: {ERR_PERCENTILE}th percentile of |error| on the crop)")
    p.add_argument("--out", type=Path, default=FIGURES / "result_fields")
    a = p.parse_args()
    if len(a.crop) == 1:
        crop, tag = CROPS[a.crop[0]], a.crop[0]
    else:
        crop = tuple(int(v) for v in a.crop)
        tag = "x{}-{}_y{}-{}".format(*crop)
    main(crop, tag, a.level, a.err_max, a.out)
