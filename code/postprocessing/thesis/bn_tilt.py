"""Thesis figure: streamlines of our early patch-trained CNN1 with the BatchNorm statistics of the patch training and
re-estimated on the training field, each on top of the true streamlines (appendix BatchNorm, fig:app-bn-tilt).

Two panels (rows) on one crop of the test field over the whole flow length: top the lines traced in CNN1's velocity
with the running statistics of the patch training, bottom after the re-estimation; in both the lines traced in the simulated velocity (true streamlines).
Panel titles give the mean-flow tilt of CNN1's velocity. Colours: true blue, CNN1 red (validated pair, CVD dE 19.6).

Input: forPaper/A_batchnorm/lines_<model>.pt (bn_tilt_lines.py on pcsgs11); coordinates in cells of CNN1's output
region (2372^2), plotted with the first axis horizontal as in result_fields.py.

usage (from code/): python -m postprocessing.thesis.bn_tilt [--model own|best] [--crop x0 x1 y0 y1] [--out <dir>]"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from postprocessing.thesis.common import FIGURES, FORPAPER, save

TRUE, CNN1 = "#2166ac", "#c0392b"
CROP = (0, 2372, 700, 1500)   # cells of CNN1's output region: first axis (flow direction, plot x) / second axis
TITLES = {"patch": "patch statistics", "reestimated": "re-estimated on the training field"}


def main(model, crop, out_dir):
    r = torch.load(FORPAPER / "A_batchnorm" / f"lines_{model}.pt", weights_only=False)
    x0, x1, y0, y1 = crop
    sim = r["simulated"]
    print(f"{model}: tilt simulated {sim['tilt_deg']:.2f} deg, " + ", ".join(
        f"{s} {r[s]['tilt_deg']:.2f} deg (vx MAE {r[s]['vx_mae']:.2f} m/y)" for s in TITLES))
    w = 6.0
    fig, axes = plt.subplots(2, 1, figsize=(w, 2 * w * (y1 - y0) / (x1 - x0) + 0.9), layout="constrained",
                             sharex=True, sharey=True)
    for ax, s in zip(axes, TITLES):
        for lines, c in ((sim["lines"], TRUE), (r[s]["lines"], CNN1)):
            for l in lines:
                ax.plot(l[0], l[1], color=c, lw=0.8, solid_capstyle="round")
        tilt = r[s]["tilt_deg"] - sim["tilt_deg"]
        ax.set_title(f"{TITLES[s]}: mean-flow tilt {tilt:+.1f}\u00b0 against the simulation", fontsize=9)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Line2D([], [], color=TRUE, lw=1.2, label="true streamlines"),
                        Line2D([], [], color=CNN1, lw=1.2, label="streamlines of CNN$_1$")],
               loc="outside lower center", ncols=2, fontsize=9, frameon=False)
    save(fig, out_dir, f"bn_tilt_{model}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["own", "best"], default="own")
    p.add_argument("--crop", nargs=4, type=int, default=CROP, help="cells of CNN1's output region x0 x1 y0 y1")
    p.add_argument("--out", type=Path, default=FIGURES / "batchnorm")
    a = p.parse_args()
    main(a.model, tuple(a.crop), a.out)
