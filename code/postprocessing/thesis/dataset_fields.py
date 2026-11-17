"""Thesis figure: inputs and labels of the training simulation (setup chapter, Section Data, fig:s-dataset).

2 x 3 panels of the full 2560^2 training field (RUN_1 of dataset_giant_100hp_varyK, data 0):
 top    inputs of the surrogate: permeability log10 k, initial pressure p, heat-pump positions i
 bottom labels of the simulation: velocity v_x, v_y, temperature T
Fields are read from the prepared datasets (tensor order [h, w], Rescale-normalised: physical = x (max - min) + min
with the values of info.yaml), block-averaged by --coarsen (default 4, keeps the pdf small; the heat pumps are drawn
as dots at their cell positions) and plotted transposed with origin lower as the other thesis field figures.
The arrow in the pressure panel points down the mean pressure gradient (direction of the background flow).
T colour scale: label minimum to the T_PERCENTILE-th percentile (the pump cells reach 15.4 degC).

usage (from code/): python -m postprocessing.thesis.dataset_fields [--coarsen C] [--out <dir>]"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from postprocessing.thesis.common import CODE_DIR, FIGURES, save

PREP = (CODE_DIR / "../../datasets_prep").resolve()
PKI = PREP / "dataset_giant_100hp_varyK inputs_pki outputs_xy"          # inputs p, k, i; labels v_x, v_y
FOR_S = PREP / "dataset_giant_100hp_varyK inputs_ixyk outputs_t for_s"  # label T
RUN = "RUN_1.pt"     # training simulation (data 0)
T_PERCENTILE = 99.9


def rescale(t, entry):
    return t * (entry["max"] - entry["min"]) + entry["min"]


def coarse(a, c):
    n0, n1 = (a.shape[0] // c) * c, (a.shape[1] // c) * c
    return a[:n0, :n1].reshape(n0 // c, c, n1 // c, c).mean(axis=(1, 3))


def main(coarsen, out_dir):
    info = yaml.safe_load((PKI / "info.yaml").read_text())
    info_t = yaml.safe_load((FOR_S / "info.yaml").read_text())
    x = torch.load(PKI / "Inputs" / RUN, weights_only=False).numpy()
    v = torch.load(PKI / "Labels" / RUN, weights_only=False).numpy()
    T = torch.load(FOR_S / "Labels" / RUN, weights_only=False).numpy()[0]
    inp, lab = info["Inputs"], info["Labels"]
    get = lambda arr, group, key: rescale(arr[group[key]["index"]], group[key])
    p = get(x, inp, "Liquid Pressure [Pa]") / 1e6
    k = np.log10(get(x, inp, "Permeability X [m^2]"))
    pumps = np.argwhere(x[inp["Material ID"]["index"]] > 0.5)   # heat-pump cells (tensor indices)
    vx = get(v, lab, "Liquid X-Velocity [m_per_y]")
    vy = get(v, lab, "Liquid Y-Velocity [m_per_y]")
    T = rescale(T, info_t["Labels"]["Temperature [C]"])
    print(f"  {RUN}: {len(pumps)} heat-pump cells, p {p.min():.3f}-{p.max():.3f} MPa, "
          f"T {T.min():.2f}-{T.max():.2f} degC, v_x mean {vx.mean():.1f}, v_y mean {vy.mean():.1f} m/y")

    n = k.shape[0]
    t_lim = (float(T.min()), float(np.percentile(T, T_PERCENTILE)))
    m_vx = float(np.percentile(np.abs(vx), 99.5))
    panels = [(k, r"permeability $\log_{10} k$ [m$^2$]", "viridis", {}),
              (p, r"pressure $p$ [MPa]", "cividis", {}),
              (None, r"heat pumps $i$", None, {}),
              (vx, r"velocity $v_x$ [m/year]", "RdBu_r", {"vmin": -m_vx, "vmax": m_vx}),
              (vy, r"velocity $v_y$ [m/year]", "viridis", {}),
              (T, r"temperature $T$ [$^\circ$C]", "RdBu_r", {"vmin": t_lim[0], "vmax": t_lim[1]})]
    fig, axes = plt.subplots(2, 3, figsize=(2.9 * 3 + 1.2, 2.9 * 2 + 0.4), layout="constrained")
    for ax, (a, title, cmap, kw) in zip(axes.ravel(), panels):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        if a is None:   # first tensor axis = plot x (transposed, origin lower)
            ax.scatter(pumps[:, 0], pumps[:, 1], s=4, color="black")
            ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_aspect("equal")
            continue
        extent = (-0.5, n - 0.5, -0.5, n - 0.5)   # coarse arrays on the axes of the full field
        im = ax.imshow(coarse(a, coarsen).T, origin="lower", cmap=cmap, interpolation="nearest", extent=extent, **kw)
        ext = "max" if a is T else ("both" if a is vx else "neither")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend=ext)
    # background flow: down the mean pressure gradient (tensor axis 0 = plot x, axis 1 = plot y)
    g = np.array([np.gradient(p.mean(axis=1)).mean(), np.gradient(p.mean(axis=0)).mean()])
    d = -g / np.linalg.norm(g) * 0.3
    ax = axes[0, 1]
    ax.annotate("", xy=(0.5 + d[0], 0.5 + d[1]), xytext=(0.5 - d[0], 0.5 - d[1]), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=15))
    ax.text(0.5, 0.5 + 0.08, "flow", color="white", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    axes[0, 0].set_ylabel("inputs"); axes[1, 0].set_ylabel("labels")
    save(fig, out_dir, "dataset_fields")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coarsen", type=int, default=4, help="block size of the averaging (default 4)")
    ap.add_argument("--out", type=Path, default=FIGURES / "dataset_fields")
    a = ap.parse_args()
    main(a.coarsen, a.out)
