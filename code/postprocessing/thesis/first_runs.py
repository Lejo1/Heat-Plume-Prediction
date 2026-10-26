"""Thesis figure: the first end-to-end run (CNN1 + CNN2 trained jointly from random initialization, no
pretraining of CNN1, pure T loss).

Input: forPaper/E3_finetune/first_runs/example_e2e_first/test_eval/test_fields.npz, written by
eval_scaling_e2e.py --split test --save_fields (current code with fade_mode per_line, which reproduces the
test T MAE logged by the July run). Rows: simulated, the run; columns: vx, vy,
soft streamline channel s + s_o, T, on one zoomed crop of the test field. Each column shares the colour
scale of the simulated field in the crop; predictions outside it are clipped (colour bar extensions).
Arrays are [h, w] in tensor order, plotted transposed with origin lower as in visualize_e2e.

usage (from code/): python -m postprocessing.thesis.first_runs [--crop x0 x1 y0 y1]"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.thesis.common import FIGURES, FORPAPER, save

SRC = FORPAPER / "E3_finetune" / "first_runs"
RUNS = {"without pretraining": "example_e2e_first"}
# (field key without suffix, column title, colormap)
COLUMNS = [("vx", r"$v_x$ [m/y]", "viridis"), ("vy", r"$v_y$ [m/y]", "viridis"),
           ("s", r"streamlines $s + s_o$", "magma"), ("T", r"$T$ [$^\circ$C]", "RdBu_r")]


def main(crop):
    x0, x1, y0, y1 = crop
    c = lambda a: a[x0:x1, y0:y1].astype(np.float32).T   # tensor [h, w] -> image rows = second axis
    data = {name: np.load(SRC / run / "test_eval" / "test_fields.npz") for name, run in RUNS.items()}
    sim = next(iter(data.values()))   # the label fields are the same in both files
    rows = [("simulated", {k: c(sim[f"{k}_label"]) for k, _, _ in COLUMNS})]
    rows += [(name, {k: c(d[f"{k}_pred"]) for k, _, _ in COLUMNS}) for name, d in data.items()]

    fig, axes = plt.subplots(len(rows), len(COLUMNS), figsize=(12, 3.3 * len(rows)), layout="constrained")
    for j, (key, title, cmap) in enumerate(COLUMNS):
        ref = rows[0][1][key]
        vmin, vmax = (0.0, 1.0) if key == "s" else (float(ref.min()), float(ref.max()))
        for i, (row_name, fields) in enumerate(rows):
            ax = axes[i, j]
            im = ax.imshow(fields[key], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(row_name)
        lo = min(f[key].min() for _, f in rows) < vmin
        hi = max(f[key].max() for _, f in rows) > vmax
        fig.colorbar(im, ax=axes[:, j], orientation="horizontal", shrink=0.9,
                     extend={(True, True): "both", (True, False): "min", (False, True): "max"}.get((lo, hi), "neither"))
    save(fig, FIGURES / "first_runs", "first_runs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--crop", type=int, nargs=4, default=[900, 1500, 1200, 1800], metavar=("x0", "x1", "y0", "y1"),
                   help="cells along the first / second array axis (plot x / y), default 900 1500 1200 1800")
    main(p.parse_args().crop)
