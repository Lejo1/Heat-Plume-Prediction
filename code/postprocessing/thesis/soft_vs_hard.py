"""Hard vs soft streamline channel as CNN2 input (results section "Soft streamline channel").

Both chains use the paper models BEST_V -> BEST_T on the test datapoint without any training:
 hard: Pelzer's chain, one-cell lines from BEST_V's velocity (prep "... prep_with_BEST_predict_v_v4 RK45")
 soft: the e2e model's start state (soft drawing, sigma and fading of the run config, here F_frozenV)
Writes the channel statistics (coverage, mean on the covered cells, max, total mass) to
tables/soft_vs_hard_channel.csv and a zoomed figure (channel, T prediction, T error for both chains)
to figures/soft_vs_hard/. Uses export_figures' field functions; full-domain forwards (GPU if available).

usage (from code/): python -m postprocessing.thesis.soft_vs_hard [--crop r0 r1 c0 c1] [--run <e2e run>]"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.thesis.common import CODE_DIR, FIGURES, FORPAPER, TABLES, save

SOFT_THRESHOLD = 0.01   # a soft cell counts as covered above this value (hard cells: > 0)


def stats(a, threshold):
    a = np.asarray(a, dtype=float)
    on = a > threshold
    return {"coverage [%]": 100 * on.mean(), "mean on covered": a[on].mean() if on.any() else 0.0,
            "max": a.max(), "total mass": a.sum()}


def main(crop, run):
    os.chdir(CODE_DIR.parent)   # export_figures resolves its paths from the repository root
    import export_figures as ef
    hard = ef.fields_two_stage(ef.resolve("../models/BEST_predict_T_add_s_outer"),
                               ef.ensure_prep(ef.resolve("../models/BEST_predict_v_v4")))
    soft = ef.fields_e2e(ef.resolve(run), at_start=True)
    arr = lambda dv: np.asarray(dv.data, dtype=float)

    channels = {"hard sf": (arr(hard["Streamlines Faded [-]"]), 0.0),
                "hard sf_outer": (arr(hard["Streamlines Faded Outer [-]"]), 0.0),
                "soft sf": (arr(soft["Streamlines Fade"]), SOFT_THRESHOLD),
                "soft sf_outer": (arr(soft["Streamlines"]) - arr(soft["Streamlines Fade"]), SOFT_THRESHOLD)}
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "soft_vs_hard_channel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "threshold", "coverage [%]", "mean on covered", "max", "total mass"])
        for name, (a, thr) in channels.items():
            s = stats(a, thr)
            w.writerow([name, thr] + [f"{v:.4g}" for v in s.values()])
            print(f"{name:14s} " + "  ".join(f"{k} {v:.4g}" for k, v in s.items()))
    print(f"  -> {TABLES / 'soft_vs_hard_channel.csv'}")

    r0, r1, c0, c1 = crop
    T = lambda d, k: arr(d[f"Temperature [C]_{k}"])
    sf_h, sf_s = arr(hard["Streamlines Faded [-]"]), arr(soft["Streamlines"])
    off = [(sf_h.shape[i] - T(hard, "out").shape[i]) // 2 for i in (0, 1)]   # channel -> T output crop
    ct = lambda a: a[r0 - off[0]:r1 - off[0], c0 - off[1]:c1 - off[1]]
    cs = lambda a: a[r0:r1, c0:c1]
    t_lim = (min(ct(T(hard, "out")).min(), ct(T(soft, "out")).min()), max(ct(T(hard, "out")).max(), ct(T(soft, "out")).max()))
    e_max = max(ct(T(hard, "error")).max(), ct(T(soft, "error")).max())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, (name, d, sf) in enumerate((("Pelzer's chain", hard, sf_h), ("soft chain, untrained", soft, sf_s))):
        for ax, (title, data, cmap, lo, hi) in zip(axes[row], (
                (f"{name}: streamline channel", cs(sf), "magma", 0, None),
                (f"{name}: T prediction [degC]", ct(T(d, "out")), "RdBu_r", *t_lim),
                (f"{name}: |T error| [degC]", ct(T(d, "error")), "magma", 0, e_max))):
            im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, FIGURES / "soft_vs_hard", "soft_vs_hard_crop")
    for name, d in (("hard", hard), ("soft", soft)):
        print(f"{name}: test T MAE full field {T(d, 'error').mean():.4f}, crop {ct(T(d, 'error')).mean():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crop", type=int, nargs=4, default=[900, 1400, 900, 1400], metavar=("r0", "r1", "c0", "c1"),
                   help="rows / columns of the streamline-channel array")
    p.add_argument("--run", default=str(FORPAPER / "E3_finetune/central/central_frozenT_r1"),
                   help="e2e run whose start state (model_v + model_T) gives the soft chain")
    a = p.parse_args()
    main(a.crop, a.run)
