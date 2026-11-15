"""Thesis figure: T result fields on the scaling field giant_double_100hp_v2 (5120^2; sec:res-scaling, runs list 11).

Columns label T | Pelzer's chain (BEST_V -> hard lines -> BEST_T) | soft chain, untrained (start state of
finetune_e2e_frame_3: BEST_V + BEST_T, soft drawing) | joint + true-line losses (finetune_e2e_frame_3);
rows T prediction, signed T error (prediction - label); one colour bar per row. Same layout as result_fields.py.

Inputs (npz with T_pred / T_label [degC], tensor order [h, w] = centred T output region of the 5120^2 field, 4787^2;
plotted transposed with origin lower):
 chain   E1_baseline/metrics/scaling_fields_BEST_predict_T_add_s_outer <scaling chain prep>.npz   eval_metrics.py --scaling --fields_only
 frame   E4_lines/frame/finetune_e2e_frame_3/scaling_test/scaling_fields.npz   eval_scaling_e2e.py --save_fields --fields_only
 start   the same folder, scaling_fields_start.npz                              eval_scaling_e2e.py --save_fields_start --fields_only
The labels of all files are checked to agree.

Views (--view): full = the whole output region, block-averaged by --coarsen (default 4, keeps the pdf small);
centre = the central 1200^2 cells at full resolution; or raw cells x0 x1 y0 y1 of the 5120^2 field.
Colour scales: T from the label minimum to the T_PERCENTILE-th percentile of the label on the view (the label is
below 11.1 degC on 99 % of the field, its maximum 15.6 degC lies at the pumps), error symmetric at the
ERR_PERCENTILE-th percentile of |error| over all configurations.

usage (from code/): python -m postprocessing.thesis.scaling_fields [--view full|centre|x0 x1 y0 y1] [--coarsen C]
                    [--T_max T] [--err_max E] [--out <dir>]"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.thesis.common import FIGURES, FORPAPER, save
from postprocessing.thesis.result_fields import ERR_PERCENTILE, ROWS

FIELD = 5120   # raw cells of the scaling field per axis
SCALING_PREP = "giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t prep_with_BEST_predict_v_v4 RK45"
FRAME3 = FORPAPER / "E4_lines" / "frame" / "finetune_e2e_frame_3" / "scaling_test"
FRAME3_A = FORPAPER / "E4_lines" / "frame" / "finetune_e2e_frame_3_ruleA" / "scaling_test"   # statistics follow the weights (2026-11-07)
FILES = {"chain": FORPAPER / "E1_baseline" / "metrics" / f"scaling_fields_BEST_predict_T_add_s_outer {SCALING_PREP}.npz",
         "start": FRAME3 / "scaling_fields_start.npz",   # start state, independent of freeze_bn
         "frame": FRAME3_A / "scaling_fields.npz"}
COLS = [("chain", "Pelzer's chain"), ("start", "soft chain, untrained"), ("frame", "joint + true-line losses")]
CENTRE = 1200          # edge of the centre view [cells]
T_PERCENTILE = 99.9    # upper end of the T scale: this percentile of the label on the view


def view_of(a, view, coarsen):
    """The view of a centred [h, w] output array: raw-cell crop (x0, x1, y0, y1) or the whole array block-averaged."""
    if view is None:
        n0, n1 = (a.shape[0] // coarsen) * coarsen, (a.shape[1] // coarsen) * coarsen
        return a[:n0, :n1].reshape(n0 // coarsen, coarsen, n1 // coarsen, coarsen).mean(axis=(1, 3))
    x0, x1, y0, y1 = view
    o = (FIELD - a.shape[0]) // 2, (FIELD - a.shape[1]) // 2
    assert x0 - o[0] >= 0 and y0 - o[1] >= 0 and x1 - o[0] <= a.shape[0] and y1 - o[1] <= a.shape[1], \
        f"view {view} outside the output region {a.shape} (origin {o})"
    return a[x0 - o[0]:x1 - o[0], y0 - o[1]:y1 - o[1]]


def load(view, coarsen):
    out, label = {}, None
    for name, f in FILES.items():
        d = np.load(f)
        T, lab = d["T_pred"].astype(np.float32), d["T_label"].astype(np.float32)
        print(f"  {name:6s} {T.shape[0]}^2  scaling T MAE {np.abs(T - lab).mean():.4f}")
        if label is None:
            label = lab
        else:
            dev = np.abs(lab - label).max()
            assert dev < 0.02, f"label of {name} deviates by {dev:.3g} degC from the reference"
        out[name] = {"T": view_of(T, view, coarsen), "err": view_of(T - lab, view, coarsen)}
    return out, view_of(label, view, coarsen)


def main(view, tag, coarsen, T_max, err_max, out_dir):
    fields, label = load(view, coarsen)
    e = err_max if err_max is not None else float(np.percentile(
        np.abs(np.concatenate([fields[c]["err"].ravel() for c, _ in COLS])), ERR_PERCENTILE))
    t1 = T_max if T_max is not None else float(np.percentile(label, T_PERCENTILE))
    scales = {"T": (float(label.min()), t1), "err": (-e, e)}
    print(f"view {tag} ({label.shape[0]}x{label.shape[1]} px): T [{scales['T'][0]:.3f}, {t1:.3f}], err +-{e:.3f}")
    n_c = len(COLS) + 1
    fig, axes = plt.subplots(2, n_c, figsize=(2.9 * n_c + 0.8, 2.9 * 2), layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.01)
    for i, r in enumerate(["T", "err"]):
        title, cmap = ROWS[r]
        vmin, vmax = scales[r]
        panels = [label if r == "T" else None] + [fields[c][r] for c, _ in COLS]
        for j, a in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if a is None:
                ax.set_frame_on(False)   # no label error
                continue
            im = ax.imshow(a.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[i, 0].set_ylabel(title)
        lo = min(a.min() for a in panels if a is not None) < vmin
        hi = max(a.max() for a in panels if a is not None) > vmax
        fig.colorbar(im, ax=axes[i, :], shrink=0.9, pad=0.01,
                     extend={(True, True): "both", (True, False): "min", (False, True): "max"}.get((lo, hi), "neither"))
    for j, t in enumerate([r"label $T$"] + [t for _, t in COLS]):
        axes[0, j].set_title(t)
    save(fig, out_dir, f"scaling_{tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--view", nargs="+", default=["full"], help="full, centre or raw cells x0 x1 y0 y1 of the 5120^2 field")
    p.add_argument("--coarsen", type=int, default=4, help="block size of the averaging in the full view")
    p.add_argument("--T_max", type=float, default=None,
                   help=f"upper limit of the T scale (default: {T_PERCENTILE}th percentile of the label on the view)")
    p.add_argument("--err_max", type=float, default=None,
                   help=f"colour limit of the signed error (default: {ERR_PERCENTILE}th percentile of |error| over both)")
    p.add_argument("--out", type=Path, default=FIGURES / "result_fields")
    a = p.parse_args()
    if a.view == ["full"]:
        view, tag = None, "full"
    elif a.view == ["centre"]:
        c0 = (FIELD - CENTRE) // 2
        view, tag = (c0, c0 + CENTRE, c0, c0 + CENTRE), "centre"
    else:
        view = tuple(int(v) for v in a.view)
        tag = "x{}-{}_y{}-{}".format(*view)
    main(view, tag, a.coarsen, a.T_max, a.err_max, a.out)
