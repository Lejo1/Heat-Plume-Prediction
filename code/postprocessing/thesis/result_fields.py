"""Thesis figures: T result fields on one crop of the test datapoint (results chapter, runs list 9).

The figures, all on the same crop with the same colour scales:
 baselines     label T | Pelzer's chain (BEST_V -> hard lines -> BEST_T) | isolated step 3 (BEST_T on the hard
               lines of the simulated velocity) | sequential CNN2 (on BEST_V's hard lines);
               rows T prediction, signed T error                                      (fig:res-baseline-fields)
 soft_vs_hard  label T | Pelzer's chain (hard) | soft chain, untrained (start state: BEST_V + BEST_T, soft drawing, no training);
               rows streamline channel s + s_o, T prediction, signed T error           (fig:res-soft-vs-hard)
 central       label T | start state | CNN1 only (central_frozenT_r2_ruleA) | joint (central_joint_r3_ruleA);
               rows T prediction, signed T error, difference to the start              (fig:res-central-fields)
 trajectory    label T | start state | trajectory loss (traj_lr1e-5);
               rows T prediction, signed T error, difference to the start              (fig:res-trajectory-fields)
 frame         label T | start state | lambda_N 1 (finetune_e2e_frame_3) | CNN2 alone (F_frozenV) | CNN2 on the
               lines of CNN1 alone (A2_lamN1V_frozenV); rows T prediction, signed T error, difference to the start
                                                                                      (fig:res-frame-fields)
The start state (BEST_V + BEST_T, soft drawing, no training) is titled "soft chain, untrained". Finetuned runs: the
reruns in which the BatchNorm statistics follow the weights (*_ruleA, 2026-11-07; CNN2 alone = F_frozenV, which already
had this setting); central / central_joint / CNN1 of A2 accordingly.
Signed error = prediction - label (positive: too warm). Configurations whose field file is missing are skipped.

Inputs (npz with T_pred / T_label [degC], optionally s_pred; tensor order [h, w], plotted transposed with
origin lower as in first_runs.py / soft_vs_hard.py):
 e2e runs    <run>/test_eval/test_fields{,_start}.npz   eval_scaling_e2e.py --split test --save_fields --save_fields_start
 two-stage   E1_baseline/metrics/test_fields_<model> <data>.npz   eval_metrics.py --fields_only
Every array is the centred output region of a 2560^2 field (T output 2227^2 for the BEST architecture, 2419^2 for
step 3 on the full field), so the crop is given in raw cells of the 2560^2 field and shifted per file by
(2560 - h) // 2. The labels of all configurations are checked to agree on the crop.

Crop presets (raw cells, first array axis = plot x):
 soft_vs_hard  rows / cols 900-1400 of the streamline channel in soft_vs_hard.py (CNN1 output region, origin at
               raw cell 94) -> 994-1494 / 994-1494
 first_runs    x 900-1500 / y 1200-1800 of the T output in first_runs.py (example_e2e_first, 2180^2, origin at
               raw cell 190) -> 1090-1690 / 1390-1990

usage (from code/): python -m postprocessing.thesis.result_fields [--crop soft_vs_hard|first_runs|x0 x1 y0 y1]
                    [--err_max E] [--diff_max D] [--out <dir>]"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.thesis.common import FIGURES, FORPAPER, save

FIELD = 2560   # raw cells of the test field per axis
CROPS = {"soft_vs_hard": (994, 1494, 994, 1494), "first_runs": (1090, 1690, 1390, 1990)}
E1 = FORPAPER / "E1_baseline" / "metrics"
PREP = "dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t"
CENTRAL = FORPAPER / "E3_finetune" / "central"
# configuration -> field file
FILES = {"chain": E1 / f"test_fields_BEST_predict_T_add_s_outer {PREP} prep_with_BEST_predict_v_v4 RK45.npz",
         "isolated": E1 / f"test_fields_BEST_predict_T_add_s_outer {PREP}.npz",
         "sequential": E1 / f"test_fields_sequential_T_bestv {PREP} prep_with_BEST_predict_v_v4 RK45.npz",
         "start": CENTRAL / "central_frozenT_r2" / "test_eval" / "test_fields_start.npz",
         "frozenT": CENTRAL / "central_frozenT_r2_ruleA" / "test_eval" / "test_fields.npz",
         "joint": CENTRAL / "central_joint_r3_ruleA" / "test_eval" / "test_fields.npz"}
FRAME = FORPAPER / "E4_lines" / "frame"
# BatchNorm statistics follow the weights (freeze_bn = freeze_weights, *_ruleA, 2026-11-07; F_frozenV had it already)
FILES.update({"lamN1": FRAME / "finetune_e2e_frame_3_ruleA" / "test_eval" / "test_fields.npz",
              "cnn2": FRAME / "F_frozenV" / "test_eval" / "test_fields.npz",
              "a2": FORPAPER / "E4_lines" / "a2" / "A2_lamN1V_frozenV_ruleA" / "test_eval" / "test_fields.npz",
              "traj": FORPAPER / "E4_lines" / "trajectory" / "traj_lr1e-5_ruleA" / "test_eval" / "test_fields.npz"})
START = "soft chain, untrained"   # BEST_V + BEST_T with the soft drawing, no training
# figure -> (columns [(configuration, title)], rows)
FIGS = {"baselines": ([("chain", "Pelzer's chain"), ("isolated", "isolated step 3"), ("sequential", "sequential CNN2")],
                      ["T", "err"]),
        "soft_vs_hard": ([("chain", "Pelzer's chain"), ("start", START)], ["s", "T", "err"]),
        "central": ([("start", START), ("frozenT", "CNN1 only"), ("joint", "joint")], ["T", "err", "diff"]),
        "trajectory": ([("start", START), ("traj", "trajectory loss")], ["T", "err", "diff"]),
        "frame": ([("start", START), ("lamN1", r"$\lambda_N = 1$"), ("cnn2", "CNN2 alone"),
                   ("a2", "CNN2 on the lines\nof CNN1 alone")], ["T", "err", "diff"])}
# row -> (row title, colormap)
ROWS = {"s": (r"streamlines $s + s_o$", "magma"), "T": (r"$T$ [$^\circ$C]", "RdBu_r"),
        "err": (r"$T - T_\mathrm{label}$ [$^\circ$C]", "PuOr_r"), "diff": (r"$T - T_\mathrm{start}$ [$^\circ$C]", "PuOr_r")}
ERR_PERCENTILE = 99   # default symmetric error / difference scale: this percentile of |values| over all panels


def crop_of(a, crop):
    """Raw-cell crop of a centred [h, w] output array."""
    x0, x1, y0, y1 = crop
    o = (FIELD - a.shape[0]) // 2, (FIELD - a.shape[1]) // 2
    assert x0 - o[0] >= 0 and y0 - o[1] >= 0 and x1 - o[0] <= a.shape[0] and y1 - o[1] <= a.shape[1], \
        f"crop {crop} outside the output region {a.shape} (origin {o})"
    return a[x0 - o[0]:x1 - o[0], y0 - o[1]:y1 - o[1]].astype(np.float32)


def load(crop):
    """{configuration: {"T", "label", "err", "s"?} on the crop} for the existing field files."""
    out = {}
    for name, f in FILES.items():
        if not f.exists():
            print(f"  skipped {name}: {f} missing")
            continue
        d = np.load(f)
        T, label = d["T_pred"].astype(np.float32), d["T_label"].astype(np.float32)
        out[name] = {"T": crop_of(T, crop), "label": crop_of(label, crop), "s": crop_of(d["s_pred"], crop) if "s_pred" in d else None}
        out[name]["err"] = out[name]["T"] - out[name]["label"]
        print(f"  {name:10s} {T.shape[0]}^2  test T MAE full {np.abs(T - label).mean():.4f}, crop {np.abs(out[name]['err']).mean():.4f}")
    ref = next(iter(out.values()), None)
    for name, d in out.items():
        # the labels have to coincide on the crop, else the alignment is wrong
        dev = np.abs(d["label"] - ref["label"]).max()
        assert dev < 0.02, f"label of {name} deviates by {dev:.3g} degC from the reference on the crop"
    if "start" in out:
        for d in out.values():
            d["diff"] = d["T"] - out["start"]["T"]
    return out


def figure(name, fields, scales, out_dir, tag):
    cols, rows = FIGS[name]
    cols = [(c, t) for c, t in cols if c in fields]
    if not cols:
        print(f"  {name}: no configuration available, skipped")
        return
    rows = [r for r in rows if r != "diff" or "start" in fields]
    label = fields[cols[0][0]]["label"]
    n_c = len(cols) + 1
    fig, axes = plt.subplots(len(rows), n_c, figsize=(2.9 * n_c + 0.8, 2.9 * len(rows)), layout="constrained", squeeze=False)
    for i, r in enumerate(rows):
        title, cmap = ROWS[r]
        vmin, vmax = scales[r]
        panels = [("label", label if r == "T" else None)] + [(c, fields[c].get(r)) for c, _ in cols]
        im = None
        for j, (c, a) in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if a is None or (r == "diff" and c == "start"):
                ax.set_frame_on(False)   # no label error / channel, start minus itself
                continue
            im = ax.imshow(a.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        for j, (_, t) in enumerate([("label", r"label $T$")] + cols):
            if i == 0:
                axes[0, j].set_title(t)
        axes[i, 0].set_ylabel(title)
        lo = min(a.min() for c, a in panels if a is not None) < vmin
        hi = max(a.max() for c, a in panels if a is not None) > vmax
        fig.colorbar(im, ax=axes[i, :], shrink=0.9, pad=0.02,
                     extend={(True, True): "both", (True, False): "min", (False, True): "max"}.get((lo, hi), "neither"))
    save(fig, out_dir, f"{name}_{tag}")


def main(crop, tag, err_max, diff_max, out_dir):
    print(f"crop {tag}: raw cells x {crop[0]}-{crop[1]}, y {crop[2]}-{crop[3]}")
    fields = load(crop)
    if not fields:
        print("no field files found")
        return
    label = next(iter(fields.values()))["label"]
    def sym(key, given):
        if given is not None:
            return given
        # errors over all configurations, differences over the finetunes of the central figure (the only diff row)
        names = list(fields) if key == "err" else [c for c, _ in FIGS["central"][0] if c in fields and c != "start"]
        a = np.concatenate([fields[n][key].ravel() for n in names])
        return float(np.percentile(np.abs(a), ERR_PERCENTILE))
    e = sym("err", err_max)
    # same scales in every figure: T = range of the label on the crop, errors / differences symmetric
    scales = {"T": (float(label.min()), float(label.max())), "s": (0.0, 1.0), "err": (-e, e)}
    if "start" in fields:
        dm = sym("diff", diff_max)
        scales["diff"] = (-dm, dm)
    print("  colour scales: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in scales.items()))
    for name in FIGS:
        figure(name, fields, scales, out_dir, tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--crop", nargs="+", default=["soft_vs_hard"],
                   help=f"preset {list(CROPS)} or raw cells x0 x1 y0 y1 of the 2560^2 field (default soft_vs_hard)")
    p.add_argument("--err_max", type=float, default=None,
                   help=f"colour limit of the signed error (default: {ERR_PERCENTILE}th percentile of |error| over all panels)")
    p.add_argument("--diff_max", type=float, default=None, help="colour limit of the difference to the start (same default)")
    p.add_argument("--out", type=Path, default=FIGURES / "result_fields")
    a = p.parse_args()
    if len(a.crop) == 1:
        crop, tag = CROPS[a.crop[0]], a.crop[0]
    else:
        crop = tuple(int(v) for v in a.crop)
        tag = "x{}-{}_y{}-{}".format(*crop)
    main(crop, tag, a.err_max, a.diff_max, a.out)
