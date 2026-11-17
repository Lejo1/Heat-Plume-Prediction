"""Figures of the final presentation (final_praesentation/example.tex).

Variants of the thesis figures (postprocessing/thesis/*, which stay unchanged) for slides: few panels per figure,
large fonts, and one orientation for all fields, the main flow going from top to bottom as in the LGCNN pipeline
figure of Pelzer et al. Tensor-order arrays [h, w] (first axis = main flow) are therefore plotted with
imshow(a, origin="upper") instead of the thesis view a.T with origin lower; the gradient diagnostic stores [w, h]
and is transposed first.

Figures (png, written to final_praesentation/fig/pres/):
 dataset_fields        training simulation (RUN_1): inputs k, p, pump positions i; labels v_x, v_y, T
 dataset_T_zoom        label T of the training simulation on a crop, with the pumps
 baseline_T            test crop: T of the simulation | Pelzer's chain | isolated step 3
 channels_T            test crop: hard streamline channel s + s_o of Pelzer's chain | its T prediction
 soft_vs_hard          test crop: T of the hard / soft drawing (no training) and their signed errors below
 first_runs            first e2e run without pretraining: v_y, streamlines, T of the simulation and of the run
 central               T gradient into CNN1 (joint, sigma_v 1): T of simulation | soft chain | after; errors below
 trajectory            trajectory loss: same layout
 frame                 losses along the true streamlines: same layout, lambda_N 1 and CNN2 alone
 line_error            signed error of the soft chain with the predicted and the true streamlines
 grad_full / grad_zoom |dL/dv| through the streamlines (soft chain, training field, sigma_v 0), whole field / zoom
 grad_blur             zoom, sigma_v 0 | sigma_v 1, with the predicted streamlines on top
 search                hyperparameter search of the finetuning: best val objective vs test T MAE
 scaling               scaling simulation, central 1200^2 cells: same layout, soft chain | joint + frame losses

Inputs: the runs in Bachelorarbeit/runs/forPaper (as the thesis scripts) and datasets_prep (dataset figures).

usage (from code/): python -m postprocessing.presentation.slides [--only name ...] [--out <dir>]"""
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from postprocessing.thesis.common import CODE_DIR, FIGURES, FORPAPER, TABLES
from postprocessing.thesis import result_fields as rf
from postprocessing.thesis import scaling_fields as sf

ROOT = (CODE_DIR / "../..").resolve()   # Bachelorarbeit/
OUT = ROOT / "final_praesentation" / "fig" / "pres"
PREP = ROOT / "datasets_prep"
PKI = PREP / "dataset_giant_100hp_varyK inputs_pki outputs_xy"
FOR_S = PREP / "dataset_giant_100hp_varyK inputs_ixyk outputs_t for_s"
TRAIN_RUN = "RUN_1.pt"          # training simulation (data 0)
TEST_CROP = rf.CROPS["first_runs"]   # raw cells of the test field, same crop as the thesis result figures
GRAD_CROP = (900, 1500, 900, 1500)   # tensor cells of the CNN1 output region of the training field (as plot_grad_flow)
DATA_CROP = (1000, 1600, 1000, 1600)   # raw cells of the training field for the T zoom
LINE_LEVEL = 0.05               # contour level of the soft channels s + s_o (as line_error_overlay)
TRAIN_C, TEST_C = "#2a7fb8", "#c1440e"   # categorical pair (validated: CVD dE 22.9)
LINE_C = {"pred": "black", "true": "#00a651"}

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out / name}.png")


def show(ax, a, title=None, **kw):
    """Field in tensor order, flow from top to bottom."""
    im = ax.imshow(a, origin="upper", interpolation=kw.pop("interpolation", "nearest"), **kw)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


def cbar(fig, im, ax, label=None, **kw):
    return fig.colorbar(im, ax=ax, shrink=kw.pop("shrink", 0.85), pad=0.02, label=label, **kw)


def rescale(t, info_entry):
    return t * (info_entry["max"] - info_entry["min"]) + info_entry["min"]


# ---------------------------------------------------------------- dataset
def dataset(out):
    import torch
    info_pki = yaml.safe_load((PKI / "info.yaml").read_text())
    info_t = yaml.safe_load((FOR_S / "info.yaml").read_text())
    x = torch.load(PKI / "Inputs" / TRAIN_RUN, weights_only=False).numpy()
    v = torch.load(PKI / "Labels" / TRAIN_RUN, weights_only=False).numpy()
    T = torch.load(FOR_S / "Labels" / TRAIN_RUN, weights_only=False).numpy()[0]
    inp, lab = info_pki["Inputs"], info_pki["Labels"]
    p = rescale(x[inp["Liquid Pressure [Pa]"]["index"]], inp["Liquid Pressure [Pa]"]) / 1e6
    k = np.log10(rescale(x[inp["Permeability X [m^2]"]["index"]], inp["Permeability X [m^2]"]))
    pumps = np.argwhere(x[inp["Material ID"]["index"]] > 0.5)   # Material ID 2 = heat pump cell
    vx = rescale(v[lab["Liquid X-Velocity [m_per_y]"]["index"]], lab["Liquid X-Velocity [m_per_y]"])
    vy = rescale(v[lab["Liquid Y-Velocity [m_per_y]"]["index"]], lab["Liquid Y-Velocity [m_per_y]"])
    T = rescale(T, info_t["Labels"]["Temperature [C]"])
    t_max = float(np.percentile(T, 99.9))
    print(f"  dataset: {len(pumps)} pump cells, vy mean {vy.mean():.0f} m/y, T {T.min():.2f}-{T.max():.2f} degC")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.6), layout="constrained")
    panels = [(k, r"permeability $\log_{10} k$", "viridis", r"[m$^2$]", None),
              (p, r"pressure $p$", "cividis", "[MPa]", None),
              (None, r"heat pumps $i$", None, None, None),
              (vx, r"velocity $v_x$", "RdBu_r", "[m/y]", "sym"),
              (vy, r"velocity $v_y$", "viridis", "[m/y]", None),
              (T, r"temperature $T$", "RdBu_r", r"[$^\circ$C]", (float(T.min()), t_max))]
    for ax, (a, title, cmap, unit, lim) in zip(axes.ravel(), panels):
        if a is None:   # pump positions as dots on an empty field
            ax.set_xlim(-0.5, k.shape[1] - 0.5); ax.set_ylim(k.shape[0] - 0.5, -0.5); ax.set_aspect("equal")
            ax.scatter(pumps[:, 1], pumps[:, 0], s=10, color="black")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
            continue
        kw = {}
        if lim == "sym":
            m = float(np.percentile(np.abs(a), 99.5)); kw = {"vmin": -m, "vmax": m}
        elif lim is not None:
            kw = {"vmin": lim[0], "vmax": lim[1]}
        im = show(ax, a, title, cmap=cmap, interpolation="antialiased", **kw)
        cbar(fig, im, ax, unit)
    # flow direction and the zoom box of dataset_T_zoom
    axes[0, 1].annotate("", xy=(0.5, 0.15), xytext=(0.5, 0.85), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=3, mutation_scale=25))
    axes[0, 1].text(0.55, 0.5, "flow", color="white", transform=axes[0, 1].transAxes, fontsize=16, va="center")
    x0, x1, y0, y1 = DATA_CROP
    axes[1, 2].add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, color="black", lw=1.5))
    save(fig, out, "dataset_fields")

    Tc = T[x0:x1, y0:y1]
    pc = pumps[(pumps[:, 0] >= x0) & (pumps[:, 0] < x1) & (pumps[:, 1] >= y0) & (pumps[:, 1] < y1)] - (x0, y0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    im = show(ax, Tc, rf"$T$, {x1 - x0}$\times${y1 - y0} cells", cmap="RdBu_r", vmin=float(T.min()), vmax=t_max)
    ax.scatter(pc[:, 1], pc[:, 0], s=40, facecolor="none", edgecolor="black", lw=1.5, label="heat pump")
    ax.legend(loc="lower right", framealpha=0.9)
    cbar(fig, im, ax, r"[$^\circ$C]")
    save(fig, out, "dataset_T_zoom")


# ---------------------------------------------------------------- test-field result figures
def test_fields():
    """Thesis loader on the test crop, plus shared colour scales (as in result_fields.main)."""
    f = rf.load(TEST_CROP)
    label = f["start"]["label"]
    e = float(np.percentile(np.abs(np.concatenate([d["err"].ravel() for d in f.values()])), rf.ERR_PERCENTILE))
    dm = float(np.percentile(np.abs(np.concatenate([f[c]["diff"].ravel() for c in ("frozenT", "joint")])), rf.ERR_PERCENTILE))
    # T scale up to the T_PERCENTILE-th percentile of the simulation (as scaling_fields): the pump cells reach 15.4 degC
    return f, {"T": (float(label.min()), float(np.percentile(label, sf.T_PERCENTILE))), "err": (-e, e), "diff": (-dm, dm), "s": (0.0, 1.0)}


CMAP = {"T": "RdBu_r", "err": "PuOr_r", "diff": "PuOr_r", "s": "magma"}
CLABEL = {"T": r"$T$ [$^\circ$C]", "err": r"$T - T_\mathrm{label}$ [$^\circ$C]",
          "diff": r"$T - T_\mathrm{soft\ chain}$ [$^\circ$C]", "s": r"$s + s_o$"}


def t_err_grid(out, name, cols, scales, sim=True, label=None, size=3.6, diff=None):
    """Temperatures on top (optionally the simulation first), their signed errors below; one colour bar per row,
    small gaps between the panels. cols = [(title, field dict with "T" and "err")].
    diff = (title, array, colour-bar label): third row with one panel below the last column, own symmetric scale
    (ERR_PERCENTILE of |array|) and own colour bar."""
    n = len(cols) + (1 if sim else 0)
    rows = 3 if diff is not None else 2
    fig, axes = plt.subplots(rows, n, figsize=(size * n + 1.0, rows * size + 0.2), squeeze=False,
                             gridspec_kw=dict(wspace=0.03, hspace=0.06))
    j0 = 0
    if sim:
        im_t = show(axes[0, 0], label if label is not None else cols[0][1]["label"], "simulation",
                    cmap=CMAP["T"], vmin=scales["T"][0], vmax=scales["T"][1])
        axes[1, 0].set_axis_off()
        j0 = 1
    for j, (title, d) in enumerate(cols, start=j0):
        im_t = show(axes[0, j], d["T"], title, cmap=CMAP["T"], vmin=scales["T"][0], vmax=scales["T"][1])
        im_e = show(axes[1, j], d["err"], None, cmap=CMAP["err"], vmin=scales["err"][0], vmax=scales["err"][1])
    fig.colorbar(im_t, ax=axes[0, :], shrink=0.9, pad=0.01, fraction=0.025, label=CLABEL["T"], extend="max")
    fig.colorbar(im_e, ax=axes[1, :], shrink=0.9, pad=0.01, fraction=0.025, label=CLABEL["err"])
    if diff is not None:
        title, a, clab = diff
        m = float(np.percentile(np.abs(a), rf.ERR_PERCENTILE))
        for ax in axes[2, :-1]:
            ax.set_axis_off()
        im_d = show(axes[2, -1], a, None, cmap=CMAP["diff"], vmin=-m, vmax=m)
        axes[2, -1].set_xlabel(title)
        fig.colorbar(im_d, ax=axes[2, :], shrink=0.9, pad=0.01, fraction=0.025, label=clab)
    save(fig, out, name)


def t_row(out, name, cols, scales, size=3.6):
    """One row of temperatures with small gaps and one colour bar; cols = [(title, array)]."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(size * n + 1.0, size + 0.4), squeeze=False,
                             gridspec_kw=dict(wspace=0.03))
    for ax, (title, a) in zip(axes[0], cols):
        im = show(ax, a, title, cmap=CMAP["T"], vmin=scales["T"][0], vmax=scales["T"][1])
    fig.colorbar(im, ax=axes[0, :], shrink=0.9, pad=0.01, fraction=0.025, label=CLABEL["T"], extend="max")
    save(fig, out, name)


def results(out, which):
    f, sc = test_fields()
    S = "soft chain"
    if "baseline_T" in which:
        t_row(out, "baseline_T", [("simulation", f["chain"]["label"]), ("Pelzer's chain", f["chain"]["T"]),
                                  ("isolated step 3", f["isolated"]["T"])], sc)
    if "channels_T" in which:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), squeeze=False, gridspec_kw=dict(wspace=0.25))
        im = show(axes[0, 0], f["chain"]["s"], r"streamlines $s + s_o$", cmap=CMAP["s"], vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[0, 0], shrink=0.9, pad=0.02, fraction=0.05)
        im = show(axes[0, 1], f["chain"]["T"], "predicted temperature", cmap=CMAP["T"], vmin=sc["T"][0], vmax=sc["T"][1])
        fig.colorbar(im, ax=axes[0, 1], shrink=0.9, pad=0.02, fraction=0.05, label=r"[$^\circ$C]", extend="max")
        save(fig, out, "channels_T")
    if "soft_vs_hard" in which:
        t_err_grid(out, "soft_vs_hard", [("hard drawing", f["chain"]), ("soft drawing", f["start"])], sc)
    if "central" in which:
        t_err_grid(out, "central", [(S, f["start"]), ("joint, T loss", f["joint"])], sc)
    if "trajectory" in which:
        t_err_grid(out, "trajectory", [(S, f["start"]), ("trajectory loss", f["traj"])], sc)
    if "frame" in which:
        # CNN2 alone before lambda_N = 1; third row below lambda_N = 1: what the frame losses change beyond CNN2's adaptation
        lam = r"$\lambda_N = 1$, $\lambda_\mathrm{TOF} = 0.3$"
        t_err_grid(out, "frame", [(S, f["start"]), ("CNN$_2$ alone", f["cnn2"]), (lam, f["lamN1"])], sc,
                   diff=(lam + "\n" + r"$-$ CNN$_2$ alone", f["lamN1"]["T"] - f["cnn2"]["T"], r"$\Delta T$ [$^\circ$C]"))
    if "summary" in which:
        # one image per clickable row of the summary slide: predicted T and signed error, shared scales
        for key in ("chain", "start", "cnn2", "joint", "lamN1"):
            d = f[key]
            fig, axes = plt.subplots(1, 2, figsize=(2 * 3.6 + 2.0, 3.6 + 0.4), squeeze=False,
                                     gridspec_kw=dict(wspace=0.32))
            im_t = show(axes[0, 0], d["T"], r"predicted $T$", cmap=CMAP["T"], vmin=sc["T"][0], vmax=sc["T"][1])
            fig.colorbar(im_t, ax=axes[0, 0], shrink=0.9, pad=0.02, fraction=0.05, label=CLABEL["T"], extend="max")
            im_e = show(axes[0, 1], d["err"], "error", cmap=CMAP["err"], vmin=sc["err"][0], vmax=sc["err"][1])
            fig.colorbar(im_e, ax=axes[0, 1], shrink=0.9, pad=0.02, fraction=0.05, label=CLABEL["err"])
            save(fig, out, f"summary_{key}")
    if "line_error" in which:
        d = np.load(rf.FILES["start"])
        fig, ax = plt.subplots(figsize=(7.4, 6.9), layout="constrained")
        im = show(ax, f["start"]["err"], None, cmap=CMAP["err"], vmin=sc["err"][0], vmax=sc["err"][1])
        for key, c in (("s_pred", LINE_C["pred"]), ("s_label", LINE_C["true"])):
            ax.contour(rf.crop_of(d[key], TEST_CROP), levels=[LINE_LEVEL], colors=c, linewidths=0.9)
        cbar(fig, im, ax, CLABEL["err"])
        ax.legend(handles=[Line2D([], [], color=LINE_C["pred"], lw=2, label="predicted streamlines"),
                           Line2D([], [], color=LINE_C["true"], lw=2, label="true streamlines")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=2, frameon=False)
        save(fig, out, "line_error")


def first_runs(out):
    d = np.load(FORPAPER / "E3_finetune" / "first_runs" / "example_e2e_first" / "test_eval" / "test_fields.npz")
    x0, x1, y0, y1 = 900, 1500, 1200, 1800   # as the thesis first_runs.py (its own 2180^2 output region)
    c = lambda a: a[x0:x1, y0:y1].astype(np.float32)
    cols = [("vy", r"$v_y$ [m/y]", "viridis"), ("s", r"streamlines $s + s_o$", "magma"), ("T", r"$T$ [$^\circ$C]", "RdBu_r")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.4), layout="constrained")
    for j, (key, title, cmap) in enumerate(cols):
        ref = c(d[f"{key}_label"])
        vmin, vmax = (0.0, 1.0) if key == "s" else (float(ref.min()), float(ref.max()))
        for i, suffix in enumerate(("label", "pred")):
            im = show(axes[i, j], c(d[f"{key}_{suffix}"]), title if i == 0 else None, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar(fig, im, axes[:, j], location="bottom", shrink=0.9, extend="both")
    axes[0, 0].set_ylabel("simulation"); axes[1, 0].set_ylabel("e2e run, no pretraining")
    save(fig, out, "first_runs")


# ---------------------------------------------------------------- gradient
def grad(out, which):
    g = {b: np.load(FIGURES / "grad_flow" / f"graddiag_blur{b}" / "grad_diag_stage2_start.npz") for b in (0, 1)}
    # stored [w, h] -> tensor order [h, w]
    G = {b: np.asarray(d["dLdv_streamline"], dtype=float).T for b, d in g.items()}
    L = {b: np.asarray(d["streamlines"], dtype=float).T for b, d in g.items()}
    vmax = max(np.log10(a.max()) for a in G.values())
    vmin = vmax - 8
    log = lambda a: np.log10(np.maximum(a, 1e-14))
    x0, x1, y0, y1 = GRAD_CROP
    lab = r"$\log_{10}|\partial \mathcal{L}/\partial v|$"
    for b, a in G.items():
        print(f"  sigma_v {b}: dL/dv != 0 on {100 * (a > 0).mean():.1f} % of the cells, "
              f"lines (> {LINE_LEVEL}) cover {100 * (L[b] > LINE_LEVEL).mean():.1f} %")
    if "grad_full" in which:
        a = G[0]
        n = (a.shape[0] // 4) * 4
        pooled = a[:n, :n].reshape(n // 4, 4, n // 4, 4).max(axis=(1, 3))   # max-pool: keeps the thin lines visible
        fig, ax = plt.subplots(figsize=(6.6, 5.8), layout="constrained")
        im = show(ax, log(pooled), None, cmap="magma", vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((y0 / 4, x0 / 4), (y1 - y0) / 4, (x1 - x0) / 4, fill=False, color="#00c8ff", lw=2))
        cbar(fig, im, ax, lab)
        save(fig, out, "grad_full")
    def zoom(ax, b):
        im = show(ax, log(G[b][x0:x1, y0:y1]), None, cmap="magma", vmin=vmin, vmax=vmax)
        ax.contour(L[b][x0:x1, y0:y1], levels=[LINE_LEVEL], colors="#00c8ff", linewidths=0.5, alpha=0.4)
        return im
    if "grad_zoom" in which:
        fig, ax = plt.subplots(figsize=(6.6, 5.8), layout="constrained")
        cbar(fig, zoom(ax, 0), ax, lab)
        save(fig, out, "grad_zoom")
    if "grad_blur" in which:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), layout="constrained")
        for ax, b in zip(axes, (0, 1)):
            im = zoom(ax, b)
            ax.set_title(rf"$\sigma_v = {b}$:  {100 * (G[b] > 0).mean():.1f} % of the cells")
        cbar(fig, im, axes, lab)
        fig.legend(handles=[Line2D([], [], color="#00c8ff", lw=2, label="predicted streamlines")],
                   loc="outside lower center", frameon=False)
        save(fig, out, "grad_blur")


# ---------------------------------------------------------------- charts
def search(out):
    from postprocessing.thesis.search_scatter import SEARCHES, reference, trials
    rows = [r for r in trials(SEARCHES["finetune"]) if r["test T MAE"] is not None]
    v_ref, t_ref = reference()
    best = min(rows, key=lambda r: r["objective"])
    n_above = sum(r["test T MAE"] > t_ref for r in rows)
    print(f"  search: {len(rows)} trials with test, {n_above} above the start on test, best by val {best['test T MAE']:.4f}")
    fig, ax = plt.subplots(figsize=(8.4, 5.6), layout="constrained")
    ax.scatter([r["objective"] for r in rows], [r["test T MAE"] for r in rows], s=40, color=TRAIN_C, label="trial")
    ax.scatter([best["objective"]], [best["test T MAE"]], s=90, color=TEST_C, label="best by validation", zorder=3)
    ax.axhline(t_ref, color="0.3", ls="--", lw=1.5)
    if v_ref is not None:
        ax.axvline(v_ref, color="0.3", ls=":", lw=1.5)
    # labels in axes coordinates along the reference lines (placed after all limits are fixed)
    ax.text(0.99, t_ref, "soft chain, test ", transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=13, color="0.3")
    if v_ref is not None:
        ax.text(v_ref, 0.98, "soft chain, val ", transform=ax.get_xaxis_transform(), va="top", ha="right",
                fontsize=13, color="0.3")
    ax.set_xlabel("best val objective (Huber, normalized T)")
    ax.set_ylabel(r"test MAE $T$ [$^\circ$C]")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="0.92"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center")
    save(fig, out, "search")


def scaling(out):
    c0 = (sf.FIELD - sf.CENTRE) // 2
    view = (c0, c0 + sf.CENTRE, c0, c0 + sf.CENTRE)
    d = {n: np.load(sf.FILES[n]) for n in ("start", "frame")}
    lab = sf.view_of(d["start"]["T_label"].astype(np.float32), view, 1)
    err = {n: sf.view_of(d[n]["T_pred"].astype(np.float32), view, 1) - lab for n in d}
    e = float(np.percentile(np.abs(np.concatenate([a.ravel() for a in err.values()])), rf.ERR_PERCENTILE))
    t_max = float(np.percentile(lab, sf.T_PERCENTILE))
    for n, a in err.items():
        print(f"  scaling centre {n}: MAE {np.abs(a).mean():.4f}")
    cols = [(t, {"T": err[n] + lab, "err": err[n]}) for n, t in
            (("start", "soft chain"), ("frame", r"joint + $\mathcal{L}_N$, $\mathcal{L}_\mathrm{TOF}$"))]
    t_err_grid(out, "scaling", cols, {"T": (float(lab.min()), t_max), "err": (-e, e)}, label=lab)


ALL = ["summary", "dataset", "baseline_T", "channels_T", "soft_vs_hard", "first_runs", "central", "trajectory", "frame", "line_error",
       "grad_full", "grad_zoom", "grad_blur", "search", "scaling"]


def main(which, out):
    if "dataset" in which:
        dataset(out)
    r = [w for w in which if w in ("baseline_T", "channels_T", "soft_vs_hard", "central", "trajectory", "frame", "line_error", "summary")]
    if r:
        results(out, r)
    if "first_runs" in which:
        first_runs(out)
    g = [w for w in which if w.startswith("grad_")]
    if g:
        grad(out, g)
    if "search" in which:
        search(out)
    if "scaling" in which:
        scaling(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.only, a.out)
