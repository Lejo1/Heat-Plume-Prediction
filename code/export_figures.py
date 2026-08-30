"""Final figure export: paper-style inference plots per run + cross-run comparison bar charts.

Run from the repository root (the directory containing runs/ and code/):

    python code/export_figures.py

Everything is configured in the CONFIG block below - edit RUNS and re-run. Nothing is read from the
command line, so a committed version of this file reproduces exactly the figure set it made.

Two kinds of entry, so a two-stage pipeline and an end-to-end run can sit in the same chart:

  two-stage  {"model": <step-3 run>, "prep_with": <step-1 run or None>[, "metrics": <yaml>]}
             The step-2 dataset is BUILT if it does not exist yet (build_streamlines) and the
             metrics COMPUTED if they do not exist yet (eval_metrics). `prep_with: None` selects the
             isolated dataset, i.e. streamlines from the simulated velocities.
  end-to-end {"run": <e2e run dir>}
             Reads the run's own measurements_test{,_start}.yaml.

Inference plots use Pelzer's own primitives (DataToVisualize / plot_datafields with the
jp_temperature / jp_linear colormaps), so every entry comes out in the same per-field
"prediction / label / absolute error" style as the baselines' val_*.png.

COST: building a step-2 dataset traces streamlines for every datapoint, and every inference plot is
a full-domain 2560^2 forward. This belongs on the GPU box. Set the MAKE_* switches to False for the
cheap half (bar charts only, which just read metric yaml files).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from postprocessing.visualization import (DataToVisualize, plot_datafields, prepare_data_to_plot,
                                          reverse_norm_one_dp)
from preprocessing.datasets.dataset import DataPoint, DataPointE2E
from preprocessing.transforms import NormalizeTransform
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.networks.unetVariants import UNetNoPad2
from step2_streamlines.streamlines_main import build_streamlines
import eval_metrics as em

# ==================================================================================================
# CONFIG - edit this block
# ==================================================================================================

DATASET = "dataset_giant_100hp_varyK"
RANDOM_K = True                      # True for the varyK datasets (flow main direction switched)
METHOD_TAG = "RK45"                  # only names the generated folder; the tracer is fixed-step RK4

RUNS = [
    # Pelzer's complete two-stage pipeline: her step 1 feeding her step 3. This is the paper's own
    # chain, and the state every end-to-end run below is finetuned from.
    dict(label="Pelzer 2-stage\n(BEST_V -> BEST_T)",
         model="../models/BEST_predict_T_add_s_outer",
         prep_with="../models/BEST_predict_v_v4"),
    # our two-stage reproduction, chained through our own step 1
    dict(label="our 2-stage\n(baseline_v -> baseline_T)",
         model="../runs/baseline_T",
         prep_with="../runs/baseline_v"),
    # our step 3 on HER velocities: isolates step 3 from our step-1 quality, and makes the
    # comparison against the finetunes fair, since they all start from BEST_V
    dict(label="our baseline_T\non BEST_V",
         model="../runs/baseline_T",
         prep_with="../models/BEST_predict_v_v4"),
    # her step 3 on simulated velocities: the isolated step-3 upper bound (no step-1 error at all)
    dict(label="BEST_T isolated\n(simulated v)",
         model="../models/BEST_predict_T_add_s_outer",
         prep_with=None,
         paper_row="step3_simulated_v"),   # different task -> different reference row
    # end-to-end runs
    dict(label="e2e: blur + lr sched", run="../runs/good_fines/finetune_e2e_blur_lr_schedule"),
    dict(label="e2e: no BN, blur",     run="../runs/good_fines/finetune_e2e_nobn_blur_lr_schedule"),
    dict(label="e2e: lr sched only",   run="../runs/good_fines/finetune_lr_schedule"),
    dict(label="e2e: no BN, lr sched only",   run="../runs/good_fines/finetune_e2e_lr_schedule_no_bn_restimate"),
    
]

OUT_DIR = Path("figures")            # figures, and any metrics this script computes itself
MAKE_INFERENCE_PLOTS = True          # full-domain forward per entry - GPU only
MAKE_COMPARISON_PLOTS = True         # bar charts from the metric yaml files - cheap
BUILD_MISSING_PREPS = True           # generate a step-2 dataset when it is not on disk (slow!)
COMPUTE_MISSING_METRICS = True       # run eval_metrics when a metrics file is not on disk
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_INDEX = 2                      # index into order_data: 0=train, 1=val, 2=test
DPI = 300                            # Pelzer used 1200; 300 keeps the files manageable
PIC_FORMAT = "png"                   # "pdf" for vector figures in the thesis

# Include each e2e run's pre-finetuning state as its own bar. Strongly recommended: the start state
# is a real configuration (the two baselines joined through the soft streamlines, no training), and
# in the current results it is among the best - a chart without it overstates what training did.
INCLUDE_START_STATES = True

COMPARISON_METRICS = ["MAE [degC]", "RMSE [degC]", "Huber [degC]", "SSIM",
                      "PAT0.1 [%]", "PAT1.0 [%]", "Linf [degC]"]
PAPER_ROW = "end_to_end"             # default PAPER_REFERENCE key for the reference line. An entry
                                     # measuring a DIFFERENT task (e.g. isolated step 3, which has
                                     # no step-1 error at all) must override it with paper_row=...,
                                     # otherwise its bar is judged against an unrelated number.
SHOW_PAPER_SPREAD = True             # shade +-1 std from PAPER_STATISTICS where available
# Zoom each panel's y-axis onto the data range instead of starting at 0. The runs differ by a few
# percent, so a 0-based axis shows nothing; the trade-off is a truncated axis, which is why every
# bar is also annotated with its exact value. Set False for honest-but-flat 0-based bars.
ZOOM_YAXIS = True
PLOT_STREAMLINE_CHANNELS = True      # render sf / sf+sf_outer for e2e entries
SEPARATE_BAR_CHARTS = True           # one standalone figure per metric -> figures/bars/<metric>.png
COMBINED_BAR_CHART = True            # the multi-panel overview -> figures/comparison_bars.png

# ==================================================================================================


def resolve(raw) -> Path:
    """Config paths are written relative to code/ (that is where main.py runs) but this script runs
    from the repo root. Accept either, and fail loudly listing both attempts."""
    q = Path(raw)
    if q.is_absolute():
        return q
    for base in (Path.cwd(), Path(__file__).parent):
        if (base / q).exists():
            return (base / q).resolve()
    raise FileNotFoundError(f"could not resolve {raw!r} from {Path.cwd()} or code/")


def prep_root() -> Path:
    return resolve("../datasets_prep")


def prep_dir_for(step1_model):
    """Where build_streamlines puts (or would put) the step-2 dataset for this step-1 model."""
    base = prep_root() / f"{DATASET} inputs_ixydk+s_outer outputs_t"
    return base if step1_model is None else Path(f"{base} prep_with_{step1_model.name} {METHOD_TAG}")


def ensure_prep(step1_model):
    """Return the step-2 dataset, generating it from step1_model's predicted velocities if absent."""
    target = prep_dir_for(step1_model)
    if target.exists():
        return target
    if step1_model is None:
        raise FileNotFoundError(f"{target} missing - produced by LGCNN_step2.py STEP 1/2")
    assert BUILD_MISSING_PREPS, f"{target} missing and BUILD_MISSING_PREPS is False"
    print(f"    building step-2 dataset from {step1_model.name} (streamline tracing, slow) ...")
    build_streamlines(model_path=step1_model, dataset_path=prep_root() / DATASET,
                      based_on_pred=True, method=METHOD_TAG, randomK=RANDOM_K)
    assert target.exists(), f"build_streamlines did not produce {target}"
    print(f"    -> {target.name}")
    return target


def ensure_metrics(model_dir: Path, data_dir: Path, explicit=None):
    """Paper-style metrics of `model_dir` on `data_dir`, computed if not already on disk.

    `explicit` short-circuits the search - useful because eval_metrics writes wherever its
    --destination pointed, which is not necessarily next to the model it evaluated."""
    if explicit:
        f = resolve(explicit)
        return yaml.safe_load(f.read_text())["test"], f.name
    fname = f"metrics_e2e-style_{model_dir.name} {data_dir.name}.yaml"
    for cand in (model_dir / fname, OUT_DIR / "metrics" / fname):
        if cand.exists():
            return yaml.safe_load(cand.read_text())["test"], cand.name
    assert COMPUTE_MISSING_METRICS, f"{fname} missing and COMPUTE_MISSING_METRICS is False"
    print(f"    evaluating {model_dir.name} on {data_dir.name} (full-domain, slow) ...")
    dest = OUT_DIR / "metrics"
    dest.mkdir(parents=True, exist_ok=True)
    loaders, model, norm, info, args, out_ch = em.preparation(data_dir, model_dir, dest)
    collected = em.collect_metrics(data_dir, model_dir, dest, loaders, model, norm, info, args, out_ch)
    styled = em.to_e2e_style(collected, out_ch)
    em.save_yaml(styled, dest / fname)
    return styled["test"], fname


def unet_args_from(hps: dict) -> dict:
    return dict(depth=hps["depth"]["values"][0], init_features=hps["init_features"]["values"][0],
                kernel_size=hps["kernel_size"]["values"][0], stride=hps["stride"]["values"][0],
                dilation=hps["dilation"]["values"][0], activation=hps["activation_fct"]["values"][0],
                norm=hps["norm"]["values"][0], repeat_inner=hps["repeat_inner"]["values"][0])


def fields_two_stage(model_dir: Path, data_dir: Path):
    """Single-UNet run evaluated on `data_dir`: Pelzer's own preparation path, unchanged."""
    args = yaml.safe_load((model_dir / "command_line_arguments.yaml").read_text())
    hps = yaml.safe_load((model_dir / "HPS_options.yaml").read_text())
    info = yaml.safe_load((model_dir / "info.yaml").read_text())
    ds = DataPoint(data_dir, i=args["order_data"][SPLIT_INDEX])
    x, y = ds[0]
    model = UNetNoPad2(in_channels=ds.input_channels, out_channels=ds.output_channels,
                       **unet_args_from(hps)).float()
    model.load(model_dir, DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0).to(DEVICE)).cpu()
    x_r, y_r, out_r = reverse_norm_one_dp(x, y, out, NormalizeTransform(info))
    return prepare_data_to_plot(x_r, y_r, out_r, info)


def fields_e2e(run_dir: Path):
    """End-to-end run: rebuild the model as training_e2e did, then mirror prepare_data_to_plot."""
    args = yaml.safe_load((run_dir / "command_line_arguments.yaml").read_text())
    hps = yaml.safe_load((run_dir / "HPS_options.yaml").read_text())
    ds = DataPointE2E(prep_root(), Path(args["data_raw"]).name, i=args["order_data"][SPLIT_INDEX])
    x, y = ds[0]
    ua = unet_args_from(hps)
    model = LGCNNEndToEnd(v_stats=ds.info_v["Labels"], unet_args=ua,
                          unet_args_T={**ua, **(args.get("unet_args_T") or {})},
                          randomK_data=args["randomK"], t_steps=args["t_steps"], sigma=args["sigma"],
                          fade_mode=args.get("fade_mode", "absolute"),
                          detach_direct_v=args.get("detach_direct_v", False),
                          v_blur=args.get("v_blur", 0.0) or 0.0).float()
    model.load(run_dir, DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(DEVICE)).cpu()[0]

    h, w = pred.shape[1:]
    i0, j0 = (y.shape[1] - h) // 2, (y.shape[2] - w) // 2
    y_c = y[:, i0:i0 + h, j0:j0 + w]
    cell = np.array(ds.info_T["CellsSize"][:2])
    extent = cell * np.array([h, w])
    dn = lambda t, s: t * (s["max"] - s["min"]) + s["min"]
    T = ds.info_T["Labels"]["Temperature [C]"]
    vx = ds.info_v["Labels"]["Liquid X-Velocity [m_per_y]"]
    vy = ds.info_v["Labels"]["Liquid Y-Velocity [m_per_y]"]

    d = {}
    for name, out, true in [("Temperature [C]", dn(pred[0], T), dn(y_c[0], T)),
                            ("Liquid X-Velocity [m_per_y]", dn(pred[1], vx), dn(y_c[1], vx)),
                            ("Liquid Y-Velocity [m_per_y]", dn(pred[2], vy), dn(y_c[2], vy))]:
        lo, hi = float(min(out.min(), true.min())), float(max(out.max(), true.max()))
        d[f"{name}_true"] = DataToVisualize(true, "Label", name, extent, vmax=hi, vmin=lo)
        d[f"{name}_out"] = DataToVisualize(out, "Prediction", name, extent, vmax=hi, vmin=lo)
        d[f"{name}_error"] = DataToVisualize(torch.abs(true - out), "Absolute Error", name, extent)
    x_ext = cell * np.array(x.shape[-2:])
    for name, meta in ds.info_v["Inputs"].items():
        d[name] = DataToVisualize(x[meta["index"]] * (meta["max"] - meta["min"]) + meta["min"],
                                  "Input", name, x_ext)
    if PLOT_STREAMLINE_CHANNELS and model.last_intermediates:
        sf = model.last_intermediates["sf"].cpu()[0, 0]
        sfo = model.last_intermediates["sf_outer"].cpu()[0, 0]
        d["Streamlines Fade"] = DataToVisualize(sf, "Streamline channel (soft)", "Streamlines Fade", extent)
        d["Streamlines"] = DataToVisualize(sf + sfo, "Streamlines incl. offsets", "Streamlines", extent)
    return d


def slug(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label.replace("\n", "_"))


def entry_metrics(e: dict, which: str = "final"):
    """(metrics dict, source name) for one RUNS entry."""
    if "run" in e:
        f = resolve(e["run"]) / ("measurements_test.yaml" if which == "final"
                                 else "measurements_test_start.yaml")
        return (yaml.safe_load(f.read_text()), f.name) if f.exists() else (None, None)
    if which != "final":
        return None, None                      # a two-stage entry has no "before training" state
    if e.get("metrics"):                       # explicit file: no prep needed to read it
        return ensure_metrics(resolve(e["model"]), None, explicit=e["metrics"])
    data_dir = ensure_prep(resolve(e["prep_with"]) if e.get("prep_with") else None)
    return ensure_metrics(resolve(e["model"]), data_dir)


def _draw_metric(ax, key, series, standalone=False):
    """One metric's bars on one axes. Shared by the combined figure and the per-metric ones."""
    ref, stat = em.PAPER_REFERENCE.get(PAPER_ROW, {}), em.PAPER_STATISTICS.get(PAPER_ROW, {})
    colors = ["#c9d3db" if st else "#2a7fb8" for _, _, st, _ in series]
    vals = [m[key] for _, m, _, _ in series]
    for b, st in zip(ax.bar(range(len(vals)), vals, color=colors), [s[2] for s in series]):
        if st:
            b.set_hatch("//")
    short = key.split(" [")[0]
    paper, sub, lo_ref, hi_ref = ref.get(short), "", None, None
    if paper is not None:
        ax.axhline(paper, color="#c1440e", lw=1.6, ls="--", zorder=3,
                   label=f"paper {short} = {paper:.4f}" if standalone else None)
        sub, lo_ref, hi_ref = f"paper {paper:.4f}", paper, paper
        if SHOW_PAPER_SPREAD and short in stat:
            mean, std = stat[short]
            ax.axhspan(mean - std, mean + std, color="#c1440e", alpha=0.12, zorder=0,
                       label=f"Table 8 {mean:.4f} $\\pm$ {std:.4f}" if standalone else None)
            sub += f" | T8 {mean:.4f}$\\pm${std:.4f}"
            lo_ref, hi_ref = min(lo_ref, mean - std), max(hi_ref, mean + std)
    if ZOOM_YAXIS:
        lo, hi = min(vals), max(vals)
        if lo_ref is not None:
            lo, hi = min(lo, lo_ref), max(hi, hi_ref)
        pad = (hi - lo) * 0.18 or abs(hi) * 0.05 or 1.0
        ax.set_ylim(lo - pad, hi + pad * 1.4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([s[0].replace("\n", " ") for s in series],
                       rotation=90, fontsize=8 if standalone else 6.5)
    # entries measuring another task get their own reference drawn over their bar only
    for i, (_, _, _, row) in enumerate(series):
        if row == PAPER_ROW:
            continue
        own = em.PAPER_REFERENCE.get(row, {}).get(short)
        if own is not None:
            ax.hlines(own, i - 0.42, i + 0.42, color="#7b3294", lw=1.8, ls=":", zorder=4,
                      label=f"{row} = {own:.4f}" if standalone else None)
    ax.grid(axis="y", ls="--", alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8 if standalone else 6.5)
    if standalone:
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=12)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")
    else:
        ax.set_title(key + (f"\n{sub}" if sub else ""), fontsize=8)


def _suptitle(series) -> str:
    others = sorted({r for _, _, _, r in series if r != PAPER_ROW})
    return ("red dashed = paper single run, pale band = Table 8 mean $\\pm$1$\\sigma$ "
            f"(row: {PAPER_ROW})   |   hatched = before e2e training"
            + (f"\npurple dotted = own reference row for a different task: {', '.join(others)}"
               if others else ""))


def comparison_plots():
    series = []
    for e in RUNS:
        if INCLUDE_START_STATES and "run" in e:
            m0, src0 = entry_metrics(e, "start")
            if m0:
                series.append((f"{e['label']}\n(before e2e)", m0, True, e.get("paper_row", PAPER_ROW)))
                print(f"    {e['label']!r} start <- {src0}")
        m, src = entry_metrics(e, "final")
        if m is None:
            print(f"    WARNING: no metrics for {e['label']!r} - skipped")
            continue
        print(f"    {e['label']!r:<44} <- {src}")
        series.append((e["label"], m, False, e.get("paper_row", PAPER_ROW)))
    if not series:
        print("    nothing to compare"); return

    usable = [k for k in COMPARISON_METRICS if all(k in m for _, m, _, _ in series)]
    if [k for k in COMPARISON_METRICS if k not in usable]:
        print(f"    note: {[k for k in COMPARISON_METRICS if k not in usable]} missing somewhere - omitted")

    if SEPARATE_BAR_CHARTS:
        d = OUT_DIR / "bars"
        d.mkdir(parents=True, exist_ok=True)
        for key in usable:
            fig, ax = plt.subplots(figsize=(0.75 * len(series) + 3.2, 6.0))
            _draw_metric(ax, key, series, standalone=True)
            fig.suptitle(_suptitle(series), fontsize=8, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out = d / f"{slug(key.split(' [')[0])}.{PIC_FORMAT}"
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
            print(f"    -> {out}")

    if COMBINED_BAR_CHART:
        ncol = min(4, len(usable))
        nrow = int(np.ceil(len(usable) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(0.62 * len(series) * ncol + 2.0, 5.4 * nrow),
                                 squeeze=False)
        for ax, key in zip(axes.flat, usable):
            _draw_metric(ax, key, series)
        for ax in axes.flat[len(usable):]:
            ax.axis("off")
        fig.suptitle("Run comparison on the test datapoint   |   " + _suptitle(series), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = OUT_DIR / f"comparison_bars.{PIC_FORMAT}"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        print(f"    -> {out}")

    csv = OUT_DIR / "comparison_table.csv"
    with csv.open("w", newline="") as f:                      # labels contain commas -> csv module
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["run"] + usable)
        for label, m, _, _ in series:
            w.writerow([label.replace("\n", " ")] + [f"{m[k]:.6f}" for k in usable])
    print(f"    -> {csv}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for e in RUNS:
        assert ("run" in e) ^ ("model" in e), f"{e['label']}: give either run= or model=+prep_with="
        resolve(e.get("run") or e["model"])            # fail early on a typo'd path

    if MAKE_INFERENCE_PLOTS:
        print(f"Inference plots (device {DEVICE}):")
        for e in RUNS:
            print(f"  {e['label']!r}")
            if "run" in e:
                d = fields_e2e(resolve(e["run"]))
            else:
                data_dir = ensure_prep(resolve(e["prep_with"]) if e.get("prep_with") else None)
                d = fields_two_stage(resolve(e["model"]), data_dir)
            target = OUT_DIR / "inference" / slug(e["label"])
            target.parent.mkdir(parents=True, exist_ok=True)
            plot_datafields(d, str(target), {"format": PIC_FORMAT, "dpi": DPI},
                            only_inner=False, plot_all_in_1_pic=False)
            print(f"    -> {len(d)} fields to {target.parent}/{target.name}_*.{PIC_FORMAT}")

    if MAKE_COMPARISON_PLOTS:
        print("Comparison charts:")
        comparison_plots()
    print(f"\nDone - everything in {OUT_DIR.resolve()}")
