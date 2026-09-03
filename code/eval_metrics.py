import argparse
from pathlib import Path
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from utils.utils_args import load_yaml, save_yaml
from preprocessing import subdomain
from preprocessing.data_init import init_data
from preprocessing.transforms import NormalizeTransform
from processing.networks.unetVariants import UNetNoPad2, UNet
from processing.loss_fcts import SSIMLoss, LinfLoss, PATLoss

def preparation(PATH_current_data: Path, PATH_current_model: Path, PATH_destination: Path, scaling:bool=False):
    # load data for dummyK, step1 (predV)

    args = load_yaml(PATH_current_model / "command_line_arguments.yaml")
    args["data_prep"] = PATH_current_data
    args["model"] = PATH_current_model
    args["case"] = "test"
    args["destination"] = PATH_destination
    print(args)

    if scaling:
        args["order_data"] = [0,0]
    input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=False, ORDER_DATA=args["order_data"])
    hparams = load_yaml(PATH_current_model / "HPS_options.yaml")
    print(hparams)

    settings = {"init_features": hparams["init_features"]["values"][0], 
                "depth": hparams["depth"]["values"][0],
                "kernel_size": hparams["kernel_size"]["values"][0],
                "stride": hparams["stride"]["values"][0],
                "dilation": hparams["dilation"]["values"][0],
                "activation": hparams["activation_fct"]["values"][0],
                "norm": hparams["norm"]["values"][0],
                "repeat_inner": hparams["repeat_inner"]["values"][0],
                }
    if "ddunet" in PATH_current_model.name:
        pass
    elif "unetnopad" in PATH_current_model.name:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    elif "unet" in PATH_current_model.name:
        model = UNet(input_channels, output_channels, **settings)
    else:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    model.load(PATH_current_model)
    model.eval()
    
    info = load_yaml(PATH_current_model / "info.yaml")
    norm = NormalizeTransform(info)
    print(info)

    return dataloaders, model, norm, info, args, output_channels

def collect_metrics(PATH_current_data, PATH_current_model, PATH_destination, dataloaders, model, norm, info, args, output_channels):
    collected_metrics = {"model": PATH_current_model.name, "data": PATH_current_data.name, "order data": args["order_data"]}
    for case, dataloader in dataloaders.items():
        collected_metrics[case] = {}
        metrics:dict = {"MSE [phys. unit^2]": MSELoss(), "MAE [phys. unit]": L1Loss(), "Linf [phys. unit]": LinfLoss(), "Huber [phys. unit]": HuberLoss(), "SSIM": SSIMLoss()}
        if output_channels == 1: # some only make sense for Temperature predictions
            metrics["PAT0.1 [%]"], metrics["PAT1.0 [%]"] = PATLoss(pat_thresholds=[0.1]), PATLoss(pat_thresholds=[1])
        for metric_name, metric in metrics.items():
            print(f"Calculating {metric_name} for {case}",end=" ")
            metrics_values = []

            for batch in dataloader:
                inputs, targets = batch
                outputs = model(inputs).detach()

                inputs, targets = crop_to_output_size(inputs, targets, outputs)
                
                if metric_name in ["SSIM", "IoU"]:
                    values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])
                else:
                    # unnormalize inputs and targets
                    reverse_normalization(norm, inputs, targets, outputs)
                    # calc metrics per output channel
                    if "PAT" in metric_name:
                        values = torch.mean(torch.Tensor(metric(outputs, targets).squeeze()))
                    else:
                        values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])

                metrics_values.append(values)

            assert len(dataloader) == 1, "I assumed I always have only one batch - otherwise please rethink this code"
            metrics[metric_name] = torch.mean(torch.stack(metrics_values), dim=0) # average over all batches
            print(f": average = {metrics[metric_name]}")

            collected_metrics[case][metric_name] = metrics[metric_name]
    save_yaml(collected_metrics, PATH_destination / f"metrics_paper25_{PATH_current_model.name} {PATH_current_data.name}.yaml")
    return collected_metrics

def to_e2e_style(collected_metrics, output_channels):
    """Convert collected metrics to exactly the names/units of the end-to-end evaluation
    (processing/training_e2e.py: evaluate_e2e), for direct baseline-vs-e2e comparison.
    Paper mapping: MAE=MAE, RMSE=sqrt(MSE), Huber=Huber, SSIM=SSIM (Linf/PAT are extras)."""
    styled = {}
    for case in [c for c in ["train", "val", "test"] if c in collected_metrics]:
        m = {k: torch.atleast_1d(torch.as_tensor(v)) for k, v in collected_metrics[case].items()}
        if output_channels == 1:  # temperature model
            styled[case] = {
                "MAE [degC]": float(m["MAE [phys. unit]"][0]),
                "RMSE [degC]": float(m["MSE [phys. unit^2]"][0]) ** 0.5,
                "Huber [degC]": float(m["Huber [phys. unit]"][0]),
                "Linf [degC]": float(m["Linf [phys. unit]"][0]),
                "SSIM": float(m["SSIM"][0]),
                "PAT0.1 [%]": float(m["PAT0.1 [%]"][0]),
                "PAT1.0 [%]": float(m["PAT1.0 [%]"][0]),
            }
        else:  # velocity model - same four metrics the paper's Step-1 row reports, per channel
            styled[case] = {
                "MAE vx [m/y]": float(m["MAE [phys. unit]"][0]),
                "MAE vy [m/y]": float(m["MAE [phys. unit]"][1]),
                "RMSE vx [m/y]": float(m["MSE [phys. unit^2]"][0]) ** 0.5,
                "RMSE vy [m/y]": float(m["MSE [phys. unit^2]"][1]) ** 0.5,
                "Huber vx [m/y]": float(m["Huber [phys. unit]"][0]),
                "Huber vy [m/y]": float(m["Huber [phys. unit]"][1]),
                "SSIM vx": float(m["SSIM"][0]),
                "SSIM vy": float(m["SSIM"][1]),
            }
    return styled


# Pelzer et al., Table "Metrics of all experiments" (main paper), LGCNN rows on synth. 3dp / test.
# ONE representative run each - Table 8 shows these are not means, so do not treat them as targets.
# Step 1 is per velocity component [m/y]; Step 3 and end-to-end are temperature [degC].
PAPER_REFERENCE = {
    "step1_vx": {"MAE": 22.3178, "RMSE": 31.1860, "Huber": 21.8237, "SSIM": 0.9739},
    "step1_vy": {"MAE": 32.7444, "RMSE": 45.0703, "Huber": 32.2488, "SSIM": 0.9733},
    "step3_simulated_v": {"MAE": 0.0417, "RMSE": 0.0762, "Huber": 0.0029, "SSIM": 0.8540},
    "step3_in_sequence": {"MAE": 0.0919, "RMSE": 0.1695, "Huber": 0.0139, "SSIM": 0.6714},
    "end_to_end": {"MAE": 0.0916, "RMSE": 0.1738, "Huber": 0.0146, "SSIM": 0.6841},
}

# Pelzer et al., Table 8 "Statistical error metrics ...": (mean, std) over REPEATED trainings,
# synth. 3dp / test / no scaling. This is the distribution a new run is drawn from, so |z| < 1 means
# a statistically ordinary result and chasing the single-run numbers above is chasing noise.
# Step 1 is extremely noisy (+-20% on the test MAE, +-71% on train vx); Step 3 is stable (+-1%).
# CAVEAT on end_to_end: that row carries the paper's footnote b, "with fixed first step
# predictions" - its small std EXCLUDES step-1 variance, so it is only comparable to a chain whose
# step-1 model is likewise held fixed.
PAPER_STATISTICS = {
    "step1_vx": {"MAE": (27.4355, 5.3603), "RMSE": (36.6786, 4.8168),
                 "Huber": (26.9402, 5.3588), "SSIM": (0.9727, 0.0025)},
    "step1_vy": {"MAE": (30.8755, 5.5264), "RMSE": (42.5114, 6.4231),
                 "Huber": (30.3799, 5.5256), "SSIM": (0.9734, 0.0042)},
    "step3_simulated_v": {"MAE": (0.0452, 0.0005), "RMSE": (0.0829, 0.0011),
                          "Huber": (0.0034, 0.0001), "SSIM": (0.8381, 0.0040)},
    "end_to_end": {"MAE": (0.0965, 0.0013), "RMSE": (0.1866, 0.0030),
                   "Huber": (0.0167, 0.0005), "SSIM": (0.6724, 0.0059)},
}


def compare_to_paper(styled, output_channels, case="test", data_name=""):
    """Print this run's metrics against the paper's single-run row AND its Table-8 distribution.

    The z column is what matters: |z| < 1 is an ordinary draw from the paper's own repeated runs.
    The single-run column is kept only because it is what the main table quotes.
    """
    if case not in styled:
        return
    v = styled[case]
    print(f"\n--- {case} vs. Pelzer et al. (synth. 3dp, test) " + "-" * 46)
    print("  'row' = single representative run (main table); 'Table 8' = mean +- std over repeated")
    print("  trainings. |z| < 1 = statistically ordinary; the row is NOT a target.")

    def line(prefix, label, ours, ref_key):
        row = PAPER_REFERENCE[ref_key][label]
        d = (ours - row) / row * 100
        stat = PAPER_STATISTICS.get(ref_key, {}).get(label)
        if stat is None:
            print(f"  {prefix:<4} {label:<7} {ours:10.4f} {row:10.4f} {d:+8.1f}% {'-':>21} {'-':>7}")
            return
        mean, std = stat
        z = (ours - mean) / std if std else float("nan")
        flag = "" if abs(z) < 1 else ("  <-- outside 1 sigma" if abs(z) < 3 else "  <-- OUTSIDE 3 sigma")
        print(f"  {prefix:<4} {label:<7} {ours:10.4f} {row:10.4f} {d:+8.1f}% "
              f"{mean:10.4f} +-{std:8.4f} {z:+7.2f}{flag}")

    print(f"  {'':<4} {'metric':<7} {'ours':>10} {'row':>10} {'delta':>9} "
          f"{'Table 8 mean':>10}   {'std':>8} {'z':>7}")
    if output_channels == 2:
        for name, ref_key in (("vx", "step1_vx"), ("vy", "step1_vy")):
            for label, key in (("MAE", f"MAE {name} [m/y]"), ("RMSE", f"RMSE {name} [m/y]"),
                               ("Huber", f"Huber {name} [m/y]"), ("SSIM", f"SSIM {name}")):
                line(name, label, v[key], ref_key)
    else:
        # which reference applies depends on where the streamlines came from: a "prep_with_<model>"
        # dataset was drawn from PREDICTED velocities (the chained pipeline), anything else from the
        # simulated ones (step 3 in isolation). Comparing against the wrong one is meaningless.
        chained = "prep_with" in str(data_name)
        ref_key, tag = ("end_to_end", "e2e") if chained else ("step3_simulated_v", "sim")
        print(f"  dataset {'contains' if chained else 'does not contain'} 'prep_with' -> streamlines "
              f"from {'PREDICTED' if chained else 'simulated'} v, comparing against '{ref_key}'.")
        keys = {"MAE": "MAE [degC]", "RMSE": "RMSE [degC]", "Huber": "Huber [degC]", "SSIM": "SSIM"}
        for label, key in keys.items():
            line(tag, label, v[key], ref_key)
        if chained:
            print("  NOTE: this Table-8 row uses FIXED first-step predictions (paper footnote b), so its")
            print("        std excludes step-1 variance - comparable only if your chain also pins step 1.")
            row = PAPER_REFERENCE["step3_in_sequence"]
            print("  'UNet in sequence' (single run) for reference: "
                  + " ".join(f"{k} {row[k]:.4f}" for k in ["MAE", "RMSE", "Huber", "SSIM"]))
        else:
            print("  CAVEAT: Table 8's step-3 mean (MAE 0.0452) disagrees with the main table's single")
            print("        run (0.0417) and with Pelzer's own shipped model (0.0347), so those repeats")
            print("        are evidently a different set. Trust its STD (~1%), not its mean, here.")

def plot_collected_metrics(collected_metrics, output_channels, save_path):
    """Bar chart of all metrics for all evaluated cases (train / val / test), per output channel."""
    cases = [c for c in ["train", "val", "test"] if c in collected_metrics]
    metric_names = list(collected_metrics[cases[0]].keys())
    channel_names = ["vx", "vy"] if output_channels == 2 else ["T"]
    colors = {"train": "blue", "val": "green", "test": "coral"}

    fig, axes = plt.subplots(1, len(metric_names), figsize=(3.2 * len(metric_names), 4))
    for ax, metric_name in zip(np.atleast_1d(axes), metric_names):
        labels, values, bar_colors = [], [], []
        for case in cases:
            vals = torch.atleast_1d(collected_metrics[case][metric_name])
            for ch in range(len(vals)):
                labels.append(f"{case} {channel_names[ch]}" if len(vals) > 1 else case)
                values.append(float(vals[ch]))
                bar_colors.append(colors[case])
        ax.bar(range(len(values)), values, color=bar_colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric_name, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.suptitle(f"{collected_metrics['model']}  on  {collected_metrics['data']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics plot to {save_path}")

def reverse_normalization(norm, inputs, labels, outputs):
    for tmp_in in inputs:
        norm.reverse(tmp_in, data_type="Inputs")
    for tmp_tar in labels:
        norm.reverse(tmp_tar, data_type="Labels")
    for tmp_out in outputs:
        norm.reverse(tmp_out, data_type="Labels")

def crop_to_output_size(inputs=None, labels=None, outputs=None):
    required_size = outputs.shape[-2:]
    start_pos = ((labels.shape[-2] - required_size[0])//2, (labels.shape[-1] - required_size[1])//2)
    if len(labels.shape) == 4:
        inputs = inputs[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    elif len(labels.shape) == 3:
        inputs = inputs[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    return inputs,labels

def metrics_of_one_model(file_name, n_outputs=2, directory=Path("."), run_ids=[1,2,3,4,5]):
    if "real" in file_name(1):
        metrics = {"train": {}, "val": {}}
        means = {"train": {}, "val": {}}
        stds = {"train": {}, "val": {}}
        mins_maxs = {"train": {}, "val": {}}
    else:
        metrics = {"test": {}, "train": {}, "val": {}}
        means = {"test": {}, "train": {}, "val": {}}
        stds = {"test": {}, "train": {}, "val": {}}
        mins_maxs = {"test": {}, "train": {}, "val": {}}
    inner_metrics = {
      "model": [],
      "Huber": [[] for _ in range(n_outputs)],
      "Linf": [[] for _ in range(n_outputs)],
      "MAE": [[] for _ in range(n_outputs)],
      "MSE": [[] for _ in range(n_outputs)],
      "SSIM": [[] for _ in range(n_outputs)],
    }
    for case in metrics.keys():
        try:
            metrics[case] = deepcopy(inner_metrics)
            if n_outputs == 1:
                metrics[case]["PAT0.1"] = [[]]
                metrics[case]["PAT1.0"] = [[]]

            # load the metrics from the yaml files
            for run in run_ids:
                try:
                    file = directory / file_name(run)
                    tmp = load_yaml(file)
                    metrics[case]["model"].append(tmp["model"])
                    for name, values in tmp[case].items():
                        name = name.split(" ")[0]
                        if n_outputs == 1:
                            metrics[case][name][0].append(values)
                        if n_outputs == 2:
                            metrics[case][name][0].append(values[0])
                            metrics[case][name][1].append(values[1])
                except FileNotFoundError:
                    print(f"File {file} not found")
                    continue
            # print(metrics_test["Huber"])
            
            # calc mean and std, max and min for each metric in metrics_test for each of the 2 entries
            means_tmp = {}
            stds_tmp = {}
            mins_maxs_tmp = {}
            for name, values in metrics[case].items():
                if name == "model":
                    continue
                means_tmp[name] = []
                stds_tmp[name] = []
                mins_maxs_tmp[name] = [[] for _ in range(n_outputs)]
                for prop in range(n_outputs):
                    means_tmp[name].append(np.mean(values[prop]))
                    stds_tmp[name].append(np.std(values[prop], ddof=1))
                    mins_maxs_tmp[name][prop].append(np.min(values[prop]))
                    mins_maxs_tmp[name][prop].append(np.max(values[prop]))
                    print(f"{case} {prop} {name}: {means_tmp[name][prop]:.4f} ± {stds_tmp[name][prop]:.4f} in [{mins_maxs_tmp[name][prop][0]:.4f}, {mins_maxs_tmp[name][prop][1]:.4f}]")
            means[case] = means_tmp
            stds[case] = stds_tmp
            mins_maxs[case] = mins_maxs_tmp
        except:
            print(f"{case} does not exist for {file_name}")
            continue
    return metrics, {"means": means, "stds": stds, "mins_maxs": mins_maxs}

def plot_metrics(metrics, title):
    n_cases = len(metrics.keys())
    n_outputs = len(metrics["train"]["MAE"])
    if n_outputs == 1:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
    elif n_outputs == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax0 = axs[0]
    ax0.boxplot(metrics["train"]["MAE"][0], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
    ax0.boxplot(metrics["val"]["MAE"][0], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
    if n_cases == 3:
        ax0.boxplot(metrics["test"]["MAE"][0], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
    ax0.boxplot(np.sqrt(metrics["train"]["MSE"][0]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax0.boxplot(np.sqrt(metrics["val"]["MSE"][0]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    if n_cases == 3:
        ax0.boxplot(np.sqrt(metrics["test"]["MSE"][0]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    if n_cases == 3:
        ax0.set_xticks([1, 2, 3, 4, 5, 6])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
    else:
        ax0.set_xticks([1, 2, 3, 4])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
    if n_outputs == 1:
        ax0.set_ylabel("T [°C]")
    else:
        ax0.set_ylabel("vx [m/y]")
        ax0.set_title("VX Metrics")
    ax0.grid(axis='y', linestyle='--', alpha=0.7)
    if n_outputs == 2:
        axs[1].boxplot(metrics["train"]["MAE"][1], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
        axs[1].boxplot(metrics["val"]["MAE"][1], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
        if n_cases == 3:
            axs[1].boxplot(metrics["test"]["MAE"][1], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
        axs[1].boxplot(np.sqrt(metrics["train"]["MSE"][1]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axs[1].boxplot(np.sqrt(metrics["val"]["MSE"][1]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        if n_cases == 3:
            axs[1].boxplot(np.sqrt(metrics["test"]["MSE"][1]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        if n_cases == 3:
            axs[1].set_xticks([1, 2, 3, 4, 5, 6])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
        else:
            axs[1].set_xticks([1, 2, 3, 4])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
        axs[1].set_ylabel("vy [m/y]")
        axs[1].set_title("VY Metrics")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    if n_cases == 3:
        plt.legend(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"], loc="upper left")
    else:
        plt.legend(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"], loc="upper left")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    # calculate metrics for one model.
    # IMPORTANT: the model and the prepared dataset must match in their channels, e.g. (baseline, run from repo root):
    #   step 1 isolated:  runs/baseline_v + "... inputs_pki outputs_xy"                                    (pki -> v)
    #   step 3 isolated:  runs/baseline_T + "... inputs_ixydk+s_outer outputs_t"                           (T on simulated v)
    #   full pipeline:    runs/baseline_T + "... inputs_ixydk+s_outer outputs_t prep_with_baseline_v RK45" (T on predicted v)
    #   step 1 vs paper:  runs/baseline_v + "... inputs_pki outputs_xy"  -> compare to Table "Step 1 (vx/vy)"
    # Defaults reproduce the previous hardcoded behaviour; override from the command line, e.g.
    #   python eval_metrics.py --model runs/baseline_v --data "../datasets_prep/dataset_giant_100hp_varyK inputs_pki outputs_xy"
    parser = argparse.ArgumentParser(description="Paper-style metrics for one trained model.")
    parser.add_argument("--model", type=Path,
                        default=Path("runs/baseline_T"))
    parser.add_argument("--data", type=Path,
                        default=Path("../datasets_prep/dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t prep_with_baseline_v RK45"))
    parser.add_argument("--destination", type=Path, default=None, help="default: the model directory")
    parser.add_argument("--scaling", action="store_true", help="evaluate on the scaling test data")
    parser.add_argument("--subdomain", action="store_true",
                        help="evaluate on the hardcoded sub-window only (preprocessing/subdomain.py); "
                             "match this to the run's own subdomain: setting")
    cli = parser.parse_args()
    subdomain.enable(cli.subdomain)
    print(subdomain.describe())
    PATH_PREP_DATA = cli.data
    PATH_MODEL = cli.model
    PATH_DESTINATION = cli.destination or PATH_MODEL
    SCALING = cli.scaling

    dataloaders, model, norm, info, args, output_channels = preparation(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION, scaling=SCALING)
    collected = collect_metrics(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION, dataloaders, model, norm, info, args, output_channels)
    plot_collected_metrics(collected, output_channels,
                           PATH_DESTINATION / f"metrics_plot_{PATH_MODEL.name} {PATH_PREP_DATA.name}.png")

    # same metric names/units as the e2e evaluation (measurements_test.yaml / metrics_RUN_x.yaml)
    styled = to_e2e_style(collected, output_channels)
    save_yaml(styled, PATH_DESTINATION / f"metrics_e2e-style_{PATH_MODEL.name} {PATH_PREP_DATA.name}.yaml")
    for case, values in styled.items():
        print(f"{case}: " + " | ".join(f"{k} {v:.4f}" for k, v in values.items()))
    compare_to_paper(styled, output_channels, data_name=PATH_PREP_DATA.name)

    # exemplary use on how to calculate mean, std, min and max for several runs of one set of hyperparameters
    # (needs a folder with metrics_run<id>.yaml files from repeated trainings - disabled by default)
    # file_name = lambda run_id: f"metrics_run{run_id}.yaml"
    # metrics = metrics_of_one_model(file_name, n_outputs=1, directory=Path("runs") / "modelA_several_runs", run_ids=[1,2,3,4,5])
    # plot_metrics(metrics, "Title")