"""Scaling test of an end-to-end run: the run's model on the 5120^2 field of giant_double_100hp_v2
(4x the area of the 2560^2 training field, same pressure gradient, cell size and pump rate; the
"scaling" row of the paper's tables).

Usage (from code/):
    python eval_scaling_e2e.py --run ../runs/finetune_e2e_frame [--device cuda] [--force]

--run: the run dir (needs model.pt + command_line_arguments.yaml + HPS_options.yaml), or its name
       under ../runs.
Writes into <run>/scaling_test/:
    measurements_scaling.yaml        same metrics as measurements_test.yaml (evaluate_e2e)
    scaling_e2e.png                  same panels as test_e2e.png (visualize_e2e)
    measurements_scaling_start.yaml  the state before stage 2 (model_v + model_T loaded, BatchNorm
    scaling_e2e_start.png            re-estimated if the run did), only for runs with model_v + model_T
Existing results are kept unless --force is given.

On first use the two scaling prep folders are created from the raw dataset (PATH_DATA_RAW), normalized
with the info.yaml of the run's training prep folders, i.e. with the training normalization.
Normalization stats, eps of the frame metrics (p5 of |v*|) and the well parameters stay those of the
training datapoint; the random frame-metric seeds are drawn in the bounding box of the scaling field's
own heat pumps (SEED_BOX_RAW is the pump box of the 2560^2 field).
"""
import argparse
import os
from pathlib import Path

# one 5120^2 forward peaks at ~18 GB; without expandable segments the second state (start) fails on a
# 24 GB card from fragmentation left by the first. Read when CUDA allocates first, so set before that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

from preprocessing import subdomain
from preprocessing.datasets.dataset import DataPointE2E
from preprocessing.preprocessing import is_unprepared, preprocessing
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.training import load_hyperparams
from processing.training_e2e import (build_e2e_model, build_frame_losses, evaluate_e2e, reestimate_bn_stats,
                                     visualize_e2e)
from utils.utils_args import read_cla, save_yaml

CODE_DIR = Path(__file__).resolve().parent
PATH_DATA_PREP = CODE_DIR / "../../datasets_prep"  # TODO: change to your path (same as main.py)
PATH_DATA_RAW = CODE_DIR / "../../datasets"        # TODO: change to your path
PATH_MODELS_DIR = CODE_DIR / "../runs"
SCALING_DATASET = "giant_double_100hp_v2"
OUT_SUBDIR = "scaling_test"
# the two prep folders DataPointE2E joins: (inputs, outputs, folder suffix)
PREP_KINDS = (("pki", "xy", ""), ("ixyk", "t", " for_s"))


def from_code(p) -> Path:
    """Config paths (model_v, model_T, ...) are written relative to code/, where main.py runs."""
    p = Path(p)
    return p if p.is_absolute() else CODE_DIR / p


def ensure_scaling_prep(train_dataset: str, scaling_dataset: str = SCALING_DATASET,
                        prep_root: Path = PATH_DATA_PREP, raw_root: Path = PATH_DATA_RAW):
    """Create the scaling prep folders if they are missing, normalized with the training prep's info.yaml."""
    for inputs, outputs, suffix in PREP_KINDS:
        target = Path(prep_root) / f"{scaling_dataset} inputs_{inputs} outputs_{outputs}{suffix}"
        if not is_unprepared(target):
            continue
        reference = Path(prep_root) / f"{train_dataset} inputs_{inputs} outputs_{outputs}{suffix}"
        assert (reference / "info.yaml").exists(), f"{reference / 'info.yaml'} not found (training normalization)"
        raw = Path(raw_root) / scaling_dataset
        assert raw.exists(), f"raw scaling dataset {raw} not found"
        print(f"Preparing {target.name} from {raw} with the normalization of {reference.name}")
        # case "test": prepare_dataset normalizes with args["model"]/info.yaml instead of new statistics
        preprocessing({"inputs": inputs, "outputs": outputs, "case": "test", "model": reference,
                       "data_raw": raw, "data_prep": target, "destination": None})


def seed_box_of_pumps(x: torch.Tensor):
    """Raw-cell bounding box ((a0, b0), (a1, b1)) of the heat pumps of a pki input [3, H, W]."""
    cells = torch.floor(LGCNNEndToEnd.heat_pump_positions(x[LGCNNEndToEnd.IDX_I])).long()
    assert cells.shape[0] > 0, "no heat pumps in the scaling datapoint"
    return tuple((int(cells[:, ax].min()), int(cells[:, ax].max())) for ax in range(2))


def resolve_run(run) -> Path:
    run = Path(run)
    return run if run.exists() else PATH_MODELS_DIR / run


def evaluate_run(run_dir, device: str = None, force: bool = False) -> dict:
    """Scaling metrics of one e2e run: {"final": {...}, "start": {...} (finetune runs only)}.
    Writes them (and the panel plots) into <run>/scaling_test/; existing files are read back instead
    of recomputed unless force."""
    run_dir = resolve_run(run_dir)
    args = read_cla(run_dir)
    assert args.get("pipeline") == "e2e", f"{run_dir} is not an end-to-end run"
    assert not subdomain.is_enabled(), "scaling test of a subdomain run: the window is defined on the 2560^2 field"
    if device is not None:
        args["device"] = device
    if "cuda" in args["device"] and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args["device"] = "cpu"
    args = load_hyperparams(args)
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(exist_ok=True)

    states = ["final"] + (["start"] if args.get("model_v") and args.get("model_T") else [])
    files = {s: out_dir / f"measurements_scaling{'' if s == 'final' else '_start'}.yaml" for s in states}
    if not force and all(f.exists() for f in files.values()):
        import yaml
        print(f"{run_dir.name}: scaling results exist in {out_dir} (--force to recompute)")
        return {s: yaml.safe_load(f.read_text()) for s, f in files.items()}

    train_name = Path(args["data_raw"]).name
    ensure_scaling_prep(train_name)
    dataset_train = DataPointE2E(PATH_DATA_PREP, train_name, i=args["order_data"][0])
    dataset_scaling = DataPointE2E(PATH_DATA_PREP, SCALING_DATASET, i=0)
    loader = DataLoader(dataset_scaling, batch_size=1, shuffle=False, num_workers=0)
    x_s, y_s = dataset_scaling[0]
    print(f"scaling datapoint {dataset_scaling.input_names[0]}: field {tuple(y_s.shape[1:])}")

    model = build_e2e_model(args, dataset_train.info_v["Labels"], inference=True)
    model.to(args["device"])
    box = seed_box_of_pumps(x_s)
    frames = build_frame_losses(args, model, dataset_train, PATH_DATA_PREP,
                                eval_field={"field_shape": tuple(y_s.shape[1:]), "seed_box_raw": box})
    print(f"random frame-metric seeds in the scaling pump box {box}")
    del x_s, y_s

    results = {}
    for state in states:
        if state == "final":
            model.load(run_dir, args["device"])
        else:
            print("start state: the two baselines before stage 2")
            model.load_baselines(from_code(args["model_v"]), from_code(args["model_T"]), args["device"])
            if args.get("bn_reestimate", False):
                n_bn, n_passes = reestimate_bn_stats(model, DataLoader(dataset_train, batch_size=1), args["device"],
                                                     repeats=args.get("bn_reestimate_repeats", 1))
                print(f"  re-estimated {n_bn} BatchNorm layers from {n_passes} pass(es)")
        model.eval()
        if "eps" not in frames["evals"]["hp"].shared:
            # eps (frame_eps "pX") is a percentile over the training datapoint's output region
            x_t, _ = dataset_train[0]
            with torch.no_grad():
                region = tuple(model(x_t.unsqueeze(0).to(args["device"])).shape[2:])
            frames["evals"]["hp"]._resolve_eps(region)
            del x_t
        metrics = evaluate_e2e(model, loader, dataset_train.info_T, args["device"],
                               info_v=dataset_train.info_v, frame_evals=frames["evals"])
        save_yaml(metrics, files[state])
        print(f"Scaling metrics ({state}):", *[f"  {k}: {v:.4f}" for k, v in metrics.items()], sep="\n")
        visualize_e2e(model, loader, args,
                      plot_path=out_dir / f"scaling_e2e{'' if state == 'final' else '_start'}.png")
        results[state] = metrics
        if "cuda" in args["device"]:
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling test (giant_double_100hp_v2) of an end-to-end run.")
    parser.add_argument("--run", type=str, required=True, help="run dir, or its name under ../runs")
    parser.add_argument("--device", type=str, default=None, help="default: the run's device")
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    cli = parser.parse_args()
    evaluate_run(cli.run, device=cli.device, force=cli.force)
