"""
Inference for the end-to-end LGCNN: pki -> CNN1 -> streamlines -> CNN2 -> T in one forward pass.

Usage:
    python LGCNN_e2e_inference.py --run example_e2e [--datapoint 2] [--device cuda]

--run: name of the trained run dir under ../runs (needs model.pt + command_line_arguments.yaml
       + HPS_options.yaml). --datapoint: index into the prepared run ids (default 2 = test).
Saves T_pred_RUN_<id>.pt (in degC) and a comparison PNG into the run dir.
"""
import argparse
from pathlib import Path

import torch

from preprocessing.datasets.dataset import DataPointE2E
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.training import load_hyperparams
from processing.training_e2e import evaluate_e2e, visualize_e2e
from utils.utils_args import read_cla, save_yaml

PATH_DATA_PREP = Path("../../datasets_prep")  # TODO: change to your path
PATH_MODELS_DIR = Path("../runs")          # TODO: change to your path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--datapoint", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    cli = parser.parse_args()

    args = read_cla(PATH_MODELS_DIR / cli.run)
    assert args.get("pipeline") == "e2e", f"{cli.run} is not an end-to-end run"
    if cli.device is not None:
        args["device"] = cli.device
    if not torch.cuda.is_available() and "cuda" in args["device"]:
        print("CUDA not available, falling back to CPU")
        args["device"] = "cpu"
    args = load_hyperparams(args)

    dataset = DataPointE2E(PATH_DATA_PREP, args["data_raw"].name, i=cli.datapoint)
    runid = dataset.input_names[0]
    print(f"Predicting {runid} on {args['device']}")

    unet_args = dict(depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"],
                     stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"],
                     norm=args["norm"], repeat_inner=args["repeat_inner"])
    model = LGCNNEndToEnd(v_stats=dataset.info_v["Labels"], unet_args=unet_args,
                          randomK_data=args["randomK"], t_steps=args["t_steps"], sigma=args["sigma"],
                          use_compile=args.get("compile", False)).float()
    model.load(args["destination"], args["device"])
    model.eval()

    x, _ = dataset[0]
    with torch.no_grad():
        T_pred = model(x.unsqueeze(0).to(args["device"])).cpu()[0]

    # de-normalize to degC and save
    stats = dataset.info_T["Labels"]["Temperature [C]"]
    T_pred_degC = T_pred * (stats["max"] - stats["min"]) + stats["min"]
    out_pt = args["destination"] / f"T_pred_{runid}"
    torch.save(T_pred_degC, out_pt)
    print(f"Saved prediction ({tuple(T_pred_degC.shape)}, degC) to {out_pt}")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1)
    metrics = evaluate_e2e(model, dataloader, dataset.info_T, args["device"])
    print("Metrics vs label:", *[f"  {k}: {v:.4f}" for k, v in metrics.items()], sep="\n")
    save_yaml(metrics, args["destination"] / f"metrics_{runid.replace('.pt', '')}.yaml")
    visualize_e2e(model, dataloader, args, plot_path=args["destination"] / f"inference_{runid.replace('.pt', '')}.png")
