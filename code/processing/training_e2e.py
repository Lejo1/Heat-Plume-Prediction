import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from preprocessing.datasets.dataset import DataPointE2E
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.solver import Solver
from processing.training import load_hyperparams


def training_e2e(args: Dict, PATH_DATA_PREP: Path):
    """End-to-end LGCNN training: CNN1 + differentiable streamlines + CNN2, trained jointly
    from scratch on the temperature loss only. Full-domain samples, batch size 1."""
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    args = load_hyperparams(args)
    dataset_name = args["data_raw"].name

    order = args["order_data"]
    dataset_train = DataPointE2E(PATH_DATA_PREP, dataset_name, i=order[0])
    dataset_val = DataPointE2E(PATH_DATA_PREP, dataset_name, i=order[1])
    dataloaders = {"train": DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0),
                   "val": DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)}
    try:
        dataset_test = DataPointE2E(PATH_DATA_PREP, dataset_name, i=order[2])
        dataloaders["test"] = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    except IndexError:
        pass

    unet_args = dict(depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"],
                     stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"],
                     norm=args["norm"], repeat_inner=args["repeat_inner"])
    model = LGCNNEndToEnd(v_stats=dataset_train.info_v["Labels"], unet_args=unet_args,
                          randomK_data=args["randomK"], t_steps=args["t_steps"], sigma=args["sigma"],
                          use_compile=args.get("compile", False)).float()
    model.to(args["device"])

    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()

    if args["case"] in ["train", "finetune"]:
        # clip_grad_norm caps exploding gradients from backprop through the chaotic advection
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=MSELoss(),
                        finetune=(args["case"] == "finetune"), learning_rate=args["lr"],
                        clip_grad_norm=args.get("clip_grad", 1.0))
        training_time = datetime.now()
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")

        training_time = datetime.now() - training_time
        model.save(args["destination"])
        solver.save_metrics_separate_yaml(args["destination"], model.num_of_params(), args["epochs"], training_time.total_seconds(), args["device"])

    visualize_e2e(model, dataloaders["val"], args, plot_path=args["destination"] / "val_e2e.png")
    return model


def visualize_e2e(model, dataloader, args, plot_path: Path):
    """Panels: prediction T / label T / error, plus the intermediates v and streamlines."""
    model.eval()
    x, y = next(iter(dataloader))
    with torch.no_grad():
        y_pred = model(x.to(args["device"])).cpu()
    v_norm = model.last_intermediates["v_norm"].cpu()[0]
    sf = model.last_intermediates["sf"].cpu()[0, 0]
    sf_outer = model.last_intermediates["sf_outer"].cpu()[0, 0]

    # center-crop the label to the prediction and de-normalize both to [degC]
    h, w = y_pred.shape[2:]
    i0, j0 = (y.shape[2] - h) // 2, (y.shape[3] - w) // 2
    y_crop = y[:, :, i0:i0+h, j0:j0+w].clone()
    stats = dataloader.dataset.info_T["Labels"]["Temperature [C]"]
    to_degC = lambda t: t * (stats["max"] - stats["min"]) + stats["min"]
    T_pred, T_label = to_degC(y_pred[0, 0]), to_degC(y_crop[0, 0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panels = [(T_pred, "prediction T [degC]", "RdBu_r", None), (T_label, "label T [degC]", "RdBu_r", None),
              ((T_pred - T_label).abs(), "abs error [degC]", "magma", None),
              (v_norm[0], "predicted vx (normed)", "viridis", None), (v_norm[1], "predicted vy (normed)", "viridis", None),
              (sf + sf_outer, "streamlines (soft, sf + sf_outer)", "magma", None)]
    for ax, (data, title, cmap, _) in zip(axes.flat, panels):
        im = ax.imshow(data.numpy().T, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")
