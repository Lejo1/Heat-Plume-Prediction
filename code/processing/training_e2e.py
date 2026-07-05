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
from torch.nn import MSELoss, L1Loss, HuberLoss
from torch.utils.data import DataLoader

from preprocessing.datasets.dataset import DataPointE2E
from processing.loss_fcts import E2ELoss, SSIMLoss, PATLoss
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.solver import Solver
from processing.training import load_hyperparams
from utils.utils_args import save_yaml


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
        # loss = MSE(T) + lambda_v * MSE(v); lambda_v=0 (or missing key) = pure temperature loss.
        # clip_grad_norm caps exploding gradients from backprop through the chaotic advection.
        solver = Solver(model, dataloaders["train"], dataloaders["val"],
                        loss_func=E2ELoss(lambda_v=args.get("lambda_v", 0.0)),
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

    if "test" in dataloaders:
        metrics_test = evaluate_e2e(model, dataloaders["test"], dataset_train.info_T, args["device"],
                                    info_v=dataset_train.info_v)
        save_yaml(metrics_test, args["destination"] / "measurements_test.yaml")
        print("Test-set metrics:", *[f"  {k}: {v:.4f}" for k, v in metrics_test.items()], sep="\n")
        visualize_e2e(model, dataloaders["test"], args, plot_path=args["destination"] / "test_e2e.png")
    return model


def evaluate_e2e(model, dataloader, info_T, device, info_v=None):
    """Paper-style metrics of the full e2e pipeline. Temperature metrics are computed in [degC]
    (de-normalized), SSIM on the normalized fields (same convention as eval_metrics.py).
    With info_v given, also reports the velocity MAE in [m/y] (paper Step-1 ref: 22.3 / 32.7)."""
    stats = info_T["Labels"]["Temperature [C]"]
    delta = stats["max"] - stats["min"]
    mse, mae, huber = MSELoss(), L1Loss(), HuberLoss()
    model.eval()
    collected = []
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x.to(device)).cpu()
            h, w = pred.shape[2:]
            i0, j0 = (y.shape[2] - h) // 2, (y.shape[3] - w) // 2
            y_crop = y[:, :, i0:i0+h, j0:j0+w]
            T_pred, T_label = pred[:, 0:1], y_crop[:, 0:1]
            pred_C, label_C = T_pred * delta + stats["min"], T_label * delta + stats["min"]
            metrics = {
                "MAE [degC]": float(mae(pred_C, label_C)),
                "RMSE [degC]": float(mse(pred_C, label_C)) ** 0.5,
                "Huber [degC]": float(huber(pred_C, label_C)),
                "Linf [degC]": float((pred_C - label_C).abs().max()),
                "SSIM": float(SSIMLoss()(T_pred, T_label)),
                "PAT0.1 [%]": float(PATLoss([0.1])(pred_C, label_C).mean()),
                "PAT1.0 [%]": float(PATLoss([1.0])(pred_C, label_C).mean()),
            }
            if info_v is not None:
                for ch, name in [(1, "Liquid X-Velocity [m_per_y]"), (2, "Liquid Y-Velocity [m_per_y]")]:
                    v_stats = info_v["Labels"][name]
                    v_delta = v_stats["max"] - v_stats["min"]
                    metrics[f"MAE v{'x' if ch == 1 else 'y'} [m/y]"] = float(
                        mae(pred[:, ch:ch+1] * v_delta, y_crop[:, ch:ch+1] * v_delta))
            collected.append(metrics)
    return {name: sum(m[name] for m in collected) / len(collected) for name in collected[0]}


def visualize_e2e(model, dataloader, args, plot_path: Path):
    """3x3 panels: T (prediction/label/error), vx and vy (prediction/simulated label), the
    velocity error magnitude and the traced streamlines. Prediction/label share one color
    scale for T; all four velocity panels share one scale."""
    model.eval()
    x, y = next(iter(dataloader))
    with torch.no_grad():
        pred = model(x.to(args["device"])).cpu()
    sf = model.last_intermediates["sf"].cpu()[0, 0]
    sf_outer = model.last_intermediates["sf_outer"].cpu()[0, 0]

    # center-crop the label to the prediction; de-normalize T to [degC]
    h, w = pred.shape[2:]
    i0, j0 = (y.shape[2] - h) // 2, (y.shape[3] - w) // 2
    y_crop = y[:, :, i0:i0+h, j0:j0+w].clone()
    stats = dataloader.dataset.info_T["Labels"]["Temperature [C]"]
    to_degC = lambda t: t * (stats["max"] - stats["min"]) + stats["min"]
    T_pred, T_label = to_degC(pred[0, 0]), to_degC(y_crop[0, 0])
    v_pred, v_label = pred[0, 1:3], y_crop[0, 1:3]

    # shared color scales, so prediction and label panels are directly comparable
    T_lim = (min(float(T_pred.min()), float(T_label.min())), max(float(T_pred.max()), float(T_label.max())))
    v_lim = (min(float(v_pred.min()), float(v_label.min())), max(float(v_pred.max()), float(v_label.max())))
    v_err = (v_pred - v_label).norm(dim=0)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    panels = [(T_pred, "prediction T [degC]", "RdBu_r", T_lim), (T_label, "label T [degC]", "RdBu_r", T_lim),
              ((T_pred - T_label).abs(), "abs error T [degC]", "magma", None),
              (v_pred[0], "predicted vx (normed)", "viridis", v_lim), (v_label[0], "simulated vx (normed)", "viridis", v_lim),
              (v_err, "velocity error |v_pred - v_sim| (normed)", "magma", None),
              (v_pred[1], "predicted vy (normed)", "viridis", v_lim), (v_label[1], "simulated vy (normed)", "viridis", v_lim),
              (sf + sf_outer, "streamlines (soft, sf + sf_outer)", "magma", None)]
    for ax, (data, title, cmap, lim) in zip(axes.flat, panels):
        vmin, vmax = lim if lim is not None else (None, None)
        im = ax.imshow(data.numpy().T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")
