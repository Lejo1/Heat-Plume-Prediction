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

from preprocessing.datasets.dataset import DataPoint, DataPointE2E
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from processing.e2e_pipeline_plot import PipelineTap
from processing.loss_fcts import E2ELoss, SSIMLoss, PATLoss
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.networks.model import weights_init
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
    # CNN2 may need a different architecture than CNN1 - the two baselines were tuned separately.
    # unet_args_T in command_line_arguments.yaml overrides individual keys, e.g. {kernel_size: 4}.
    unet_args_T = {**unet_args, **(args.get("unet_args_T") or {})}
    if unet_args_T != unet_args:
        print(f"CNN2 architecture overrides: {args['unet_args_T']}")
    model = LGCNNEndToEnd(v_stats=dataset_train.info_v["Labels"], unet_args=unet_args,
                          unet_args_T=unet_args_T,
                          randomK_data=args["randomK"], t_steps=args["t_steps"], sigma=args["sigma"],
                          use_compile=args.get("compile", False),
                          fade_mode=args.get("fade_mode", "absolute"),
                          detach_direct_v=args.get("detach_direct_v", False),
                          v_blur=args.get("v_blur", 0.0) or 0.0).float()
    model.to(args["device"])

    if args["case"] in ["test", "finetune"]:
        # two sources of weights: either one end-to-end checkpoint (model:) or the two separately
        # trained baselines (model_v: + model_T:), which is how a two-stage run is continued e2e
        if args.get("model_v") and args.get("model_T"):
            print(f"Loading the two baselines for end-to-end {args['case']}:")
            model.load_baselines(args["model_v"], args["model_T"], args["device"])
        else:
            assert args.get("model"), "case 'finetune'/'test' needs either model: or model_v: + model_T:"
            model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()

    if args["case"] == "train":
        # initialize both UNets ONCE here; the Solvers below run with finetune=True so that
        # stage 2 does not wipe the stage-1 weights of unet_v
        model.apply(weights_init)

    # STAGE 1: pretrain CNN1 (unet_v) alone on the velocity task with the partitioned dataset
    # (~1000 patch-batches per epoch instead of 1 full-domain step; no streamlines/CNN2 -> cheap).
    # The trained weights are cached to unet_v_pretrained.pt and reused on later runs, so the
    # ~35 min pretraining runs only once (delete the file or set stage1_force: true to retrain).
    if args["case"] == "train" and args.get("stage1_epochs", 0) > 0:
        stage1_dir = args["destination"] / "stage1_v"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage1_ckpt = stage1_dir / "unet_v_pretrained.pt"
        val_v = DataPoint(dataset_train.path_v, i=order[1])

        if stage1_ckpt.exists() and not args.get("stage1_force", False):
            print(f"STAGE 1: loading cached pretrained unet_v from {stage1_ckpt} "
                  f"(skipping the {args['stage1_epochs']}-epoch pretraining; "
                  f"delete the file or set stage1_force: true to retrain)")
            model.unet_v.load_state_dict(torch.load(stage1_ckpt, map_location=args["device"]))
        else:
            cuts_train = SimulationDatasetCuts(dataset_train.path_v, skip_per_dir=args["skip_per_dir"],
                                               box_size=args["len_box"], ids=order[0])
            dataloaders_v = {"train": DataLoader(cuts_train, batch_size=args["batchsize"], shuffle=True, num_workers=0),
                             "val": DataLoader(val_v, batch_size=1, shuffle=False, num_workers=0)}
            args_stage1 = {**args, "epochs": args["stage1_epochs"], "destination": stage1_dir}

            solver_v = Solver(model.unet_v, dataloaders_v["train"], dataloaders_v["val"],
                              loss_func=MSELoss(), finetune=True, learning_rate=args["lr"])
            solver_v.lr_schedule = {0: args["lr"]}
            print(f"STAGE 1: velocity pretraining, {len(dataloaders_v['train'])} patch-batches/epoch, {args['stage1_epochs']} epochs")
            stage1_time = datetime.now()
            try:
                solver_v.train(args_stage1)
            except KeyboardInterrupt:
                logging.warning(f"Manually stopping stage 1 early with best model found in epoch {solver_v.best_model_params['epoch']}.")
            finally:
                solver_v.save_lr_schedule(stage1_dir / "learning_rate_history.csv")
            print(f"STAGE 1 finished after {datetime.now()-stage1_time}")
            # solver_v.train() has already loaded the best unet_v weights back into the model
            torch.save(model.unet_v.state_dict(), stage1_ckpt)
            print(f"Saved stage-1 unet_v checkpoint to {stage1_ckpt}")
            # release stage-1 memory (Solver, its Adam moments, the cut dataset/loaders) before the
            # memory-heavy full-domain diagnostic and stage-2 forward
            del solver_v, cuts_train, dataloaders_v
            if "cuda" in str(args["device"]):
                torch.cuda.empty_cache()

        print_velocity_mae(model.unet_v, val_v, dataset_train.info_v, args["device"])
        del val_v
        if "cuda" in str(args["device"]):
            torch.cuda.empty_cache()

    elif args["case"] == "finetune":
        # sanity gate on the loaded CNN1: a wrong checkpoint or mismatched normalization stats show
        # up here as a wildly off MAE, before any streamline or CNN2 issue can confuse the picture
        val_v = DataPoint(dataset_train.path_v, i=order[1])
        print("Loaded CNN1 check (paper step-1 reference: 22.3 / 32.7 m/y):")
        print_velocity_mae(model.unet_v, val_v, dataset_train.info_v, args["device"])
        del val_v
        if "cuda" in str(args["device"]):
            torch.cuda.empty_cache()

    # BATCHNORM RE-ESTIMATION: CNN1 reaches this point trained on 256^2 patches (batch 20) - by
    # stage-1 pretraining in the "train" case, by baseline_v in the "finetune" case - but from here
    # on the model only ever sees one full-domain sample at a time. Recompute the buffers on
    # full-domain inputs before anything is measured, so the starting-point plots/metrics and the
    # gradient diagnostic describe the model stage 2 actually starts from. Weights are untouched.
    # (In the "train" case CNN2 is still randomly initialized, so only CNN1 gains anything here;
    # CNN2's buffers are overwritten by stage 2 either way.)
    if args.get("bn_reestimate", False) and args["case"] in ["train", "finetune"]:
        print("Re-estimating BatchNorm statistics on full-domain inputs (weights unchanged):")
        n_bn, n_passes = reestimate_bn_stats(model, dataloaders["train"], args["device"],
                                             repeats=args.get("bn_reestimate_repeats", 1))
        print(f"  {n_bn} BatchNorm layers updated from {n_passes} full-domain forward pass(es)")
        # same velocity check as above, now on the re-estimated buffers: the two prints bracket the
        # effect of the recalibration on CNN1 (paper step-1 reference: 22.3 / 32.7 m/y)
        val_v = DataPoint(dataset_train.path_v, i=order[1])
        print("CNN1 after re-estimation:")
        print_velocity_mae(model.unet_v, val_v, dataset_train.info_v, args["device"])
        del val_v
        if "cuda" in str(args["device"]):
            torch.cuda.empty_cache()

    # STARTING-POINT PLOTS: the same panels as the end-of-run val_e2e/test_e2e plots, rendered on
    # the model as it enters stage 2. For a finetune that is the pristine two-stage baseline
    # pipeline (baseline_v + hard-drawing-trained baseline_T now fed soft streamlines), so the
    # _start/final pair isolates exactly what the joint training changed - and shows up front how
    # much of any difference is CNN2 reacting to the new streamline channel rather than to training.
    # Placed before the gradient diagnostic so grad_diag_only still produces it.
    if args.get("visualize", False) and args["case"] in ["train", "finetune"]:
        print("Rendering the stage-2 starting point (compare against val_e2e.png / test_e2e.png):")
        visualize_e2e(model, dataloaders["val"], args, plot_path=args["destination"] / "val_e2e_start.png")
        if "test" in dataloaders:
            # same metrics as the end-of-run measurements_test.yaml, so the two files subtract
            metrics_start = evaluate_e2e(model, dataloaders["test"], dataset_train.info_T, args["device"],
                                         info_v=dataset_train.info_v)
            save_yaml(metrics_start, args["destination"] / "measurements_test_start.yaml")
            print("Test-set metrics BEFORE stage 2:",
                  *[f"  {k}: {v:.4f}" for k, v in metrics_start.items()], sep="\n")
            visualize_e2e(model, dataloaders["test"], args, plot_path=args["destination"] / "test_e2e_start.png")
        if "cuda" in str(args["device"]):
            torch.cuda.empty_cache()

    # GRADIENT DIAGNOSTIC: how much of CNN1's update comes through the differentiable streamlines
    # vs. the direct velocity channels, evaluated on the *stage-2 starting* model (post stage 1 =
    # "first epoch"). grad_diag_only stops here so the plot is available without the full run.
    if args.get("grad_diag", False) and args["case"] in ["train", "finetune"]:
        from processing.e2e_grad_diag import diagnose_e2e_gradients
        x0, y0 = next(iter(dataloaders["train"]))
        diagnose_e2e_gradients(model, x0, y0, args, plot_path=args["destination"] / "grad_diag_stage2_start.png")
        if args.get("grad_diag_only", False):
            return model

    # STAGE 2: joint training of the full pipeline (CNN1 + streamlines + CNN2), full domain
    if args["case"] in ["train", "finetune"]:
        # loss = MSE(T) + lambda_v * MSE(v); lambda_v=0 (or missing key) = pure temperature loss.
        # clip_grad_norm caps exploding gradients from backprop through the chaotic advection.
        # optional per-step pipeline diagnostic: every N optimizer steps, save each stage's
        # input/output and the loss gradients on both sides of it (pipeline_plot_every: 0 = off)
        tap = PipelineTap(every=args.get("pipeline_plot_every", 0),
                          destination=args["destination"],
                          max_plots=args.get("pipeline_plot_max", 0),
                          t_stats=dataset_train.info_T["Labels"]["Temperature [C]"])
        if tap.enabled:
            print(f"pipeline diagnostic: every {tap.every} optimizer steps"
                  + (f", at most {tap.max_plots} plots" if tap.max_plots > 0 else "")
                  + f" -> {tap.dir}")

        solver = Solver(model, dataloaders["train"], dataloaders["val"],
                        loss_func=E2ELoss(lambda_v=args.get("lambda_v", 0.0)),
                        finetune=True, learning_rate=args["lr"],
                        clip_grad_norm=args.get("clip_grad", 1.0),
                        pipeline_tap=tap if tap.enabled else None,
                        epoch_callback=make_v_blur_schedule(model, args))
        # programmatic schedule instead of load_lr_schedule: the fallback default_lr_schedule.csv
        # would silently drop the lr to 1e-5 at epoch 100, stale history files would be re-applied.
        # lr_decay re-enables the 10x drop after 70% of the epochs; off by default, so the lr stays
        # constant for the whole of stage 2.
        solver.lr_schedule = {0: args["lr"]}
        if args.get("lr_decay", False):
            solver.lr_schedule[int(0.7 * args["epochs"])] = args["lr"] / 10
        print(f"STAGE 2 lr: {args['lr']:.2e}"
              + (f", dropping to {args['lr']/10:.2e} at epoch {int(0.7 * args['epochs'])}"
                 if args.get("lr_decay", False) else " (constant)"))
        print(f"STAGE 2: joint end-to-end training, {args['epochs']} epochs")
        training_time = datetime.now()
        try:
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history_stage2.csv")
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


def reestimate_bn_stats(model, dataloader, device, repeats: int = 1):
    """Recompute every BatchNorm's running_mean/running_var from full-domain inputs.

    The nets are fully convolutional, so patch-trained weights apply to the full domain unchanged -
    but the BatchNorm buffers do not: they were accumulated over batches of twenty 256^2 patches,
    while inference sees a single 2560^2 sample. That mismatch is not cosmetic. Swapping these
    buffers alone accounted for ~92% of the velocity-field tilt difference across a whole finetune
    run (baseline -9.16 deg, finetuned -4.49 deg, finetuned weights + baseline buffers -8.76 deg,
    simulated truth -2.84 deg), because the streamline tracer integrates a small coherent velocity
    offset along each trajectory while an MSE barely registers it.

    momentum=None makes each layer accumulate an exact cumulative average over the passes below
    rather than an exponential one, so the result does not depend on pass order and one pass over a
    single datapoint yields exactly that datapoint's statistics. Runs under no_grad in train mode:
    only the buffers change, no weights and no optimizer state.

    Returns (number of BatchNorm layers, number of forward passes).
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0, 0
    saved_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    n_passes = 0
    with torch.no_grad():
        for _ in range(max(int(repeats), 1)):
            for x, _ in dataloader:
                model(x.to(device))
                n_passes += 1
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    model.train(was_training)
    return len(bns), n_passes


def make_v_blur_schedule(model, args):
    """Annealing of the tracer's velocity smoothing: fn(epoch) -> None, or None if not needed.

    v_blur widens the support of dL/dv but also changes the forward model (trajectories follow the
    coarse-grained flow). Annealing it to v_blur_end resolves that: the run starts with a broad,
    well-conditioned gradient and finishes tracing the field it will be evaluated on - a
    continuation scheme, not a permanent approximation.
    """
    start = float(args.get("v_blur", 0.0) or 0.0)
    end = args.get("v_blur_end")
    if start <= 0 and end in (None, 0, 0.0):
        return None
    end = start if end is None else float(end)
    n = int(args.get("v_blur_anneal_epochs") or args["epochs"])
    if end == start:
        print(f"tracer velocity smoothing: v_blur = {start} (constant)")
        return None  # model.v_blur is already set from the config; no per-epoch work to do
    print(f"tracer velocity smoothing: v_blur {start} -> {end} linearly over {n} epochs")

    def set_v_blur(epoch):
        frac = min(epoch / max(n - 1, 1), 1.0)
        model.v_blur = start + (end - start) * frac
    return set_v_blur


def print_velocity_mae(unet_v, dataset_val, info_v, device):
    """Stage-1 diagnostic: velocity MAE on the val datapoint in [m/y] (paper Step-1: 22.3/32.7)."""
    unet_v.eval()
    x, y = dataset_val[0]
    with torch.no_grad():
        v_pred = unet_v(x.unsqueeze(0).to(device)).cpu()
    h, w = v_pred.shape[2:]
    i0, j0 = (y.shape[1] - h) // 2, (y.shape[2] - w) // 2
    y_crop = y[:, i0:i0+h, j0:j0+w].unsqueeze(0)
    for ch, name in [(0, "Liquid X-Velocity [m_per_y]"), (1, "Liquid Y-Velocity [m_per_y]")]:
        delta = info_v["Labels"][name]["max"] - info_v["Labels"][name]["min"]
        print(f"  stage-1 val MAE v{'x' if ch == 0 else 'y'}: {float((v_pred[:, ch] - y_crop[:, ch]).abs().mean() * delta):.1f} m/y")


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
