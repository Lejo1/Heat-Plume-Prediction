"""Streamlines of CNN1 on the test field with the BatchNorm statistics of the patch training and re-estimated on the
training field, and of the simulated velocity (appendix BatchNorm, fig:app-bn-tilt; plotted by bn_tilt.py).

Start state of an e2e run config (model_v + model_T): CNN1's velocity on the test datapoint once with the running
statistics of the patch training and once after reestimate_bn_stats on the training datapoint, the heat-pump lines
traced in both and in the simulated velocity (trace_streamlines, the run's tracer settings), and the mean-flow tilt
atan(mean vx / mean vy) [deg]. Needs the GPU (full field): run on pcsgs11 from a code copy, e.g.
    cd <code copy> && PYTHONPATH=. python postprocessing/thesis/bn_tilt_lines.py <config dir> <out.pt>
<config dir> holds command_line_arguments.yaml + HPS_options.yaml (model paths relative to code/ or absolute).
Saves {state: {"lines": [[x, y] per heat pump, every 10th point, cells of CNN1's output region], "tilt_deg",
"tilt_deg_Tregion", "vx_mae", "vy_mae", "T_mae" (T output region, as evaluate_e2e)}, "shape"} for the states
patch / reestimated / simulated (lines and tilt only).
Results 2026-11-07 in forPaper/A_batchnorm/: lines_own.pt (our early patch-trained models runs/baseline_v_old +
baseline_T_old, the models of runs/olds/finetune_e2e_first), lines_best.pt (BEST_V + BEST_T, config of
E3_finetune/vblur_bn/vbn_blur0_bn0)."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import eval_scaling_e2e as E
from processing.training_e2e import build_e2e_model, reestimate_bn_stats
from step2_streamlines.streamlines_helpers import trace_streamlines

cfg_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
args = E.read_cla(cfg_dir)
args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
args = E.load_hyperparams(args)
dev = args["device"]
name = Path(args["data_raw"]).name
ds_train = E.DataPointE2E(E.PATH_DATA_PREP, name, i=args["order_data"][0])
ds_test = E.DataPointE2E(E.PATH_DATA_PREP, name, i=args["order_data"][2])
model = build_e2e_model(args, ds_train.info_v["Labels"], inference=True).to(dev)
model.load_baselines(E.from_code(args["model_v"]), E.from_code(args["model_T"]), dev)

x, y = ds_test[0]
x, y = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
crop = lambda t, h, w: t[:, :, (t.shape[2] - h) // 2:(t.shape[2] - h) // 2 + h, (t.shape[3] - w) // 2:(t.shape[3] - w) // 2 + w]


def lines_of(v, hp, shape):
    with torch.no_grad():
        ls = trace_streamlines(hp, v[0], v[1], shape, randomK_data=model.randomK_data, t_steps=model.t_steps)
    return [np.stack([lx[::10].cpu().numpy(), ly[::10].cpu().numpy()]).astype(np.float32) for lx, ly, _ in ls]


def tilt(v):
    return float(np.degrees(np.arctan2(float(v[0].mean()), float(v[1].mean()))))


res = {}
for state in ("patch", "reestimated"):
    if state == "reestimated":
        n_bn, n_pass = reestimate_bn_stats(model, DataLoader(ds_train, batch_size=1), dev)
        print(f"re-estimated {n_bn} BatchNorm layers from {n_pass} pass(es) on the training field")
    model.eval()
    with torch.no_grad():
        v_norm = model.unet_v(x)
        pred = model(x)   # T output region: the region of the stored velocity and T metrics (evaluate_e2e)
    h, w = v_norm.shape[2:]
    ht, wt = pred.shape[2:]
    v = (v_norm * model.v_delta + model.v_min)[0]
    v_sim = (crop(y, h, w)[:, 1:3] * model.v_delta + model.v_min)[0]
    hp = model.heat_pump_positions(crop(x, h, w)[0, model.IDX_I])
    vt, vst = crop(v[None], ht, wt)[0], crop(v_sim[None], ht, wt)[0]
    st = ds_train.info_T["Labels"]["Temperature [C]"]
    res[state] = {"lines": lines_of(v, hp, (h, w)), "tilt_deg": tilt(v), "tilt_deg_Tregion": tilt(vt),
                  "vx_mae": float((vt[0] - vst[0]).abs().mean()), "vy_mae": float((vt[1] - vst[1]).abs().mean()),
                  "T_mae": float((pred[0, 0] - crop(y, ht, wt)[0, 0]).abs().mean()) * (st["max"] - st["min"])}
    print(f"{state}: tilt {res[state]['tilt_deg']:.2f} deg (T region {res[state]['tilt_deg_Tregion']:.2f}), "
          f"vx MAE {res[state]['vx_mae']:.2f}, vy MAE {res[state]['vy_mae']:.2f} m/y, T MAE {res[state]['T_mae']:.4f}")
res["simulated"] = {"lines": lines_of(v_sim, hp, (h, w)), "tilt_deg": tilt(v_sim)}
res["shape"] = (h, w)
print(f"simulated: tilt {res['simulated']['tilt_deg']:.2f} deg; {len(hp)} heat pumps, region {h}x{w}")
torch.save(res, out)
print(f"saved {out}")
