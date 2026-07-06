"""
Smoke test for the end-to-end LGCNN (CNN1 -> differentiable streamlines -> CNN2).

Runs on a crop of the real data with small UNets so it finishes on CPU in a few minutes.
Checks:
1. forward produces a T prediction smaller than the input (valid convolutions),
2. the gradient of the T loss reaches CNN1's first conv layer THROUGH the streamlines,
3. a few Adam steps reduce the loss.

Usage:
    python test_e2e_training.py
"""
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing.datasets.dataset import DataPointE2E
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from processing.e2e_patched import ContextCache, measure_border, window_forward
from processing.loss_fcts import E2ELoss
from processing.networks.lgcnn_e2e import LGCNNEndToEnd
from processing.networks.unetVariants import UNetNoPad2

PATH_DATA_PREP = Path(__file__).parents[2] / "datasets_prep"
DATASET_NAME = "dataset_giant_100hp_varyK"
CROP = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop_like(y, y_pred):
    h, w = y_pred.shape[2:]
    i0, j0 = (y.shape[2] - h) // 2, (y.shape[3] - w) // 2
    return y[:, :, i0:i0+h, j0:j0+w]


if __name__ == "__main__":
    torch.manual_seed(0)

    # 0. stage-1 wiring: partitioned velocity training (SimulationDatasetCuts on the pki dataset)
    dataset = DataPointE2E(PATH_DATA_PREP, DATASET_NAME, i=0)
    cuts = SimulationDatasetCuts(dataset.path_v, skip_per_dir=16, box_size=256, ids=0)
    unet_v = UNetNoPad2(in_channels=3, out_channels=2, depth=2, init_features=8, kernel_size=5,
                        stride=1, dilation=1, activation="ReLU", norm="batchnorm", repeat_inner=False).float()
    opt_v = torch.optim.Adam(unet_v.parameters(), lr=1e-3)
    loader = DataLoader(cuts, batch_size=8, shuffle=True)
    v_losses = []
    for bi, (xb, yb) in enumerate(loader):
        if bi == 6:
            break
        opt_v.zero_grad()
        vb = unet_v(xb)
        loss_v = torch.nn.functional.mse_loss(vb, center_crop_like(yb, vb))
        loss_v.backward()
        opt_v.step()
        v_losses.append(float(loss_v.detach()))
    print(f"stage-1 smoke: {len(cuts)} patches available, v-loss {v_losses[0]:.4f} -> {v_losses[-1]:.4f} over {len(v_losses)} batches")
    assert v_losses[-1] < v_losses[0], f"stage-1 velocity loss did not decrease: {v_losses}"
    x, y = dataset[0]
    torch.manual_seed(0)  # decouple from the RNG use of the stage-1 section: the gradient check
    # below is init-sensitive (chaotic backward can overflow to inf/NaN for unlucky random nets)
    x = x[:, :CROP, :CROP].unsqueeze(0).to(DEVICE)   # [1, 3, CROP, CROP]
    y = y[:, :CROP, :CROP].unsqueeze(0).to(DEVICE)   # [1, 1, CROP, CROP]
    n_hps = int((x[0, LGCNNEndToEnd.IDX_I] == 1.0).sum())
    print(f"crop {CROP}^2 of {dataset.input_names[0]}, {n_hps} heat pumps in crop, device: {DEVICE}")
    assert n_hps > 0, "crop contains no heat pumps - increase CROP"

    unet_args = dict(depth=2, init_features=8, kernel_size=5, stride=1, dilation=1,
                     activation="ReLU", norm="batchnorm", repeat_inner=False)
    model = LGCNNEndToEnd(v_stats=dataset.info_v["Labels"], unet_args=unet_args,
                          randomK_data=True, t_steps=500, sigma=1.0).float().to(DEVICE)

    # 1. forward: shapes ([T, vx, vy] output)
    t0 = time.time()
    y_pred = model(x)
    print(f"forward {time.time()-t0:.1f}s: input {tuple(x.shape)} -> prediction {tuple(y_pred.shape)}")
    assert y_pred.shape[1] == 3 and y_pred.shape[2] < CROP and y_pred.shape[3] < CROP
    assert y.shape[1] == 3, "label must carry [T, vx, vy]"
    assert model.last_intermediates["sf"].max() > 0, "no streamlines drawn"

    # 2. gradient reaches CNN1 through the streamlines
    # IMPORTANT: lambda_v=0 here - with lambda_v>0, CNN1 receives gradient directly through the
    # v output channels, which would mask a broken streamline path
    loss = E2ELoss(lambda_v=0.0)(y_pred, center_crop_like(y, y_pred))
    t0 = time.time()
    loss.backward()
    first_conv_weight = next(p for n, p in model.unet_v.named_parameters() if "weight" in n)
    g_v = float(first_conv_weight.grad.abs().max())
    g_T = float(next(p for n, p in model.unet_T.named_parameters() if "weight" in n).grad.abs().max())
    print(f"backward {time.time()-t0:.1f}s: |grad| CNN1 first conv {g_v:.2e} | CNN2 first conv {g_T:.2e}")
    assert g_v > 0, "no gradient reached CNN1 through the streamlines - end-to-end chain is broken!"
    assert g_T > 0, "no gradient reached CNN2"

    # 3. a few optimizer steps with the combined loss (and gradient clipping, as in Solver)
    loss_fct = E2ELoss(lambda_v=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(4):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fct(y_pred, center_crop_like(y, y_pred))
        loss.backward()
        for p in model.parameters():  # sanitize inf/NaN before clipping, as in Solver
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step {step}: loss {losses[-1]:.5f}")
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    # 4. patched stage 2: cached context + live window
    torch.manual_seed(0)
    T_STEPS_P = 800
    model_p = LGCNNEndToEnd(v_stats=dataset.info_v["Labels"], unet_args=unet_args, randomK_data=True,
                            t_steps=T_STEPS_P, sigma=1.0, fade_mode="absolute").float()
    x_full, y_full = dataset[0]
    x_full, y_full = x_full[:, :CROP, :CROP], y_full[:, :CROP, :CROP]  # pseudo-full domain
    b1 = measure_border(model_p.unet_v, 3, "cpu")
    b2 = measure_border(model_p.unet_T, 6, "cpu")
    cache = ContextCache(x_full, y_full, model_p, b1, t_steps_full=T_STEPS_P)
    cache.refresh(model_p)

    # 4a. composition consistency: window = whole frame -> ctx ~ 0, composed sf ~ direct drawing
    V_full = cache.frame[0]
    ctx, live_starts, _ = cache.window_context("sf", 0, 0, V_full)
    assert float(ctx.abs().max()) < 1e-4, f"context density should vanish when all lines are live: {float(ctx.abs().max())}"
    with torch.no_grad():
        _, _, channels = window_forward(model_p, cache, 0, 0, V_full, T_STEPS_P, b1, b2)
        model_p(x_full.unsqueeze(0))
    sf_direct = model_p.last_intermediates["sf"]
    diff = float((channels["sf"] - sf_direct).abs().mean())
    print(f"patched: full-frame composition vs direct drawing: mean abs diff {diff:.5f}")
    assert diff < 5e-3, "composed occupancy deviates from the direct full drawing"

    # 4b. pass-through gradient: a window without heat-pump origins, crossed by cached lines
    V = 256 - 2 * b1
    found = None
    for fi in range(0, cache.frame[0] - V, 32):
        for fj in range(0, cache.frame[1] - V, 32):
            hp_in = ((cache.hp_positions[:, 0] >= fi) & (cache.hp_positions[:, 0] < fi + V) &
                     (cache.hp_positions[:, 1] >= fj) & (cache.hp_positions[:, 1] < fj + V)).any()
            _, starts, _ = cache.window_context("sf", fi, fj, V)
            if not bool(hp_in) and len(starts) > 0:
                found = (fi, fj)
                break
        if found:
            break
    assert found, "no pass-through-only window found - adjust CROP/geometry"
    model_p.zero_grad()
    pred, label, _ = window_forward(model_p, cache, found[0], found[1], V, T_STEPS_P, b1, b2)
    E2ELoss(lambda_v=0.0)(pred, label).backward()
    g_pt = float(next(p for n, p in model_p.unet_v.named_parameters() if "weight" in n).grad.abs().max())
    print(f"patched: pass-through window {found}, |grad| CNN1 first conv {g_pt:.2e}")
    assert g_pt > 0, "no gradient through pass-through streamline segments!"

    # 4c. a few patched updates on a fixed window reduce the combined loss
    loss_fct = E2ELoss(lambda_v=0.5)
    opt = torch.optim.Adam(model_p.parameters(), lr=1e-3)
    p_losses = []
    for step in range(3):
        opt.zero_grad()
        pred, label, _ = window_forward(model_p, cache, 0, 0, V, T_STEPS_P, b1, b2)
        loss = loss_fct(pred, label)
        loss.backward()
        for p in model_p.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model_p.parameters(), 1.0)
        opt.step()
        p_losses.append(float(loss.detach()))
        print(f"patched step {step}: loss {p_losses[-1]:.5f}")
    assert p_losses[-1] < p_losses[0], f"patched loss did not decrease: {p_losses}"

    print("ALL CHECKS PASSED")
