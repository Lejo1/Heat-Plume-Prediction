"""
Test + demo for make_streamlines_gradients: gradient of the drawn (soft-occupancy)
streamlines w.r.t. the velocity fields vx and vy, on the real dataset.

Loads the data exactly as LGCNN_step2.py / build_streamlines does (based_on_pred=False):
reverse-normalized inputs from the `for_s` dataset, simulated velocities from the
`pki -> xy` dataset, vx/vy channels overwritten, randomK convention.

1. computes occupancy + gradients on the real velocity field,
2. validates the autograd gradients against central finite differences (spot check),
3. plots full-field + zoomed views of the occupancy and both gradient fields.

Usage:
    python step2_streamlines/test_streamlines_gradients.py [RUN_1]
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.utils_args import load_yaml
from preprocessing.transforms import NormalizeTransform
from step2_streamlines.streamlines_helpers import (build_velocity_grid, calc_streamlines,
                                                   draw_streamlines_soft, extend_inputs_dims,
                                                   make_streamlines_gradients)

PATH_DATA_PREP = Path(__file__).parents[3] / "datasets_prep"  # ../datasets_prep as in LGCNN_step2
DATASET_NAME = "dataset_giant_100hp_varyK"
RANDOM_K = True

T_STEPS = 1200   # caps the RK4 steps; backward through the loop is the expensive part
SIGMA = 2.5      # splat width in cells, sized so lines/gradients are visible on the 2560^2 grid
ZOOM = 340       # zoom window size in cells


def load_real_data(runid: str):
    """Same data flow as streamlines_main.build_streamlines with based_on_pred=False."""
    origin_T = PATH_DATA_PREP / f"{DATASET_NAME} inputs_ixyk outputs_t for_s"
    origin_v = PATH_DATA_PREP / f"{DATASET_NAME} inputs_pki outputs_xy"
    idx = {"vx": 1, "vy": 2}

    vv = torch.load(origin_v / "Labels" / runid)
    NormalizeTransform(load_yaml(origin_v / "info.yaml")).reverse(vv, "Labels")
    inputs = torch.load(origin_T / "Inputs" / runid)
    NormalizeTransform(load_yaml(origin_T / "info.yaml")).reverse(inputs, "Inputs")

    inputs = extend_inputs_dims(inputs)
    required_size = vv.shape[1:]
    start = ((inputs.shape[1] - required_size[0]) // 2, (inputs.shape[2] - required_size[1]) // 2)
    inputs_reduced = inputs[:, start[0]:start[0]+required_size[0], start[1]:start[1]+required_size[1]]
    inputs_reduced[idx["vx"]] = vv[0]
    inputs_reduced[idx["vy"]] = vv[1]
    return inputs_reduced[0], inputs_reduced[idx["vx"]], inputs_reduced[idx["vy"]], tuple(vv.shape[1:])


def occupancy_loss(mat_ids, vx, vy, dims):
    """Same forward pass as make_streamlines_gradients, without autograd (for finite differences)."""
    pos_hps = torch.nonzero(torch.as_tensor(mat_ids) == 2).float() + torch.tensor([0.5, 0.5])
    with torch.no_grad():
        velocity = build_velocity_grid(torch.as_tensor(vx) / 5, torch.as_tensor(vy) / 5, dims,
                                       randomK_data=RANDOM_K)
        lines = calc_streamlines(pos_hps, velocity, (dims[0] - 1, dims[1] - 1), t_steps=T_STEPS)
        occ = draw_streamlines_soft(lines, dims, faded=True, sigma=SIGMA)
    return float(occ.double().sum())


def finite_difference_check(mat_ids, vx, vy, dims, grad, component, n_cells=3, eps=0.5):
    """Compare d(loss)/d(v) against central finite differences at the strongest-gradient cells."""
    flat = grad.abs().flatten()
    cells = [np.unravel_index(int(k), tuple(dims)) for k in flat.argsort(descending=True)[:n_cells]]
    print(f"\nfinite-difference check for d(loss)/d({component}):")
    max_rel_err = 0.
    for (i, j) in cells:
        vp, vm = vx.clone(), vx.clone()
        wp, wm = vy.clone(), vy.clone()
        if component == "vx":
            vp[i, j] += eps; vm[i, j] -= eps
        else:
            wp[i, j] += eps; wm[i, j] -= eps
        fd = (occupancy_loss(mat_ids, vp, wp, dims) - occupancy_loss(mat_ids, vm, wm, dims)) / (2 * eps)
        ag = float(grad[i, j])
        rel_err = abs(fd - ag) / max(abs(fd), 1e-9)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  cell ({i:4d},{j:4d}): autograd {ag:+.5f} | finite diff {fd:+.5f} | rel err {rel_err:.1%}")
    return max_rel_err


def plot_all(vx, vy, occ, grad_vx, grad_vy, starts, out_path):
    # zoom window centered on the strongest gradient
    gmag = (grad_vx.abs() + grad_vy.abs()).numpy()
    ci, cj = np.unravel_index(gmag.argmax(), gmag.shape)
    i0 = int(np.clip(ci - ZOOM // 2, 0, gmag.shape[0] - ZOOM))
    j0 = int(np.clip(cj - ZOOM // 2, 0, gmag.shape[1] - ZOOM))
    zoom = (slice(i0, i0 + ZOOM), slice(j0, j0 + ZOOM))

    fig, axes = plt.subplots(3, 3, figsize=(24, 22))

    def heat(ax, data, title, cmap, vmin=None, vmax=None, extent=None):
        im = ax.imshow(np.asarray(data).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", extent=extent)
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def glim(g, q=0.999):
        a = np.abs(np.asarray(g))
        nz = a[a > 0]
        return float(np.quantile(nz, q)) if nz.size else 1.

    def maxabs_pool(g, k=4):
        # display-only downsampling that keeps thin gradient lines visible at full-field scale:
        # per k x k block take the value with the largest magnitude (sign preserved)
        gp = torch.nn.functional.max_pool2d(g[None, None], k)[0, 0]
        gn = -torch.nn.functional.max_pool2d(-g[None, None], k)[0, 0]
        return torch.where(gp.abs() >= gn.abs(), gp, gn)

    zoom_extent = (i0, i0 + ZOOM, j0, j0 + ZOOM)  # keep original cell coordinates on the zoom axes
    full_extent = (0, occ.shape[0], 0, occ.shape[1])
    occ_max = float(occ.max())

    # row 1 (full field): the velocity inputs and the differentiable drawing
    heat(axes[0, 0], vx, "input vx [m/y] (full field)", "viridis")
    heat(axes[0, 1], vy, "input vy [m/y] (full field)", "viridis")
    heat(axes[0, 2], occ, f"soft occupancy (full field, scaled to max {occ_max:.2f})", "magma", vmin=0, vmax=occ_max)
    axes[0, 2].scatter(starts[:, 0], starts[:, 1], marker="x", c="white", s=12, linewidths=.7)

    # row 2 (full field): the gradients, max-|.|-pooled so the thin lines survive downscaling
    gx_pool, gy_pool = maxabs_pool(grad_vx), maxabs_pool(grad_vy)
    lim = max(glim(gx_pool, 0.99), glim(gy_pool, 0.99))
    heat(axes[1, 0], gx_pool, "d(occupancy)/d(vx) (full field, 4x max-pooled)", "RdBu_r", vmin=-lim, vmax=lim, extent=full_extent)
    heat(axes[1, 1], gy_pool, "d(occupancy)/d(vy) (full field, 4x max-pooled)", "RdBu_r", vmin=-lim, vmax=lim, extent=full_extent)
    heat(axes[1, 2], occ[zoom], f"soft occupancy (zoom, scaled to max {occ_max:.2f})", "magma", vmin=0, vmax=occ_max, extent=zoom_extent)

    # row 3 (zoom): gradients around the strongest-gradient region, unpooled
    lim_z = max(glim(grad_vx[zoom]), glim(grad_vy[zoom]))
    heat(axes[2, 0], grad_vx[zoom], "d(occupancy)/d(vx) (zoom)", "RdBu_r", vmin=-lim_z, vmax=lim_z, extent=zoom_extent)
    heat(axes[2, 1], grad_vy[zoom], "d(occupancy)/d(vy) (zoom)", "RdBu_r", vmin=-lim_z, vmax=lim_z, extent=zoom_extent)
    heat(axes[2, 2], (grad_vx.abs() + grad_vy.abs())[zoom], "|d/dvx| + |d/dvy| (zoom)", "magma", extent=zoom_extent)

    # mark the zoom window on all full-field panels
    for ax in axes[0].tolist() + axes[1].tolist()[:2]:
        ax.add_patch(Rectangle((i0, j0), ZOOM, ZOOM, fill=False, edgecolor="lime", linewidth=1.5))
    for ax in axes.flat:
        ax.tick_params(labelsize=9)

    fig.suptitle(f"{DATASET_NAME}: streamline occupancy and its gradients w.r.t. vx, vy "
                 f"(green box = zoom region, x = heat pumps; image x-axis = array axis 0)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    runid = (sys.argv[1] if len(sys.argv) > 1 else "RUN_1").replace(".pt", "") + ".pt"
    mat_ids, vx, vy, dims = load_real_data(runid)
    starts = np.array(np.where(mat_ids == 2)).T
    print(f"{runid}: dims {dims}, {starts.shape[0]} heat pumps, v in [{float(vx.min()):.0f}, {float(vx.max()):.0f}] m/y")

    t0 = time.time()
    occ, grad_vx, grad_vy = make_streamlines_gradients(mat_ids, vx, vy, dims, randomK_data=RANDOM_K,
                                                       t_steps=T_STEPS, sigma=SIGMA, faded=True)
    print(f"forward+backward took {time.time()-t0:.1f}s")
    print(f"occupancy: sum {float(occ.sum()):.1f}, max {float(occ.max()):.3f}")
    print(f"|d/dvx| max {float(grad_vx.abs().max()):.5f} | |d/dvy| max {float(grad_vy.abs().max()):.5f}")
    assert grad_vx.abs().max() > 0 and grad_vy.abs().max() > 0, "gradients are all zero!"

    err_x = finite_difference_check(mat_ids, vx, vy, dims, grad_vx, "vx")
    err_y = finite_difference_check(mat_ids, vx, vy, dims, grad_vy, "vy")
    assert max(err_x, err_y) < 0.2, f"finite-difference mismatch: {max(err_x, err_y):.1%}"

    out = Path(__file__).parent / "test_streamlines_gradients.png"
    plot_all(vx, vy, occ, grad_vx, grad_vy, starts, out)
    print("ALL CHECKS PASSED")
