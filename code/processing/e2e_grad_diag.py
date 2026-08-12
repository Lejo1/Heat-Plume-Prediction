"""Diagnostic: how much gradient reaches CNN1 (unet_v) *through the differentiable streamline
solver* versus the path that bypasses it, under the pure temperature loss of stage 2.

CNN1's velocity output influences the temperature loss along two routes:
  (S) streamline path : v -> trace/draw -> sf, sf_outer -> CNN2 -> T
  (D) direct path     : v -> CNN2's vx/vy input channels    -> T

The question this module answers: is route (S) a meaningful training signal for CNN1, or is it
negligible because the soft occupancy is nonzero only on the thin streamlines (a "vanishing"
gradient concern)?  We isolate each route without re-tracing the streamlines: detaching the
*direct* channels (v.detach()) leaves route (S) live; detaching sf/sf_outer leaves route (D)
live.  detach() does not change values, so both counterfactual losses equal the real loss -- the
gradients we read are exactly the per-route partials of the real dL/dtheta.

Note on magnitudes: backprop through the chaotic RK4 advection can *explode* (entries ~1e14 /
inf), the opposite of vanishing.  The Solver sanitizes non-finite grads to 0 and then clips the
global norm, so the training-effective contribution of each route is its norm *after* that
nan_to_num sanitize -- we report both raw and sanitized.  Global-norm clipping scales both routes
by the same factor, so the sanitized ratio S : D is what actually moves the weights.
"""
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss

from step2_streamlines.streamlines_helpers import trace_and_draw_soft


def _param_grads(loss, params, retain, extra_inputs):
    """autograd.grad of `loss` w.r.t. `params` (+ any `extra_inputs`), in one backward.
    Returns (raw L2 norm over params, sanitized L2 norm, [grads for extra_inputs])."""
    grads = torch.autograd.grad(loss, list(params) + list(extra_inputs),
                                retain_graph=retain, allow_unused=True)
    n_p = len(list(params))
    pgrads, egrads = grads[:n_p], grads[n_p:]
    raw = torch.sqrt(sum((g.double() ** 2).sum() for g in pgrads if g is not None))
    san = torch.sqrt(sum(torch.nan_to_num(g.double(), nan=0.0, posinf=0.0, neginf=0.0).pow(2).sum()
                         for g in pgrads if g is not None))
    return float(raw), float(san), egrads


def diagnose_e2e_gradients(model, x: torch.Tensor, y: torch.Tensor, args: Dict, plot_path: Path):
    """Run the streamline-vs-direct gradient decomposition on one full-domain datapoint and save
    a plot + return a stats dict.

    x: [1, 3, H, W] pki input (normalized), y: [1, 3, h, w] with channels [T, vx, vy] (normalized).
    Uses eval-mode BatchNorm (deterministic, no running-stat mutation); the relative route
    magnitudes -- the actual question -- are robust to the BN mode."""
    device = args["device"]
    was_training = model.training
    model.eval()
    x, y = x.to(device), y.to(device)
    mse = MSELoss()

    M = model  # channel-index and buffer access mirrors LGCNNEndToEnd.forward
    v = M.unet_v(x)                                   # [1, 2, h, w], depends on unet_v params
    h, w = v.shape[2:]
    i0, j0 = (x.shape[2] - h) // 2, (x.shape[3] - w) // 2
    x_crop = x[:, :, i0:i0 + h, j0:j0 + w]
    v_phys = v * M.v_delta + M.v_min

    # streamlines (live graph, single batch element -- same call as the model's forward)
    hp_positions = torch.nonzero(x_crop[0, M.IDX_I] == 1.0).float() + 0.5
    occs = trace_and_draw_soft(hp_positions, v_phys[0, 0], v_phys[0, 1], (h, w),
                               offsets=M.offsets, randomK_data=M.randomK_data, faded=True,
                               t_steps=M.t_steps, sigma=M.sigma, use_compile=M.use_compile,
                               fade_mode=M.fade_mode)
    sf = occs[0].unsqueeze(0).unsqueeze(0)
    sf_outer = (sum(occs[1:]) if len(occs) > 1 else torch.zeros_like(occs[0])).unsqueeze(0).unsqueeze(0)

    i_ch = x_crop[:, M.IDX_I:M.IDX_I + 1]
    k_ch = x_crop[:, M.IDX_K:M.IDX_K + 1]

    def cnn2_loss(v_direct, sf_in, sfo_in):
        x_T = torch.cat([i_ch, v_direct, sf_in, k_ch, sfo_in], dim=1)
        T_pred = M.unet_T(x_T)
        ht, wt = T_pred.shape[2:]
        it, jt = (y.shape[2] - ht) // 2, (y.shape[3] - wt) // 2
        T_label = y[:, 0:1, it:it + ht, jt:jt + wt]
        return mse(T_pred, T_label)

    # route S: only the streamline channels carry gradient to v (direct channels detached)
    L_stream = cnn2_loss(v.detach(), sf, sf_outer)
    # route D: only the direct velocity channels carry gradient (streamlines detached)
    L_direct = cnn2_loss(v, sf.detach(), sf_outer.detach())

    params = [p for p in M.unet_v.parameters() if p.requires_grad]
    # one backward for S also yields the spatial grads dL/dv (through streamlines) and dL/dsf
    raw_s, san_s, (gv_s, gsf) = _param_grads(L_stream, params, True, [v, sf])
    raw_d, san_d, (gv_d,) = _param_grads(L_direct, params, False, [v])

    # ---- spatial magnitudes (sanitize non-finite for plotting/statistics) ----
    def mag(g):
        g = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)[0]
        return g.pow(2).sum(0).sqrt().cpu().numpy().T  # [w, h] for imshow(origin=lower)

    m_stream, m_direct = mag(gv_s), mag(gv_d)
    m_gsf = torch.nan_to_num(gsf.detach(), nan=0.0, posinf=0.0, neginf=0.0)[0, 0].abs().cpu().numpy().T
    streamlines_np = (sf.detach()[0, 0] + sf_outer.detach()[0, 0]).cpu().numpy().T

    # on-line vs off-line: is the streamline gradient confined to the drawn lines?
    on_line = streamlines_np > 0.05
    frac_on = float(on_line.mean())
    mean_on = float(m_stream[on_line].mean()) if on_line.any() else 0.0
    mean_off = float(m_stream[~on_line].mean()) if (~on_line).any() else 0.0
    tot = m_stream.sum()
    mass_on_line = float(m_stream[on_line].sum() / tot) if tot > 0 else 0.0

    denom = san_s + san_d
    stats = {
        "param_grad_norm_streamline_raw": raw_s,
        "param_grad_norm_streamline_sanitized": san_s,
        "param_grad_norm_direct_raw": raw_d,
        "param_grad_norm_direct_sanitized": san_d,
        "streamline_share_of_sanitized_grad": (san_s / denom if denom > 0 else 0.0),
        "streamline_grad_had_nonfinite": bool(not np.isfinite(raw_s)),
        "spatial_frac_cells_on_streamlines": frac_on,
        "spatial_mean_|dL/dv|_on_streamlines": mean_on,
        "spatial_mean_|dL/dv|_off_streamlines": mean_off,
        "spatial_mass_on_streamlines": mass_on_line,
    }

    _plot(streamlines_np, m_stream, m_direct, m_gsf,
          dict(streamline=san_s, direct=san_d), stats, plot_path)

    print("\n=== e2e gradient decomposition (unet_v parameter gradients, pure T loss) ===")
    print(f"  streamline route : raw {raw_s:.3e}  sanitized {san_s:.3e}"
          + ("   [contained inf/nan -> exploding, not vanishing]" if not np.isfinite(raw_s) else ""))
    print(f"  direct route     : raw {raw_d:.3e}  sanitized {san_d:.3e}")
    print(f"  streamline share of the (sanitized) CNN1 update: "
          f"{stats['streamline_share_of_sanitized_grad'] * 100:.1f}%")
    print(f"  spatial: streamlines cover {frac_on * 100:.1f}% of cells but hold "
          f"{mass_on_line * 100:.1f}% of |dL/dv|_stream mass "
          f"(mean on-line {mean_on:.3e} vs off-line {mean_off:.3e})")
    print(f"  saved plot -> {plot_path}")

    if was_training:
        model.train()
    return stats


def _plot(streamlines, m_stream, m_direct, m_gsf, norms, stats, plot_path):
    def logmap(a):
        a = np.asarray(a, dtype=float)
        pos = a[a > 0]
        floor = pos.min() if pos.size else 1e-12
        return np.log10(np.maximum(a, floor))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    im = axes[0, 0].imshow(streamlines, origin="lower", cmap="magma", interpolation="nearest")
    axes[0, 0].set_title("streamlines (soft occupancy sf + sf_outer)", fontsize=10)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(logmap(m_stream), origin="lower", cmap="magma", interpolation="nearest")
    axes[0, 1].set_title("log10 |dL/dv| via STREAMLINES (route S)", fontsize=10)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(logmap(m_direct), origin="lower", cmap="viridis", interpolation="nearest")
    axes[0, 2].set_title("log10 |dL/dv| via DIRECT channels (route D)", fontsize=10)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(logmap(m_gsf), origin="lower", cmap="magma", interpolation="nearest")
    axes[1, 0].set_title("log10 |dL/dsf| at CNN2's streamline input", fontsize=10)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # sparsity histogram of the streamline-route velocity gradient
    pos = m_stream[m_stream > 0]
    axes[1, 1].hist(np.log10(pos) if pos.size else [0], bins=60, color="#b5651d")
    axes[1, 1].set_title("histogram log10 |dL/dv|_stream (per cell > 0)", fontsize=10)
    axes[1, 1].set_xlabel("log10 magnitude")
    axes[1, 1].set_ylabel("cells")
    frac0 = float((m_stream == 0).mean())
    axes[1, 1].text(0.02, 0.95, f"{frac0*100:.1f}% of cells exactly 0\n"
                                f"{stats['spatial_mass_on_streamlines']*100:.1f}% of mass on lines",
                    transform=axes[1, 1].transAxes, va="top", fontsize=9)

    # headline bar chart: which route actually moves CNN1's weights (sanitized norms)
    labels = list(norms.keys())
    vals = [norms[k] for k in labels]
    colors = {"streamline": "#c1440e", "direct": "#2a7fb8"}
    axes[1, 2].bar(labels, vals, color=[colors[k] for k in labels])
    axes[1, 2].set_title("||dL/dtheta_unet_v|| per route (sanitized)", fontsize=10)
    axes[1, 2].set_ylabel("L2 norm")
    total = sum(vals) or 1.0
    for i, val in enumerate(vals):
        axes[1, 2].text(i, val, f"{val:.2e}\n{val/total*100:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Gradient reaching CNN1 through the differentiable streamline solver vs. the direct bypass",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
