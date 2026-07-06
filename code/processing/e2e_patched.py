"""
Patch-based end-to-end training for the LGCNN: cached global context + live window.

Per update, one window of the domain is re-predicted by CNN1 and its local streamlines are
re-traced differentiably; the contribution of everything outside the window comes from a
periodically refreshed, detached full-domain cache ("treat outside velocities as fixed").
Streamline densities are additive, so cached and live line sets compose exactly BEFORE the
final occupancy saturation: occ = 1 - exp(-(density_ctx.detach() + density_live)).

Coordinates: all streamline data lives in CNN1's valid output frame (full size minus border b1
per side). frame_coord = full_coord - b1.
"""
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from processing.loss_fcts import E2ELoss
from step2_streamlines.streamlines_helpers import build_velocity_grid, calc_streamlines, draw_streamlines_soft

T_END = 27.5
RESOLUTION = 5


def measure_border(unet, in_channels, device, size=256):
    """Valid-conv border of a no-padding UNet, measured with a dummy forward."""
    was_training = unet.training
    unet.eval()
    with torch.no_grad():
        out = unet(torch.zeros(1, in_channels, size, size, device=device))
    unet.train(was_training)
    return (size - out.shape[-1]) // 2


def find_runs(mask):
    """Start/end indices (inclusive) of contiguous True runs in a 1D bool tensor."""
    prev = torch.cat([mask.new_zeros(1), mask[:-1]])
    nxt = torch.cat([mask[1:], mask.new_zeros(1)])
    starts = (mask & ~prev).nonzero(as_tuple=True)[0]
    ends = (mask & ~nxt).nonzero(as_tuple=True)[0]
    return list(zip(starts.tolist(), ends.tolist()))


class ContextCache:
    """Full-domain context of one datapoint: CNN1 prediction, streamline trajectories and
    density maps per channel, refreshed periodically under no_grad."""

    CHANNEL_OFFSETS = {"sf": (0,), "sf_outer": (10, -10)}

    def __init__(self, x_full, y_full, model, b1, t_steps_full=10_000):
        self.x_full = x_full            # [3, H, W] normalized pki, on device
        self.y_full = y_full            # [3, H, W] normalized T/vx/vy labels
        self.b1 = b1
        self.sigma = model.sigma
        self.frame = (x_full.shape[1] - 2*b1, x_full.shape[2] - 2*b1)
        self.t_steps_full = t_steps_full
        # heat pump positions in frame coordinates (cell centers)
        i_chan = x_full[2, b1:b1+self.frame[0], b1:b1+self.frame[1]]
        self.hp_positions = torch.nonzero(i_chan == 1.0).float() + 0.5
        self.updates_since_refresh = None  # None = never refreshed
        self.density = {}
        self.lines = {}

    def maybe_refresh(self, model, refresh_every):
        if self.updates_since_refresh is None or self.updates_since_refresh >= refresh_every:
            self.refresh(model)
            self.updates_since_refresh = 0
        self.updates_since_refresh += 1

    @torch.no_grad()
    def refresh(self, model):
        was_training = model.training
        model.eval()
        v_norm = model.unet_v(self.x_full.unsqueeze(0))[0]
        v_phys = v_norm * model.v_delta + model.v_min
        velocity = build_velocity_grid(v_phys[0]/RESOLUTION, v_phys[1]/RESOLUTION, self.frame,
                                       randomK_data=model.randomK_data)
        for ch, offsets in self.CHANNEL_OFFSETS.items():
            device = self.x_full.device
            starts = torch.cat([self.hp_positions + torch.tensor([0., float(o)], device=device)
                                for o in offsets])
            lines = calc_streamlines(starts, velocity, (self.frame[0]-1, self.frame[1]-1),
                                     t_end=T_END, t_steps=self.t_steps_full, use_compile=model.use_compile)
            self.lines[ch] = lines
            self.density[ch] = draw_streamlines_soft(lines, self.frame, faded=True, sigma=model.sigma,
                                                     fade_mode="absolute", t_end=T_END, return_density=True)
        model.train(was_training)

    def window_context(self, ch, fi, fj, V):
        """For one window (frame coords [fi:fi+V, fj:fj+V]) and channel: the detached context
        density plus the live start states (window coords + absolute start times) of every
        cached line segment inside the window (heat-pump origins and pass-through entries)."""
        live_starts, live_t0, cached_segments = [], [], []
        for sol_x, sol_y, t in self.lines[ch]:
            inside = (sol_x >= fi) & (sol_x < fi + V) & (sol_y >= fj) & (sol_y < fj + V)
            if not bool(inside.any()):
                continue
            for s, e in find_runs(inside):
                live_starts.append([float(sol_x[s]) - fi, float(sol_y[s]) - fj])
                live_t0.append(float(t[s]))
                cached_segments.append((sol_x[s:e+1] - fi, sol_y[s:e+1] - fj, t[s:e+1]))
        ctx = self.density[ch][fi:fi+V, fj:fj+V]
        if cached_segments:
            redraw = draw_streamlines_soft(cached_segments, (V, V), faded=True, sigma=self.sigma,
                                           fade_mode="absolute", t_end=T_END, return_density=True)
            ctx = (ctx - redraw).clamp(min=0)
        return ctx.detach(), live_starts, live_t0


def window_forward(model, cache, fi, fj, V, t_steps_window, b1, b2):
    """One window at frame position (fi, fj): live CNN1 forward, live streamline re-trace,
    composition with the cached context, CNN2 forward.
    Returns (pred [1,3,h,h], label [1,3,h,h], channels dict)."""
    W = V + 2 * b1
    x_win = cache.x_full[:, fi:fi+W, fj:fj+W].unsqueeze(0)
    v_norm = model.unet_v(x_win)                       # [1, 2, V, V], differentiable
    v_phys = v_norm[0] * model.v_delta + model.v_min
    velocity = build_velocity_grid(v_phys[0]/RESOLUTION, v_phys[1]/RESOLUTION, (V, V),
                                   randomK_data=model.randomK_data)

    channels = {}
    for ch in ContextCache.CHANNEL_OFFSETS:
        ctx, live_starts, live_t0 = cache.window_context(ch, fi, fj, V)
        if live_starts:
            lines = calc_streamlines(torch.tensor(live_starts, device=ctx.device), velocity,
                                     (V-1, V-1), t_end=T_END, t_steps=t_steps_window,
                                     use_compile=model.use_compile, t_offsets=live_t0)
            d_live = draw_streamlines_soft(lines, (V, V), faded=True, sigma=model.sigma,
                                           fade_mode="absolute", t_end=T_END, return_density=True)
        else:
            d_live = torch.zeros_like(ctx)
        channels[ch] = (1 - torch.exp(-(ctx + d_live))).unsqueeze(0).unsqueeze(0)

    # CNN2 input in the T-dataset channel order [i, vx, vy, sf, k, sf_outer]
    region = cache.x_full[:, fi+b1:fi+b1+V, fj+b1:fj+b1+V].unsqueeze(0)
    x_T = torch.cat([region[:, 2:3], v_norm, channels["sf"], region[:, 1:2], channels["sf_outer"]], dim=1)
    T_pred = model.unet_T(x_T)                          # [1, 1, V-2*b2, V-2*b2]

    h = T_pred.shape[-1]
    pred = torch.cat([T_pred, v_norm[:, :, b2:b2+h, b2:b2+h]], dim=1)
    o = b1 + b2  # T-region offset in full coordinates
    label = cache.y_full[:, fi+o:fi+o+h, fj+o:fj+o+h].unsqueeze(0)
    return pred, label, channels


def window_update_loss(model, cache, V, t_steps_window, loss_fct, b1, b2, generator):
    """Sample a random window and compute the combined loss (backward done by the caller)."""
    fi = int(torch.randint(0, cache.frame[0] - V + 1, (1,), generator=generator))
    fj = int(torch.randint(0, cache.frame[1] - V + 1, (1,), generator=generator))
    pred, label, _ = window_forward(model, cache, fi, fj, V, t_steps_window, b1, b2)
    return loss_fct(pred, label)


def train_stage2_patched(model, dataset_train, dataset_val, args):
    """Custom training loop (Solver's DataLoader pattern does not fit the cache interplay):
    updates_per_epoch window updates per epoch, periodic cache refresh, full-domain validation,
    tensorboard, best-model tracking. Returns a metrics dict."""
    device = args["device"]
    b1 = measure_border(model.unet_v, 3, device)
    b2 = measure_border(model.unet_T, 6, device)
    V = args.get("window", 512) - 2 * b1
    assert V > 2 * b2 + 16, f"window {args.get('window', 512)} too small for the CNN borders ({b1}, {b2})"
    print(f"STAGE 2 (patched): window {args.get('window', 512)} -> interior {V}, borders CNN1 {b1} / CNN2 {b2}")

    caches = []
    for idx in range(len(dataset_train)):
        x, y = dataset_train[idx]
        caches.append(ContextCache(x.to(device), y.to(device), model, b1))

    x_val, y_val = dataset_val[0]
    x_val, y_val = x_val.unsqueeze(0).to(device), y_val.unsqueeze(0).to(device)

    loss_fct = E2ELoss(lambda_v=args.get("lambda_v", 0.0))
    opt = torch.optim.Adam(model.parameters(), args["lr"], weight_decay=1e-4)
    lr_schedule = {0: args["lr"], int(0.7 * args["epochs"]): args["lr"] / 10}
    writer = SummaryWriter(args["destination"])
    generator = torch.Generator().manual_seed(0)
    refresh_every = args.get("refresh_every", 50)
    updates_per_epoch = args.get("updates_per_epoch", 100)
    t_steps_window = args.get("t_steps_window", 2000)
    best = None
    update = 0
    start_time = datetime.now()

    try:
        for epoch in range(args["epochs"]):
            if epoch in lr_schedule:
                opt.param_groups[0]["lr"] = lr_schedule[epoch]
            model.train()
            epoch_losses = []
            for _ in range(updates_per_epoch):
                cache = caches[update % len(caches)]
                cache.maybe_refresh(model, refresh_every)
                opt.zero_grad()
                loss = window_update_loss(model, cache, V, t_steps_window, loss_fct, b1, b2, generator)
                loss.backward()
                for p in model.parameters():  # sanitize + clip, as in Solver
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                clip_grad_norm_(model.parameters(), args.get("clip_grad", 1.0))
                opt.step()
                epoch_losses.append(float(loss.detach()))
                update += 1
            train_loss = sum(epoch_losses) / len(epoch_losses)

            # full-domain validation with the real pipeline (same fade_mode via the model)
            model.eval()
            with torch.no_grad():
                pred_val = model(x_val)
                h, w = pred_val.shape[2:]
                i0, j0 = (y_val.shape[2] - h) // 2, (y_val.shape[3] - w) // 2
                val_loss = float(loss_fct(pred_val.cpu(), y_val[:, :, i0:i0+h, j0:j0+w].cpu()))

            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("val_loss", val_loss, epoch)
            writer.add_scalar("learning_rate", opt.param_groups[0]["lr"], epoch)
            print(f"epoch {epoch}: train {train_loss:.2e} | val {val_loss:.2e} | lr {opt.param_groups[0]['lr']:.1e}")

            if best is None or val_loss < best["loss"]:
                best = {"epoch": epoch, "loss": val_loss, "train loss": train_loss,
                        "state_dict": deepcopy(model.state_dict())}
    except KeyboardInterrupt:
        logging.warning(f"Manually stopping patched training early (best epoch {best['epoch'] if best else '-'}).")

    if best is not None:
        model.load_state_dict(best["state_dict"])
        print(f"Best model was found in epoch {best['epoch']}.")
    return {"best_epoch": best["epoch"] if best else None,
            "val loss": best["loss"] if best else None,
            "train loss": best["train loss"] if best else None,
            "updates": update,
            "training_time [s]": (datetime.now() - start_time).total_seconds()}
