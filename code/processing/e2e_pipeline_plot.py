"""Periodic per-step diagnostic of the end-to-end pipeline: the input and output of every stage
(CNN1 -> differentiable streamlines -> CNN2) together with the loss gradient on both sides of each
stage, saved every N optimizer steps during stage-2 training.

Where the numbers come from: LGCNNEndToEnd._tap() keeps the live stage tensors and calls
retain_grad() on them for the armed step, so the *normal* training backward - not a second,
counterfactual one - populates .grad at every stage boundary. Nothing about the step changes:
retain_grad() only asks autograd to keep a gradient it computes anyway.

Reading the gradient panels
---------------------------
  dL/dT_pred    after CNN2  : the raw error signal, ~2(T_pred - T_label)/N for the MSE
  dL/dx_T       before CNN2 : per-channel, this is what CNN2 hands back to each of its 6 inputs.
                              Channel 3 (sf) is the streamline route, channels 1/2 (vx, vy) the
                              direct bypass - their ratio is the end-to-end question in one number.
  dL/dsf        after the streamlines = dL/dx_T channel 3, kept separate for the log-scale plot
  dL/dv_phys    before the streamlines: what survives backprop through the RK4 advection and the
                              soft drawing. Scaled by v_delta it is comparable to dL/dv_norm.
  dL/dv_norm    after CNN1  : the total CNN1 sees = streamline route + direct route
                              (with detach_direct_v it is the streamline route alone)
  dL/dx         before CNN1 : input sensitivity, only populated because the tap sets
                              x.requires_grad_(True) for the armed step

Backprop through the chaotic advection can produce inf/nan (see e2e_grad_diag.py); every field is
nan_to_num-sanitized for plotting, and both raw and sanitized norms are logged so an exploding step
is visible rather than silently zeroed. Norms are read BEFORE gradient clipping, i.e. they are the
gradients as produced, not as applied.
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# CNN2's input channel order, mirroring LGCNNEndToEnd.forward / the T-dataset info.yaml
X_T_CHANNELS = ["i (material)", "vx (direct)", "vy (direct)", "sf (streamlines)", "k (perm)", "sf_outer"]


def _sanitize(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t.detach(), nan=0.0, posinf=0.0, neginf=0.0)


def _norms(g: Optional[torch.Tensor]):
    """(raw, sanitized) L2 norm; raw is inf/nan exactly when the gradient exploded."""
    if g is None:
        return float("nan"), float("nan")
    g = g.detach().double()
    return float(g.pow(2).sum().sqrt()), float(_sanitize(g).pow(2).sum().sqrt())


def _field(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """[.., h, w] -> [w, h] numpy, magnitude over the channel axis, for imshow(origin='lower')."""
    if t is None:
        return None
    t = _sanitize(t)[0]
    if t.dim() == 3:
        t = t.pow(2).sum(0).sqrt() if t.shape[0] > 1 else t[0]
    return t.float().cpu().numpy().T


def _logmap(a: np.ndarray) -> np.ndarray:
    pos = a[a > 0]
    floor = pos.min() if pos.size else 1e-12
    return np.log10(np.maximum(a, floor))


class PipelineTap:
    """Arms the model to capture stage tensors every `every` optimizer steps and renders the plot.

    every:     plot cadence in optimizer steps (0/None = disabled)
    max_plots: stop after this many plots (0 = unlimited), so a long run cannot fill the disk
    t_stats:   info_T["Labels"]["Temperature [C]"] - de-normalizes the T panels to [degC]
    """

    def __init__(self, every: int, destination: Path, max_plots: int = 0, t_stats: Optional[dict] = None):
        self.every = int(every or 0)
        self.max_plots = int(max_plots or 0)
        self.t_stats = t_stats
        self.dir = Path(destination) / "pipeline_plots"
        self.csv = self.dir / "pipeline_scalars.csv"
        self.step = 0
        self.n_plots = 0
        self._csv_header_written = False
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def arm(self, model, x: torch.Tensor) -> bool:
        """Called before the forward pass. Returns whether this step is captured."""
        due = (self.enabled and self.step % self.every == 0
               and (self.max_plots <= 0 or self.n_plots < self.max_plots))
        model.capture_intermediates = due
        if due and not x.requires_grad:
            x.requires_grad_(True)  # so dL/dx (CNN1's input-side gradient) exists
        return due

    def advance(self):
        """Called once per optimizer step, captured or not."""
        self.step += 1

    def plot(self, model, y_label: torch.Tensor, loss: torch.Tensor):
        """Called after loss.backward() and before gradient clipping."""
        try:
            tap = model.tapped
            grads = {k: getattr(v, "grad", None) for k, v in tap.items()}
            stats = self._scalars(model, tap, grads, loss)
            path = self.dir / f"pipeline_step{self.step:06d}.png"
            self._figure(tap, grads, y_label, stats, path)
            self._append_csv(stats)
            self.n_plots += 1
            print(f"  [pipeline plot] step {self.step}: {path.name}  "
                  f"(at CNN2's input: streamline route {stats.get('grad_x_T_stream', float('nan')):.2e} "
                  f"= {stats.get('sf_share_at_cnn2_input', 0.0)*100:.0f}%, direct-v route "
                  f"{stats.get('grad_x_T_v', float('nan')):.2e})")
        finally:
            model.capture_intermediates = False
            model.tapped = {}

    # ---------------------------------------------------------------- scalars
    def _scalars(self, model, tap, grads, loss) -> Dict[str, float]:
        s = {"step": self.step, "loss": float(loss.detach()),
             "detach_direct_v": float(bool(getattr(model, "detach_direct_v", False)))}
        for name in ("x", "v_norm", "v_phys", "sf", "sf_outer", "x_T", "T_pred"):
            raw, san = _norms(grads.get(name))
            s[f"grad_{name}"] = san
            s[f"grad_{name}_raw_nonfinite"] = float(not np.isfinite(raw))

        # per-channel breakdown at CNN2's input: which route feeds CNN2 how much
        g_xT = grads.get("x_T")
        if g_xT is not None:
            per_ch = _sanitize(g_xT)[0].flatten(1).double().pow(2).sum(1).sqrt()
            for idx, name in enumerate(X_T_CHANNELS):
                s[f"grad_x_T_ch{idx}"] = float(per_ch[idx])
            s["grad_x_T_sf"] = float(per_ch[3])
            s["grad_x_T_sf_outer"] = float(per_ch[5])
            s["grad_x_T_v"] = float((per_ch[1] ** 2 + per_ch[2] ** 2).sqrt())
            # full streamline route = both occupancy channels, full direct route = both v channels
            s["grad_x_T_stream"] = float((per_ch[3] ** 2 + per_ch[5] ** 2).sqrt())
            denom = s["grad_x_T_stream"] + s["grad_x_T_v"]
            s["sf_share_at_cnn2_input"] = s["grad_x_T_stream"] / denom if denom > 0 else 0.0

        # streamline route mapped into normalized-velocity units (chain rule of the reverse-Rescale)
        if grads.get("v_phys") is not None:
            _, s["grad_v_stream_in_norm_units"] = _norms(_sanitize(grads["v_phys"]) * model.v_delta)

        for label, module in (("cnn1", model.unet_v), ("cnn2", model.unet_T)):
            pg = [p.grad for p in module.parameters() if p.grad is not None]
            raw, san = _norms(torch.cat([g.flatten() for g in pg])) if pg else (float("nan"),) * 2
            s[f"grad_params_{label}"] = san
            s[f"grad_params_{label}_raw_nonfinite"] = float(not np.isfinite(raw))
        return s

    def _append_csv(self, stats: Dict[str, float]):
        if not self._csv_header_written:
            self.csv.write_text(",".join(stats.keys()) + "\n")
            self._csv_header_written = True
        with self.csv.open("a") as f:
            f.write(",".join(f"{v:.6e}" for v in stats.values()) + "\n")

    # ----------------------------------------------------------------- figure
    def _figure(self, tap, grads, y_label, stats, path: Path):
        fig, axes = plt.subplots(4, 5, figsize=(26, 19))

        def show(ax, data, title, cmap="viridis", log=False, lim=None):
            if data is None:
                ax.axis("off")
                ax.set_title(f"{title}\n(not available)", fontsize=9)
                return
            data = _logmap(data) if log else data
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
            im = ax.imshow(data, origin="lower", cmap=cmap, interpolation="nearest", **kw)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        x = _sanitize(tap["x"])[0] if "x" in tap else None
        v = _sanitize(tap["v_norm"])[0] if "v_norm" in tap else None

        # --- row 0: CNN1 forward (input pki -> output v) ---
        show(axes[0, 0], x[0].float().cpu().numpy().T if x is not None else None, "CNN1 in: p (normed)", "cividis")
        show(axes[0, 1], x[1].float().cpu().numpy().T if x is not None else None, "CNN1 in: k (normed)", "cividis")
        show(axes[0, 2], x[2].float().cpu().numpy().T if x is not None else None, "CNN1 in: i (heat pumps)", "gray")
        show(axes[0, 3], v[0].float().cpu().numpy().T if v is not None else None, "CNN1 out: vx (normed)")
        show(axes[0, 4], v[1].float().cpu().numpy().T if v is not None else None, "CNN1 out: vy (normed)")

        # --- row 1: gradients around CNN1 ---
        show(axes[1, 0], _field(grads.get("x")), "log10 |dL/dx|  BEFORE CNN1", "magma", log=True)
        show(axes[1, 1], _field(grads.get("v_norm")), "log10 |dL/dv_norm|  AFTER CNN1 (total)", "magma", log=True)
        show(axes[1, 2], _field(grads.get("v_phys")), "log10 |dL/dv_phys|  streamline route only", "magma", log=True)
        g_xT = grads.get("x_T")
        show(axes[1, 3], _field(g_xT[:, 1:3]) if g_xT is not None else None,
             "log10 |dL/dv| direct route (x_T ch 1,2)", "viridis", log=True)
        self._route_bars(axes[1, 4], stats)

        # --- row 2: streamlines + CNN2 forward ---
        show(axes[2, 0], _field(tap.get("sf")), "streamlines out: sf", "magma")
        show(axes[2, 1], _field(tap.get("sf_outer")), "streamlines out: sf_outer", "magma")
        T_pred, T_label, unit = self._temperatures(tap.get("T_pred"), y_label)
        lim = None
        if T_pred is not None and T_label is not None:
            lim = (min(T_pred.min(), T_label.min()), max(T_pred.max(), T_label.max()))
        show(axes[2, 2], T_pred, f"CNN2 out: T prediction {unit}", "RdBu_r", lim=lim)
        show(axes[2, 3], T_label, f"label T {unit}", "RdBu_r", lim=lim)
        show(axes[2, 4], None if T_pred is None else np.abs(T_pred - T_label), f"abs error T {unit}", "magma")

        # --- row 3: gradients around CNN2 ---
        show(axes[3, 0], _field(grads.get("sf")), "log10 |dL/dsf|  AFTER streamlines", "magma", log=True)
        show(axes[3, 1], _field(grads.get("sf_outer")), "log10 |dL/dsf_outer|", "magma", log=True)
        show(axes[3, 2], _field(g_xT), "log10 |dL/dx_T|  BEFORE CNN2 (all channels)", "magma", log=True)
        show(axes[3, 3], _field(grads.get("T_pred")), "log10 |dL/dT_pred|  AFTER CNN2", "magma", log=True)
        self._summary_text(axes[3, 4], stats)

        fig.suptitle(f"e2e pipeline at optimizer step {self.step}   |   loss {stats['loss']:.4e}   |   "
                     f"forward fields (rows 1, 3) and loss gradients on both sides of each stage "
                     f"(rows 2, 4)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _temperatures(self, T_pred, y_label):
        if T_pred is None:
            return None, None, ""
        pred = _sanitize(T_pred)[0, 0]
        label = y_label[0, 0].detach() if y_label is not None else None
        unit = "(normed)"
        if self.t_stats is not None:
            delta = self.t_stats["max"] - self.t_stats["min"]
            pred = pred * delta + self.t_stats["min"]
            label = label * delta + self.t_stats["min"] if label is not None else None
            unit = "[degC]"
        to_np = lambda t: None if t is None else t.float().cpu().numpy().T
        return to_np(pred), to_np(label), unit

    def _route_bars(self, ax, stats):
        """Streamline vs direct-velocity route, measured at CNN2's input in one backward.

        With detach_direct_v the direct channels still receive a gradient here - the stop-gradient
        sits between x_T and v_norm, so this bar is measured but never reaches CNN1. It is drawn
        hatched and faded in that case, matching the convention of e2e_grad_diag.py."""
        keys = [("grad_x_T_sf", "sf\n(streamline)"), ("grad_x_T_sf_outer", "sf_outer\n(streamline)"),
                ("grad_x_T_v", "vx,vy\n(direct)")]
        vals = [stats.get(k, 0.0) for k, _ in keys]
        if not any(np.isfinite(v) and v > 0 for v in vals):
            ax.axis("off"); ax.set_title("route comparison\n(not available)", fontsize=9)
            return
        detach = bool(stats.get("detach_direct_v", 0.0))
        colors = ["#c1440e", "#e8a33d", "#c9d3db" if detach else "#2a7fb8"]
        bars = ax.bar([lbl for _, lbl in keys], vals, color=colors)
        if detach:
            bars[2].set_hatch("//")
        ax.set_yscale("log")
        ax.set_ylim(top=max(v for v in vals if np.isfinite(v)) * 3)  # headroom for the value labels
        ax.set_title("||dL/dx_T|| per route at CNN2's input"
                     + ("\n(direct route BLOCKED before CNN1: detach_direct_v)" if detach else ""),
                     fontsize=9)
        ax.set_ylabel("L2 norm (log)")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
        share = stats.get("sf_share_at_cnn2_input")
        if share is not None:
            note = f"streamline route = {share*100:.1f}% of the signal entering CNN2"
            if detach:
                note += "\nCNN1's update is 100% streamline (direct is cut)"
            ax.text(0.5, 0.97, note, transform=ax.transAxes, fontsize=8, ha="center", va="top",
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    def _summary_text(self, ax, stats):
        ax.axis("off")
        lines = [f"step {self.step}    loss {stats['loss']:.4e}", "",
                 "gradient L2 norms (sanitized, pre-clipping)", "-" * 46,
                 f"  dL/dx        before CNN1  {stats.get('grad_x', float('nan')):.3e}",
                 f"  dL/dv_norm   after  CNN1  {stats.get('grad_v_norm', float('nan')):.3e}",
                 f"  dL/dv_phys   before lines {stats.get('grad_v_phys', float('nan')):.3e}",
                 f"    (in v_norm units)       {stats.get('grad_v_stream_in_norm_units', float('nan')):.3e}",
                 f"  dL/dsf       after  lines {stats.get('grad_sf', float('nan')):.3e}",
                 f"  dL/dx_T      before CNN2  {stats.get('grad_x_T', float('nan')):.3e}",
                 f"  dL/dT_pred   after  CNN2  {stats.get('grad_T_pred', float('nan')):.3e}", "",
                 "per channel at CNN2's input", "-" * 46]
        lines += [f"  ch{i} {name:<18} {stats.get(f'grad_x_T_ch{i}', float('nan')):.3e}"
                  for i, name in enumerate(X_T_CHANNELS)]
        lines += ["", "parameter gradients", "-" * 46,
                  f"  ||dL/dtheta|| CNN1        {stats.get('grad_params_cnn1', float('nan')):.3e}",
                  f"  ||dL/dtheta|| CNN2        {stats.get('grad_params_cnn2', float('nan')):.3e}",
                  "", f"  detach_direct_v: {bool(stats.get('detach_direct_v', 0.0))}"]
        exploded = [k.replace("grad_", "").replace("_raw_nonfinite", "")
                    for k, v in stats.items() if k.endswith("_raw_nonfinite") and v]
        if exploded:
            lines += ["", "!! inf/nan before sanitizing: " + ", ".join(exploded)]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace",
                transform=ax.transAxes)
