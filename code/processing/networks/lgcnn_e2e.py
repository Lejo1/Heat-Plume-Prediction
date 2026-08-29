from pathlib import Path

import torch

from processing.networks.model import Model
from processing.networks.unetVariants import UNetNoPad2
from step2_streamlines.streamlines_helpers import smooth_velocity_field, trace_and_draw_soft


class LGCNNEndToEnd(Model):
    """End-to-end LGCNN: CNN1 (pki -> v) -> differentiable streamlines -> CNN2 (ixydkc -> T).

    Input x: [B, 3, H, W] with channels ordered as in the pki dataset (p=0, k=1, i=2), normalized.
    Output: [B, 3, h, w] with channels [T, vx, vy] (normalized): the temperature prediction plus
    CNN1's velocity prediction center-cropped to the same size (for the auxiliary velocity loss).
    Spatially smaller than the input because both UNets use valid (no-padding) convolutions.

    CNN1's normalized velocity output feeds CNN2 directly (both datasets share the same Rescale
    stats); only the streamline tracer needs physical velocities, obtained by the differentiable
    affine reverse of the Rescale normalization (stats stored as buffers).
    """

    IDX_P, IDX_K, IDX_I = 0, 1, 2  # channel order of the pki input
    # channel order of CNN2's input, matching the T-dataset info.yaml:
    # 0=Material ID, 1=vx, 2=vy, 3=Streamlines Faded, 4=Permeability, 5=Streamlines Faded Outer

    def __init__(self, v_stats: dict, unet_args: dict, randomK_data: bool = False,
                 t_steps: int = 10_000, sigma: float = 1.0, offsets=(0, 10, -10), use_compile: bool = False,
                 fade_mode: str = "absolute", detach_direct_v: bool = False, unet_args_T: dict = None,
                 v_blur: float = 0.0):
        """v_stats: info.yaml "Labels" dict of the pki->xy dataset (Rescale min/max of vx, vy).

        unet_args_T defaults to unet_args; pass it when CNN2 must differ architecturally from CNN1,
        as when finetuning from two separately tuned baselines (the step-1 and step-3 baselines of
        this repo were trained with kernel_size 5 and 4 respectively)."""
        super().__init__()
        self.unet_v = UNetNoPad2(in_channels=3, out_channels=2, **unet_args)
        self.unet_T = UNetNoPad2(in_channels=6, out_channels=1, **(unet_args_T or unet_args))

        vx_stats = v_stats["Liquid X-Velocity [m_per_y]"]
        vy_stats = v_stats["Liquid Y-Velocity [m_per_y]"]
        assert vx_stats["norm"] == "Rescale" and vy_stats["norm"] == "Rescale", \
            "reverse-normalization implemented for Rescale only"
        v_min = torch.tensor([vx_stats["min"], vy_stats["min"]], dtype=torch.float32)
        v_delta = torch.tensor([vx_stats["max"] - vx_stats["min"], vy_stats["max"] - vy_stats["min"]], dtype=torch.float32)
        self.register_buffer("v_min", v_min.reshape(2, 1, 1))
        self.register_buffer("v_delta", v_delta.reshape(2, 1, 1))

        self.randomK_data = randomK_data
        self.t_steps = t_steps
        self.sigma = sigma
        self.offsets = tuple(offsets)
        self.use_compile = use_compile
        self.fade_mode = fade_mode  # "absolute" (default) or "per_line" (legacy/paper convention)
        # Gaussian smoothing of the velocity field seen by the TRACER only (0 = off). Widens the
        # support of dL/dv from the one-cell thread along each streamline to a band of ~2*v_blur
        # cells; see smooth_velocity_field. Plain attribute, not a buffer, so an annealing schedule
        # can rewrite it per epoch - which means it is NOT stored in the checkpoint and must be set
        # from the config again at inference time.
        self.v_blur = float(v_blur)
        self.detach_direct_v = detach_direct_v  # stop-gradient on CNN2's direct v channels (see forward)
        self.last_intermediates = {}  # detached v/streamlines of the last forward, for plots
        # set by PipelineTap for one step: keep the LIVE stage tensors and retain their .grad, so a
        # single backward yields every stage's input/output *and* the loss gradient on both sides
        # of it. Off by default - it pins a few full-size tensors plus their gradients.
        self.capture_intermediates = False
        self.tapped = {}

    def _load_unet(self, net, path, label: str, device: str = "cpu", model_name: str = "model.pt"):
        """Load one sub-net from a separately trained run directory, with a legible shape check.

        The step-1/step-3 runs were tuned independently, so their architectures need not agree with
        each other or with this run's HPS_options.yaml. A mismatch would otherwise surface as an
        opaque state_dict error, so every disagreement is reported with the parameter shapes."""
        location = "cuda:0" if "cuda" in str(device) else "cpu"
        ckpt = Path(path) / model_name
        assert ckpt.exists(), f"{label}: checkpoint {ckpt} not found"
        sd = torch.load(ckpt, map_location=location)
        sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
        here = net.state_dict()
        bad = [(k, tuple(v.shape), tuple(here[k].shape)) for k, v in sd.items()
               if k in here and v.shape != here[k].shape]
        missing = [k for k in here if k not in sd] + [k for k in sd if k not in here]
        if bad or missing:
            detail = "".join(f"\n    {k}: checkpoint {a} vs model {b}" for k, a, b in bad[:8])
            if missing:
                detail += f"\n    {len(missing)} key(s) present in only one of the two, e.g. {missing[:3]}"
            raise RuntimeError(
                f"{label}: {ckpt} does not fit this run's architecture.{detail}\n"
                f"  The first conv's kernel shape reveals the trained kernel_size; set it in "
                f"HPS_options.yaml (CNN1) or via unet_args_T in command_line_arguments.yaml (CNN2).")
        net.load_state_dict(sd)
        print(f"  loaded {label} from {ckpt} ({sum(p.numel() for p in net.parameters()):,} params)")

    def load_baselines(self, path_v, path_T, device: str = "cpu", model_name: str = "model.pt"):
        """Initialize CNN1 and CNN2 from the two separately trained runs (step 1: pki->v,
        step 3: ixydk+s_outer->T) instead of from one end-to-end checkpoint."""
        self._load_unet(self.unet_v, path_v, "CNN1 (unet_v)", device, model_name)
        self._load_unet(self.unet_T, path_T, "CNN2 (unet_T)", device, model_name)
        self.to(device)

    def load_pretrained_v(self, path_v, device: str = "cpu", model_name: str = "model.pt"):
        """Seed CNN1 alone from an existing step-1 run, leaving CNN2 as initialized. Used by the
        from-scratch pipeline in place of its own stage-1 pretraining."""
        self._load_unet(self.unet_v, path_v, "CNN1 (unet_v)", device, model_name)
        self.to(device)

    def _tap(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Record a stage boundary while capturing. retain_grad() makes .grad available on this
        non-leaf tensor after backward; values and the graph are unchanged."""
        if self.capture_intermediates:
            if t.requires_grad:
                t.retain_grad()
            self.tapped[name] = t
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture_intermediates:
            self.tapped = {}
        self._tap("x", x)                        # CNN1 input
        v_norm = self.unet_v(x)  # [B, 2, h, w], normalized velocities
        self._tap("v_norm", v_norm)              # CNN1 output = streamline + direct-channel input

        # center-crop the input channels to CNN1's (valid-conv reduced) output size
        h, w = v_norm.shape[2:]
        i0, j0 = (x.shape[2] - h) // 2, (x.shape[3] - w) // 2
        x_crop = x[:, :, i0:i0+h, j0:j0+w]

        # streamlines per sample (physical velocities via differentiable reverse-Rescale).
        # v_phys is the streamline branch alone: its gradient is the route (S) signal, while the
        # direct route shows up only in x_T's velocity channels - so tapping both decomposes the
        # two routes from a single backward pass.
        v_phys = self._tap("v_phys", v_norm * self.v_delta + self.v_min)
        # only the tracer sees the coarse-grained field; CNN2's direct v channels keep the sharp
        # prediction, and so does the model's own v output
        v_trace = smooth_velocity_field(v_phys, self.v_blur) if self.v_blur > 0 else v_phys
        sf, sf_outer = [], []
        for b in range(x.shape[0]):
            # heat pump cells: raw Material ID == 2 <=> normalized i-channel == 1
            hp_positions = torch.nonzero(x_crop[b, self.IDX_I] == 1.0).float() + 0.5  # cell-center offset
            occs = trace_and_draw_soft(hp_positions, v_trace[b, 0], v_trace[b, 1], (h, w),
                                       offsets=self.offsets, randomK_data=self.randomK_data,
                                       faded=True, t_steps=self.t_steps, sigma=self.sigma,
                                       use_compile=self.use_compile, fade_mode=self.fade_mode)
            sf.append(occs[0])
            sf_outer.append(sum(occs[1:]) if len(occs) > 1 else torch.zeros_like(occs[0]))
        sf = self._tap("sf", torch.stack(sf).unsqueeze(1))          # [B, 1, h, w], streamline output
        sf_outer = self._tap("sf_outer", torch.stack(sf_outer).unsqueeze(1))

        # CNN2 input in the T-dataset channel order: [i, vx, vy, sf, k, sf_outer].
        # detach_direct_v cuts the gradient from CNN2's direct velocity channels back to CNN1, so
        # CNN1's temperature-loss signal is forced entirely through the differentiable streamlines
        # (CNN2's forward still sees v). v_norm stays live for the streamline trace and the output.
        v_direct = v_norm.detach() if self.detach_direct_v else v_norm
        x_T = self._tap("x_T", torch.cat([x_crop[:, self.IDX_I:self.IDX_I+1], v_direct, sf,
                                          x_crop[:, self.IDX_K:self.IDX_K+1], sf_outer], dim=1))
        self.last_intermediates = {"v_norm": v_norm.detach(), "sf": sf.detach(), "sf_outer": sf_outer.detach()}
        T_pred = self._tap("T_pred", self.unet_T(x_T))  # CNN2 output

        # append v center-cropped to T's size, so the auxiliary velocity loss can supervise CNN1
        ht, wt = T_pred.shape[2:]
        it, jt = (v_norm.shape[2] - ht) // 2, (v_norm.shape[3] - wt) // 2
        return torch.cat([T_pred, v_norm[:, :, it:it+ht, jt:jt+wt]], dim=1)
