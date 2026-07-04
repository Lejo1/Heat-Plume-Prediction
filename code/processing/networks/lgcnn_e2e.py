import torch

from processing.networks.model import Model
from processing.networks.unetVariants import UNetNoPad2
from step2_streamlines.streamlines_helpers import trace_and_draw_soft


class LGCNNEndToEnd(Model):
    """End-to-end LGCNN: CNN1 (pki -> v) -> differentiable streamlines -> CNN2 (ixydkc -> T).

    Input x: [B, 3, H, W] with channels ordered as in the pki dataset (p=0, k=1, i=2), normalized.
    Output: [B, 1, h, w] normalized temperature, spatially smaller than the input because both
    UNets use valid (no-padding) convolutions.

    CNN1's normalized velocity output feeds CNN2 directly (both datasets share the same Rescale
    stats); only the streamline tracer needs physical velocities, obtained by the differentiable
    affine reverse of the Rescale normalization (stats stored as buffers).
    """

    IDX_P, IDX_K, IDX_I = 0, 1, 2  # channel order of the pki input
    # channel order of CNN2's input, matching the T-dataset info.yaml:
    # 0=Material ID, 1=vx, 2=vy, 3=Streamlines Faded, 4=Permeability, 5=Streamlines Faded Outer

    def __init__(self, v_stats: dict, unet_args: dict, randomK_data: bool = False,
                 t_steps: int = 10_000, sigma: float = 1.0, offsets=(0, 10, -10), use_compile: bool = False):
        """v_stats: info.yaml "Labels" dict of the pki->xy dataset (Rescale min/max of vx, vy)."""
        super().__init__()
        self.unet_v = UNetNoPad2(in_channels=3, out_channels=2, **unet_args)
        self.unet_T = UNetNoPad2(in_channels=6, out_channels=1, **unet_args)

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
        self.last_intermediates = {}  # detached v/streamlines of the last forward, for plots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_norm = self.unet_v(x)  # [B, 2, h, w], normalized velocities

        # center-crop the input channels to CNN1's (valid-conv reduced) output size
        h, w = v_norm.shape[2:]
        i0, j0 = (x.shape[2] - h) // 2, (x.shape[3] - w) // 2
        x_crop = x[:, :, i0:i0+h, j0:j0+w]

        # streamlines per sample (physical velocities via differentiable reverse-Rescale)
        v_phys = v_norm * self.v_delta + self.v_min
        sf, sf_outer = [], []
        for b in range(x.shape[0]):
            # heat pump cells: raw Material ID == 2 <=> normalized i-channel == 1
            hp_positions = torch.nonzero(x_crop[b, self.IDX_I] == 1.0).float() + 0.5  # cell-center offset
            occs = trace_and_draw_soft(hp_positions, v_phys[b, 0], v_phys[b, 1], (h, w),
                                       offsets=self.offsets, randomK_data=self.randomK_data,
                                       faded=True, t_steps=self.t_steps, sigma=self.sigma,
                                       use_compile=self.use_compile)
            sf.append(occs[0])
            sf_outer.append(sum(occs[1:]) if len(occs) > 1 else torch.zeros_like(occs[0]))
        sf = torch.stack(sf).unsqueeze(1)          # [B, 1, h, w]
        sf_outer = torch.stack(sf_outer).unsqueeze(1)

        # CNN2 input in the T-dataset channel order: [i, vx, vy, sf, k, sf_outer]
        x_T = torch.cat([x_crop[:, self.IDX_I:self.IDX_I+1], v_norm, sf,
                         x_crop[:, self.IDX_K:self.IDX_K+1], sf_outer], dim=1)
        self.last_intermediates = {"v_norm": v_norm.detach(), "sf": sf.detach(), "sf_outer": sf_outer.detach()}
        return self.unet_T(x_T)
