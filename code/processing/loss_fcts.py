import builtins
import re
import weakref
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import max, abs, zeros, sum
from skimage.metrics import structural_similarity as ssim

builtins_max, builtins_min = builtins.max, builtins.min  # the torch imports above shadow the builtins
    
    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1., second_loss:nn.Module = nn.L1Loss()):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha}) with {self.secondary_loss_function}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)

        return self.alpha * self.mse(predictions, labels) + (1. - self.alpha) * eval_second


class TrajectoryLoss:
    """Streamline-position loss for the e2e LGCNN: traces every heat pump's centre streamline in
    CNN1's predicted velocity and in the simulated velocity and returns the mean equal-time distance
    [cells] - per line averaged over the two lines' common in-domain length, then over all lines.

    The temperature loss reaches a streamline only within ~4 sigma of where it is drawn; this one pulls
    a line back however far it has drifted, and its gradient runs through the whole RK4 trajectory
    (full adjoint, independent of detach_trajectory / v_blur, which only concern the drawing route).
    Both lines are traced in CNN2's output region - the region predictions and labels are cropped to -
    from the heat pumps the model's last forward found there (model.last_hp).
    Plain class, not an nn.Module, so the model is not registered as a submodule of the loss.
    """
    def __init__(self, model):
        self.model = model

    def _lines(self, v_phys, hp):
        from step2_streamlines.streamlines_helpers import trace_streamlines
        m = self.model
        return trace_streamlines(hp, v_phys[0], v_phys[1], tuple(v_phys.shape[1:]), randomK_data=m.randomK_data,
                                 t_steps=m.t_steps, use_compile=m.use_compile)

    def __call__(self, predictions, labels):
        m = self.model
        terms = []
        for b, hp in enumerate(m.last_hp):
            if hp.shape[0] == 0:
                continue
            v_pred = predictions[b, 1:3] * m.v_delta + m.v_min
            with torch.no_grad():
                sim = self._lines(labels[b, 1:3] * m.v_delta + m.v_min, hp)
            for (px, py, _), (sx, sy, _) in zip(self._lines(v_pred, hp), sim):
                k = min(len(px), len(sx))
                if k >= 2:
                    terms.append(((px[:k] - sx[:k]) ** 2 + (py[:k] - sy[:k]) ** 2 + 1e-6).sqrt().mean())
        if not terms:  # no heat pump in the output region: zero, but still part of the graph
            return predictions[:, 1:3].sum() * 0.0
        return torch.stack(terms).mean()


# ------------------------------------------------------------------ streamline-frame losses
# Raw-cell box spanned by the heat pumps of data 0-2 of dataset_giant_100hp_varyK (axis 0 = PFLOTRAN y,
# axis 1 = PFLOTRAN x). The dataset generator is not available, so random seeds are drawn uniformly in
# this box; the pumps show no minimum spacing (closest pair 7 cells), so none is imposed.
SEED_BOX_RAW = ((27, 2325), (115, 2443))
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def well_parameters(data_raw) -> dict:
    """Q [m^3/y] per heat pump, aquifer thickness b [m] and porosity phi [-] of the simulation, for the
    flow-based well mask: rate and thickness from inputs/settings.yaml, porosity from inputs/pflotran.in."""
    from utils.utils_args import load_yaml
    data_raw = Path(data_raw)
    settings = load_yaml(data_raw / "inputs" / "settings.yaml")
    match = re.search(r"POROSITY\s+([0-9.eEdD+-]+)", (data_raw / "inputs" / "pflotran.in").read_text())
    assert match, f"no POROSITY in {data_raw / 'inputs' / 'pflotran.in'}"
    return {"Q": float(settings["injection"]["rate"]) * SECONDS_PER_YEAR,
            "b": float(settings["grid"]["size"][2]),
            "phi": float(match.group(1).replace("d", "e").replace("D", "e"))}


def trace_true_lines(velocity, seeds, eps: float, ds: float, t_end: float, t_steps: int,
                     max_length: float = None, use_compile: bool = False):
    """Lines of the simulated velocity, resampled to uniform arc length. No gradient.

    velocity: tracer grid [2, h, w] in cells/y (build_velocity_grid order), seeds [n, 2] in cells.
    A line stops at the first of: leaving the grid, t_end, |v*| < eps, arc length max_length.
    Returns nodes [n, N, 2] at the midpoints of consecutive segments of length ds, and valid [n, N]
    (a prefix per line: only complete segments are kept)."""
    from step2_streamlines.streamlines_helpers import calc_streamlines, sample_velocity
    n, dev, dt = seeds.shape[0], velocity.device, velocity.dtype
    with torch.no_grad():
        if n == 0:
            return torch.zeros((0, 0, 2), dtype=dt, device=dev), torch.zeros((0, 0), dtype=torch.bool, device=dev)
        h, w = velocity.shape[1:]
        lines = calc_streamlines(seeds.to(dev, dt), velocity, (h - 1, w - 1), t_end=t_end, t_steps=t_steps,
                                 use_compile=use_compile)
        T = builtins_max(len(lx) for lx, _, _ in lines)
        P = torch.zeros((n, builtins_max(T, 1), 2), dtype=dt, device=dev)
        length = torch.zeros(n, dtype=torch.long, device=dev)
        for j, (lx, ly, _) in enumerate(lines):
            if len(lx):
                P[j, :len(lx), 0], P[j, :len(lx), 1] = lx, ly
                P[j, len(lx):] = P[j, len(lx) - 1]  # pad with the last position: zero-length segments
                length[j] = len(lx)
        T = P.shape[1]
        # stop where the simulated flow is slower than eps
        speed = sample_velocity(velocity, P.reshape(-1, 2)).norm(dim=1).reshape(n, T)
        ok = (torch.arange(T, device=dev)[None] < length[:, None]) & (speed >= eps)
        length = torch.cumprod(ok.long(), dim=1).sum(dim=1)
        # cumulative arc length, frozen after each line's end
        seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1) * (torch.arange(1, T, device=dev)[None] < length[:, None])
        s = torch.cat([torch.zeros((n, 1), dtype=dt, device=dev), seg.cumsum(dim=1)], dim=1)
        S = s[:, -1] if max_length is None else s[:, -1].clamp(max=float(max_length))
        N = int(torch.floor(S.max() / ds)) if n else 0
        if N == 0:
            return torch.zeros((n, 0, 2), dtype=dt, device=dev), torch.zeros((n, 0), dtype=torch.bool, device=dev)
        k = torch.arange(N, device=dev, dtype=dt)
        target = ((k + 0.5) * ds)[None].expand(n, N).contiguous()
        hi = torch.searchsorted(s.contiguous(), target).clamp(1, T - 1)
        lo = hi - 1
        s_lo, s_hi = s.gather(1, lo), s.gather(1, hi)
        frac = ((target - s_lo) / (s_hi - s_lo).clamp_min(1e-12)).clamp(0, 1)[..., None]
        p_lo = P.gather(1, lo[..., None].expand(n, N, 2))
        p_hi = P.gather(1, hi[..., None].expand(n, N, 2))
        nodes = p_lo + frac * (p_hi - p_lo)
        valid = (k + 1)[None] * ds <= S[:, None] + 1e-9
    return nodes, valid


def line_frame(velocity, nodes):
    """Frame of the simulated flow at the nodes: unit tangent t*, normal n* = (-t*_1, t*_0), speed V*.
    Taken from the field (bilinear, as the tracer samples), not from finite differences of the line."""
    from step2_streamlines.streamlines_helpers import sample_velocity
    n, N = nodes.shape[:2]
    with torch.no_grad():
        v = sample_velocity(velocity, nodes.reshape(-1, 2)).reshape(n, N, 2)
        V = v.norm(dim=-1)
        t = v / V.clamp_min(1e-12)[..., None]
        normal = torch.stack([-t[..., 1], t[..., 0]], dim=-1)
    return t, normal, V


def well_mask(nodes, valid, pumps, V, mode: str = "fixed", radius: float = 5.0, c: float = 1.0,
              params: dict = None, chunk: int = 1 << 18):
    """valid with every node within r_w cells of ANY heat pump removed (the injection flow is radial there).
    fixed: r_w = radius. flow: r_w = max(2, c * Q / (b * phi * V*)) [m -> cells], with V* the simulated
    speed at that node - largest where the flow is slow, i.e. where the injected water stays longest."""
    from step2_streamlines.streamlines_helpers import STREAMLINE_CELL_SIZE
    if pumps.shape[0] == 0 or nodes.numel() == 0:
        return valid.clone()
    with torch.no_grad():
        flat = nodes.reshape(-1, 2)
        pumps = pumps.to(flat.device, flat.dtype)
        dmin = torch.cat([torch.cdist(part, pumps).min(dim=1).values for part in flat.split(chunk)])
        dmin = dmin.reshape(valid.shape)
        if mode == "fixed":
            r_w = torch.full_like(dmin, float(radius))
        elif mode == "flow":
            assert params is not None, "well_mask_mode flow needs Q, b, phi (well_parameters)"
            V_m = (V * STREAMLINE_CELL_SIZE).clamp_min(1e-12)  # cells/y -> m/y
            r_w = (c * params["Q"] / (params["b"] * params["phi"] * V_m) / STREAMLINE_CELL_SIZE).clamp_min(2.0)
        else:
            raise ValueError(f"well_mask_mode must be fixed | flow, got {mode!r}")
    return valid & (dmin >= r_w)


def soft_floor(x, delta: float):
    """g(x) = delta * softplus(x / delta): ~x for x >> delta, positive everywhere. The inner floor 1e-6
    only keeps 1/g finite where the flow is strongly reversed (x < ~-14 delta), where softplus underflows."""
    return delta * F.softplus(x / delta).clamp_min(1e-6)


def frame_terms(u_grid, line, eps: float, floor: float, checkpoint_nodes: int, node_weight=None) -> dict:
    """The two losses on one set of true lines. Only u_grid (CNN1's velocity, tracer grid in cells/y)
    carries a gradient; it enters through one bilinear lookup at the fixed true-line nodes.

    line: dict from FrameLoss.prepare - nodes [n, N, 2], m [n, N] (valid and outside the well radius),
          valid [n, N], t, normal [n, N, 2], V [n, N].
    L_N   = sum m*w*ds * e_perp^2 / (V*^2 + eps^2) / sum m*ds,  e_perp = n*.u
    L_TOF = mean over lines and checkpoints of (log tau_theta - log tau*)^2,
            tau(s_k) = sum_{n < n_k} m*ds / g(.),  g = soft_floor(., floor) on BOTH sides
    angle = arc-length weighted mean of asin(|e_perp| / V*) [deg]."""
    from step2_streamlines.streamlines_helpers import sample_velocity
    m = line["m"]
    n, N = m.shape
    zero = u_grid.sum() * 0.0  # keeps the graph when there are no nodes
    if m.sum() == 0:
        return {"L_N": zero, "L_TOF": zero, "angle [deg]": zero.detach()}
    u = sample_velocity(u_grid, line["nodes"].reshape(-1, 2)).reshape(n, N, 2)
    e_perp = (line["normal"] * u).sum(-1)
    q = (line["t"] * u).sum(-1)
    V = line["V"]
    mf = m.to(u.dtype)
    w = mf if node_weight is None else mf * node_weight
    # padded / masked nodes are replaced BEFORE the nonlinearities, so their (zero) gradient stays finite
    e_safe = torch.where(m, e_perp, torch.zeros_like(e_perp))
    L_N = (w * e_safe ** 2 / (V ** 2 + eps ** 2)).sum() / mf.sum()

    q_safe = torch.where(m, q, V.clamp_min(eps))
    tau_theta = (mf / soft_floor(q_safe, floor)).cumsum(dim=1)  # ds is uniform: cancels in the log ratio
    tau_star = (mf / soft_floor(torch.where(m, V, torch.ones_like(V)), floor)).cumsum(dim=1)
    idx = torch.arange(N, device=m.device)
    last = line["valid"].sum(dim=1) - 1
    checkpoint = (((idx + 1) % checkpoint_nodes == 0)[None] & line["valid"]) | (idx[None] == last[:, None])
    checkpoint &= tau_star > 0
    if checkpoint.any():
        L_TOF = ((tau_theta[checkpoint].log() - tau_star[checkpoint].log()) ** 2).mean()
    else:
        L_TOF = zero
    with torch.no_grad():
        ratio = (e_safe.abs() / V.clamp_min(1e-12)).clamp(0, 1)
        angle = torch.rad2deg((mf * torch.asin(ratio)).sum() / mf.sum())
    return {"L_N": L_N, "L_TOF": L_TOF, "angle [deg]": angle}


class FrameLoss:
    """Streamline-frame losses on CNN1's velocity, evaluated on the TRUE streamlines.

    Lines are traced once (no gradient) in the simulated velocity, resampled to uniform arc length and
    turned into a frame (t*, n*, V*) per node. CNN1's physical velocity u is looked up at those fixed
    nodes; its normal component gives the direction loss L_N, its along-track speed the travel-time
    loss L_TOF (see frame_terms). Unlike TrajectoryLoss there is no RK4 adjoint: an error is blamed on
    the cells next to the true line, wherever CNN1's own line would have gone.

    Lines live in CNN2's output region, the region predictions and labels are cropped to (same as
    TrajectoryLoss). Seeds (seed_mode): the datapoint's heat pumps (hp; + the model's +-offset variants
    with seed_hp_offsets), n_seeds random cells in SEED_BOX_RAW (random), or both. In training mode
    random seeds are redrawn every epoch (resample_seeds_each_epoch, set_epoch); in eval mode a fresh
    generator seeded with seed_rng is used on every call, so val/test seeds are identical across
    epochs and runs. Everything that does not depend on CNN1 is cached per datapoint.

    `shared` holds the resolved eps / floor, so all FrameLoss objects of one run use the same values.
    eps_reference: physical simulated velocity [2, H, W] (m/y, label order vx, vy) of the training
    datapoint, for frame_eps = "pX" (percentile of |v*| over the loss region of that datapoint).
    raw_origin / field_shape: where the label field sits in the raw 2560^2 domain and its size, to
    place SEED_BOX_RAW (subdomain-aware). Plain class, not an nn.Module, like TrajectoryLoss.
    """
    DEFAULTS = dict(seed_mode="hp", n_seeds=200, seed_hp_offsets=False, resample_seeds_each_epoch=True,
                    seed_rng=0, frame_ds_cells=0.5, frame_eps="p5", tof_floor=None, tof_checkpoint_cells=50,
                    well_mask_mode="fixed", well_mask_radius=5.0, well_mask_c=1.0, seed_max_length_cells=None,
                    seed_box_raw=SEED_BOX_RAW)

    def __init__(self, model, eps_reference=None, raw_origin=(0, 0), field_shape=None, well_params=None,
                 shared: dict = None, **options):
        unknown = set(options) - set(self.DEFAULTS)
        assert not unknown, f"FrameLoss: unknown options {sorted(unknown)}"
        self.model = model
        self.opt = {**self.DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
        assert self.opt["seed_mode"] in ("hp", "random", "both"), "seed_mode must be hp | random | both"
        assert self.opt["well_mask_mode"] in ("fixed", "flow"), "well_mask_mode must be fixed | flow"
        self.eps_reference = eps_reference
        self.raw_origin = tuple(int(o) for o in raw_origin)
        self.field_shape = None if field_shape is None else tuple(int(s) for s in field_shape)
        self.well_params = well_params
        self.shared = {} if shared is None else shared
        self.node_weight = None  # hook: fn(line dict) -> [n, N] per-node weights omega (None = 1)
        self.epoch = 0
        self._train_gen = torch.Generator().manual_seed(int(self.opt["seed_rng"]))
        self._cache = {}
        self._memo = (None, None, None)

    # ------------------------------------------------------------ constants of the run
    def _resolve_eps(self, region_shape):
        """eps and the travel-time floor in cells/y, resolved once and shared by all FrameLoss objects."""
        from step2_streamlines.streamlines_helpers import STREAMLINE_CELL_SIZE
        if "eps" not in self.shared:
            spec = self.opt["frame_eps"]
            if isinstance(spec, str) and spec.startswith("p"):
                assert self.eps_reference is not None, "frame_eps as a percentile needs eps_reference"
                ref = self.eps_reference
                h, w = region_shape
                i0, j0 = (ref.shape[1] - h) // 2, (ref.shape[2] - w) // 2
                speed = ref[:, i0:i0 + h, j0:j0 + w].double().norm(dim=0).flatten()
                k = builtins_max(1, int(round(float(spec[1:]) / 100 * speed.numel())))
                eps_m = float(speed.kthvalue(k).values)
                print(f"frame_eps = {spec} of |v*| on the training datapoint's loss region = {eps_m:.3f} m/y")
            else:
                eps_m = float(spec)
            floor_m = eps_m / 10 if self.opt["tof_floor"] is None else float(self.opt["tof_floor"])
            self.shared["eps"] = eps_m / STREAMLINE_CELL_SIZE
            self.shared["floor"] = floor_m / STREAMLINE_CELL_SIZE
            self.shared["eps [m/y]"], self.shared["floor [m/y]"] = eps_m, floor_m
        return self.shared["eps"], self.shared["floor"]

    def set_epoch(self, epoch: int):
        """Epoch callback: training-mode random seeds are redrawn when the epoch changes."""
        self.epoch = int(epoch)

    # ------------------------------------------------------------ seeds
    def seeds(self, b: int, region_shape, training: bool, device="cpu", dtype=torch.float32):
        """[n, 2] seed positions (cells, output-region coordinates) for sample b of the last forward."""
        h, w = region_shape
        parts = []
        if self.opt["seed_mode"] in ("hp", "both"):
            hp = self.model.last_hp[b].to(device, dtype)
            if self.opt["seed_hp_offsets"]:
                hp = torch.cat([hp + torch.tensor([0., float(o)], device=device, dtype=dtype)
                                for o in self.model.offsets])
            inside = (hp[:, 0] >= 0) & (hp[:, 0] <= h - 1) & (hp[:, 1] >= 0) & (hp[:, 1] <= w - 1)
            parts.append(hp[inside])
        if self.opt["seed_mode"] in ("random", "both"):
            gen = self._train_gen if (training and self.opt["resample_seeds_each_epoch"]) \
                else torch.Generator().manual_seed(int(self.opt["seed_rng"]))
            parts.append(self.random_seeds(region_shape, gen).to(device, dtype))
        return torch.cat(parts) if parts else torch.zeros((0, 2), device=device, dtype=dtype)

    def region_origin(self, region_shape):
        """Raw-domain cell of the output region's (0, 0): the label field's origin plus the Solver's crop."""
        if self.field_shape is None:
            return self.raw_origin
        return tuple(o + (f - r) // 2 for o, f, r in zip(self.raw_origin, self.field_shape, region_shape))

    def random_seeds(self, region_shape, gen):
        """n_seeds cell centres, uniform in seed_box_raw clipped to the output region."""
        origin = self.region_origin(region_shape)
        lo, hi = [], []
        for ax in range(2):
            a, b = self.opt["seed_box_raw"][ax]
            lo.append(builtins_max(int(a) - origin[ax], 0))
            hi.append(builtins_min(int(b) - origin[ax], int(region_shape[ax]) - 1))
            assert hi[ax] >= lo[ax], f"seed box {self.opt['seed_box_raw']} does not overlap the output region"
        n = int(self.opt["n_seeds"])
        r = torch.rand((n, 2), generator=gen, dtype=torch.float64)
        cells = torch.stack([lo[ax] + torch.floor(r[:, ax] * (hi[ax] - lo[ax] + 1)) for ax in range(2)], dim=1)
        return cells.float() + 0.5

    # ------------------------------------------------------------ cached true lines
    def prepare(self, v_star_grid, seeds, pumps):
        """Everything that does not depend on CNN1: true lines, frame, mask (see frame_terms)."""
        from step2_streamlines.streamlines_helpers import STREAMLINE_T_END
        eps, _ = self._resolve_eps(tuple(v_star_grid.shape[1:]))
        ds = float(self.opt["frame_ds_cells"])
        nodes, valid = trace_true_lines(v_star_grid, seeds, eps=eps, ds=ds, t_end=STREAMLINE_T_END,
                                        t_steps=self.model.t_steps, max_length=self.opt["seed_max_length_cells"],
                                        use_compile=getattr(self.model, "use_compile", False))
        t, normal, V = line_frame(v_star_grid, nodes)
        m = well_mask(nodes, valid, pumps, V, mode=self.opt["well_mask_mode"], radius=self.opt["well_mask_radius"],
                      c=self.opt["well_mask_c"], params=self.well_params)
        return {"nodes": nodes, "valid": valid, "m": m, "t": t, "normal": normal, "V": V, "n_seeds": seeds.shape[0]}

    def _grid(self, v_norm_2):
        """Normalized [2, h, w] velocity (label/prediction channels vx, vy) -> tracer grid in cells/y."""
        from step2_streamlines.streamlines_helpers import build_velocity_grid, STREAMLINE_CELL_SIZE
        m = self.model
        v = (v_norm_2 * m.v_delta + m.v_min) / STREAMLINE_CELL_SIZE
        return build_velocity_grid(v[0], v[1], tuple(v.shape[1:]), randomK_data=m.randomK_data, dtype=v.dtype)

    def _line(self, b, labels, training):
        region = tuple(labels.shape[2:])
        key = (training, b, region, float(labels[b, 1:3].detach().double().sum()))
        epoch_key = self.epoch if (training and self.opt["resample_seeds_each_epoch"]
                                   and self.opt["seed_mode"] != "hp") else None
        hit = self._cache.get(key)
        if hit is not None and hit[0] == epoch_key:
            return hit[1]
        with torch.no_grad():
            v_star = self._grid(labels[b, 1:3].detach())
            seeds = self.seeds(b, region, training, device=v_star.device, dtype=v_star.dtype)
            pumps_all = getattr(self.model, "last_hp_all", self.model.last_hp)[b]
            line = self.prepare(v_star, seeds, pumps_all)
        if training:  # training data: keep only the current entry per sample
            self._cache = {k: v for k, v in self._cache.items() if not (k[0] and k[1] == b)}
        self._cache[key] = (epoch_key, line)
        return line

    def __call__(self, predictions, labels) -> dict:
        """{"L_N", "L_TOF", "angle [deg]"} averaged over the batch; L_N, L_TOF carry CNN1's gradient.
        Must run right after the model's forward (it reads the heat pumps from model.last_hp)."""
        training = bool(self.model.training)
        eps, floor = self._resolve_eps(tuple(labels.shape[2:]))
        K = builtins_max(1, int(round(float(self.opt["tof_checkpoint_cells"]) / float(self.opt["frame_ds_cells"]))))
        out = {"L_N": [], "L_TOF": [], "angle [deg]": []}
        for b in range(predictions.shape[0]):
            line = self._line(b, labels, training)
            weight = None if self.node_weight is None else self.node_weight(line)
            terms = frame_terms(self._grid(predictions[b, 1:3]), line, eps, floor, K, node_weight=weight)
            for k in out:
                out[k].append(terms[k])
        return {k: torch.stack(v).mean() for k, v in out.items()}

    def metric(self, name: str):
        """Solver metric for one term; the three metrics of one evaluation share a single computation."""
        def fn(predictions, labels):
            ref_p, ref_l, values = self._memo
            if ref_p is None or ref_p() is not predictions or ref_l() is not labels:
                values = {k: v.detach() for k, v in self(predictions, labels).items()}
                self._memo = (weakref.ref(predictions), weakref.ref(labels), values)
            return values[name]
        return fn


class E2ELoss(nn.Module):
    """
    Loss for the end-to-end LGCNN: L(T) + lambda_v * L_v(v), with L = MSE, MAE or Huber (`base`)
    and L_v the same choice for the velocity term (`base_v`; None = same as `base`).
    Predictions and labels carry 3 channels [T, vx, vy] (normalized units, comparable scales).
    lambda_v = 0 recovers the pure temperature loss.
    lambda_traj > 0 with a TrajectoryLoss `traj` adds lambda_traj * (mean streamline distance [cells]).
    lambda_N / lambda_TOF > 0 with a FrameLoss `frame` add lambda_N * L_N + lambda_TOF * L_TOF (direction
    and travel-time error of CNN1's velocity on the true streamlines); added last, so with both 0 the
    loss is computed exactly as before.
    Huber keeps PyTorch's delta=1.0: normalized errors are ~1e-2, so it stays in its quadratic
    branch, where it equals 0.5 * MSE.
    """
    BASES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}

    def __init__(self, lambda_v: float = 0.5, base: str = "mse", base_v: str = None,
                 lambda_traj: float = 0.0, traj: "TrajectoryLoss" = None,
                 lambda_N: float = 0.0, lambda_TOF: float = 0.0, frame: "FrameLoss" = None):
        super(E2ELoss, self).__init__()
        base_v = base if base_v is None else base_v
        for label, b in (("base", base), ("base_v", base_v)):
            if b.lower() not in self.BASES:
                raise ValueError(f"E2ELoss {label} must be one of {list(self.BASES)}, got {b!r}")
        self.fn = self.BASES[base.lower()]()
        self.fn_v = self.BASES[base_v.lower()]()
        self.lambda_v = lambda_v
        if lambda_traj and traj is None:
            raise ValueError("E2ELoss: lambda_traj > 0 needs a TrajectoryLoss (traj=...)")
        self.lambda_traj, self.traj = lambda_traj, traj
        if (lambda_N or lambda_TOF) and frame is None:
            raise ValueError("E2ELoss: lambda_N / lambda_TOF > 0 need a FrameLoss (frame=...)")
        self.lambda_N, self.lambda_TOF, self.frame = lambda_N, lambda_TOF, frame
        v_part = "" if base_v.lower() == base.lower() else f", v: {base_v.upper()}"
        traj_part = f", lambda_traj={lambda_traj}" if lambda_traj else ""
        frame_part = "".join(f", {k}={v}" for k, v in (("lambda_N", lambda_N), ("lambda_TOF", lambda_TOF)) if v)
        self.name = rf"E2ELoss ({base.upper()}{v_part}, lambda_v={lambda_v}{traj_part}{frame_part})"

    def forward(self, predictions, labels):
        loss = self.fn(predictions[:, 0:1], labels[:, 0:1])
        if self.lambda_v != 0:
            loss = loss + self.lambda_v * self.fn_v(predictions[:, 1:3], labels[:, 1:3])
        if self.lambda_traj:
            loss = loss + self.lambda_traj * self.traj(predictions, labels)
        if self.lambda_N or self.lambda_TOF:
            terms = self.frame(predictions, labels)
            if self.lambda_N:
                loss = loss + self.lambda_N * terms["L_N"]
            if self.lambda_TOF:
                loss = loss + self.lambda_TOF * terms["L_TOF"]
        return loss
    

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.min = 0
        self.max = 1

    def forward(self, predictions, labels):
        num_channels = predictions.shape[1]
        ssim_total = 0
        for dp in range(predictions.shape[0]):
            for channel in range(num_channels):
                ssim_val = ssim(predictions[dp, channel].detach().cpu().numpy(), labels[dp, channel].detach().cpu().numpy(), data_range=self.max - self.min)
                ssim_total += ssim_val
        return ssim_total / num_channels
    

class LinfLoss(nn.Module):
    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, output, target):
        return max(abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    """

    def __init__(self, pat_thresholds: list):
        super(PATLoss, self).__init__()
        self.pat_thresholds = pat_thresholds

    def forward(self, output, label):
        if len(output.shape) == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)
        pat = zeros((output.shape[0], len(self.pat_thresholds)), device=output.device)
        for idx in range(output.shape[1]):
            pat[:, idx] = sum(abs(output[:, idx] - label[:, idx]) > self.pat_thresholds[idx], dim=(1, 2)) / (output.shape[2] * output.shape[3])
        return pat * 100