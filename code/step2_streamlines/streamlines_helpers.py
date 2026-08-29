import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from utils.utils_args import load_yaml, save_yaml


def build_new_args(args, model_name: str = None):
    args["inputs"] = [
    "Material ID",
    "Liquid X-Velocity [m_per_y]",
    "Liquid Y-Velocity [m_per_y]",
    "Streamlines Faded [-]",
    "Permeability X [m^2]",
    "Streamlines Faded Outer [-]"
    ]
    args["outputs"] = ["Temperature [C]"]
    if model_name:
        idx_vx = 1
        idx_vy = 2
        args["inputs"][idx_vx] = f"Liquid X-Velocity [m_per_y] - predicted by '{model_name.name}'"
        args["inputs"][idx_vy] = f"Liquid Y-Velocity [m_per_y] - predicted by '{model_name.name}'"

def build_new_info(info:dict, info_vx:dict, info_vy:dict):
    info["Inputs"]["Streamlines Faded [-]"] = {
    "index": 3,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Streamlines Faded Outer [-]"] = {
    "index": 5,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Permeability X [m^2]"]["index"] = 4

    info["Inputs"]["Liquid X-Velocity [m_per_y]"] = info_vx
    info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"] = 1
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"] = info_vy
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"] = 2

# data processing + streamline calculation
def build_velocity_grid(vx, vy, dims, randomK_data:bool=False, dtype=torch.float32):
    # dtype float64 is only needed for gradient verification (autograd.gradcheck)
    vx = torch.as_tensor(vx, dtype=dtype)
    vy = torch.as_tensor(vy, dtype=dtype)
    if randomK_data:
        u_axis0, u_axis1 = vy, vx
    else:
        u_axis0, u_axis1 = vx, vy
    return torch.stack([u_axis0, u_axis1])  # (2, dims[0], dims[1])

def sample_velocity(velocity, pos):
    # bilinear interpolation with velocity samples at integer coordinates 0..N-1 and edge
    # clamping, same convention as the previous RegularGridInterpolator
    n0, n1 = velocity.shape[1:]
    x = pos[:,0].clamp(0, n0-1)
    y = pos[:,1].clamp(0, n1-1)
    i = x.floor().long().clamp(max=n0-2)
    j = y.floor().long().clamp(max=n1-2)
    fx = (x - i).unsqueeze(0)
    fy = (y - j).unsqueeze(0)
    v = (velocity[:,i,j]     * (1-fx) * (1-fy) + velocity[:,i+1,j]   * fx * (1-fy)
       + velocity[:,i,j+1]   * (1-fx) * fy     + velocity[:,i+1,j+1] * fx * fy)
    return v.T  # (n_points, 2)

def rk4_step(x, velocity, dt):
    k1 = sample_velocity(velocity, x)
    k2 = sample_velocity(velocity, x + dt/2*k1)
    k3 = sample_velocity(velocity, x + dt/2*k2)
    k4 = sample_velocity(velocity, x + dt*k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

_rk4_step_compiled = None
_compile_failed = False

def get_rk4_step(use_compile:bool):
    # torch.compile fuses the ~450 small tensor ops of one RK4 step into a few kernels, removing
    # the per-step Python dispatch overhead that pins one CPU core and starves the GPU.
    # Compiled once per process; any failure falls back to the eager step permanently.
    global _rk4_step_compiled, _compile_failed
    if not use_compile or _compile_failed:
        return rk4_step
    if _rk4_step_compiled is None:
        try:
            # deliberately NOT mode="reduce-overhead": its CUDA graphs cannot replay a step that
            # is invoked thousands of times within one forward while all outputs still require
            # backward ("Unable to hit fast path" warning, slower than eager). Plain inductor
            # kernel fusion is what removes the per-step dispatch overhead.
            _rk4_step_compiled = torch.compile(rk4_step)
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}), running RK4 without compilation")
            _compile_failed = True
            return rk4_step
    return _rk4_step_compiled

def calc_streamlines(start_points, velocity, maxs_xy, t_end=27.5, t_steps=1000, max_step_cells=0.5, use_compile:bool=False):
    # Solve for all start points at once with fixed-step RK4. The step count is chosen so that
    # no point moves more than max_step_cells per step; the coarse solution is then linearly
    # upsampled to tdraw_streamlines: instead of setting single cells (gradient zero
    # almost everywhere), every sample spreads a normalized Gaussian of width sigma. Samples are
    # weighted by their arc length ds, so a cell's density is the faded line length crossing it
    # (independent of the time-sampling density) and does not saturate where samples are dense.
    # occupancy = 1 - exp(-density) keeps values in [0,1).
    # fade_mode "absolute" (default): fade = 1 - t/t_end - independent of where a line is cut,
    # so patch/window training and full-domain_steps samples for drawing.
    # Differentiable: gradients flow through the RK4 steps and the bilinear velocity sampling
    # (wrap calls in torch.no_grad() when gradients are not needed, it is much faster).
    global _compile_failed
    x = torch.as_tensor(start_points, dtype=velocity.dtype).clone()  # dtype follows the velocity grid
    v_max = float(velocity.detach().norm(dim=0).max())  # only picks the step count, no gradient needed
    n_int = max(min(int(np.ceil(t_end * v_max / max_step_cells)), t_steps - 1), 16)
    dt = t_end / n_int

    step_fn = get_rk4_step(use_compile)
    # dt as 0-d tensor for the compiled path: a python float would be baked into the compiled
    # graph as a constant and trigger a recompilation whenever the step size changes
    dt_arg = torch.tensor(dt, dtype=velocity.dtype, device=x.device) if step_fn is not rk4_step else dt

    trajectory = torch.empty((x.shape[0], n_int+1, 2), dtype=x.dtype, device=x.device)
    trajectory[:,0] = x
    for i in range(n_int):
        try:
            x = step_fn(x, velocity, dt_arg)
        except Exception as e:
            if step_fn is rk4_step:
                raise
            print(f"WARNING: compiled RK4 step failed ({e}), falling back to eager execution")
            _compile_failed = True
            step_fn, dt_arg = rk4_step, dt
            x = step_fn(x, velocity, dt_arg)
        trajectory[:,i+1] = x

    # linear upsampling to t_steps samples, vectorized over all lines
    t = torch.linspace(0, t_end, t_steps, dtype=x.dtype, device=x.device)
    pos = t / dt
    idx = pos.long().clamp(max=n_int - 1)
    w = (pos - idx).reshape(1, -1, 1)
    fine = trajectory[:, idx] * (1 - w) + trajectory[:, idx + 1] * w

    # cut each line where it first leaves the domain (0..x.max(), 0..y.max())
    inside = (fine[:,:,0] >= 0) & (fine[:,:,0] <= maxs_xy[0]) & (fine[:,:,1] >= 0) & (fine[:,:,1] <= maxs_xy[1])
    lengths = torch.cumprod(inside.long(), dim=1).sum(dim=1)  # samples before first exit
    return [(line[:length,0], line[:length,1], t[:length]) for line, length in zip(fine, lengths)]

def draw_streamlines(image_data:torch.Tensor, streamlines:list, faded:bool=False):
    time = datetime.now()
    for streamline in streamlines:
        if faded and len(streamline[2]) > 0:
            val = streamline[2].flip(0)
            val = val / val.max()
        else:
            val = torch.ones(len(streamline[2]))
        image_data[((streamline[0]+0.5).long(),(streamline[1]+0.5).long())] = val
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

SIGMA_BLUR_MIN = 1.0  # at/above this sigma "auto" drawing switches from the splat to the blur path

def _scatter_bilinear(density, x, y, amp, dims):
    # Deposit each sample's mass `amp` on the 4 cell centers surrounding it (centers sit at integer
    # coordinates, the same convention the splat path's floor(x+0.5) indexing uses). Which 4 cells
    # are hit is discrete -> detached; the bilinear weights carry the gradient w.r.t. position.
    i0, j0 = x.floor(), y.floor()
    fx = (x - i0.detach()).unsqueeze(1)
    fy = (y - j0.detach()).unsqueeze(1)
    i0 = i0.detach().long().unsqueeze(1)
    j0 = j0.detach().long().unsqueeze(1)
    ii = torch.cat([i0, i0+1, i0,   i0+1], dim=1)
    jj = torch.cat([j0, j0,   j0+1, j0+1], dim=1)
    w = torch.cat([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], dim=1) * amp.unsqueeze(1)
    mask = (ii >= 0) & (ii < dims[0]) & (jj >= 0) & (jj < dims[1])
    density.index_put_((ii[mask], jj[mask]), w[mask], accumulate=True)

def _gaussian_blur_separable(img, sigma, radius):
    # The same analytic Gaussian the splat path evaluates, applied as two 1D passes:
    # exp(-(di^2+dj^2)/2s^2) / (2 pi s^2) factorizes into [exp(-di^2/2s^2) / (sqrt(2 pi) s)] per
    # axis. The kernel is left un-renormalized so the truncation at +-radius drops the same tail
    # mass the splat path's window drops. Zero padding discards what blurs out of the domain,
    # matching the splat path's bounds mask. Cost: 2*(2*radius+1) taps over the grid.
    r = torch.arange(-radius, radius+1, dtype=img.dtype, device=img.device)
    k = torch.exp(-r**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    out = img.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(out, k.view(1, 1, -1, 1), padding=(radius, 0))
    out = F.conv2d(out, k.view(1, 1, 1, -1), padding=(0, radius))
    return out[0, 0]

def smooth_velocity_field(v, sigma: float):
    """Coarse-grain a velocity field before tracing. Input/output [..., H, W] (any leading dims).

    Why this exists: the tracer reads the velocity grid ONLY through bilinear interpolation at the
    trajectory points (sample_velocity), so dL/dv is supported on the ~4 cells around each RK4
    evaluation - a one-cell-wide thread along the streamline. The drawing width sigma cannot change
    that: it controls where the drawn mass lands (d sf / d position), not which velocity cells the
    trajectory reads (d position / dv). Measured support is identical for sigma 0.7 ... 20.

    Pre-smoothing inserts a linear operator in front of that sampling. Its transpose spreads each
    thread's gradient over the whole kernel, so every cell within ~2*sigma of the path receives
    signal (measured: 6.5% -> 29% of cells at sigma 4, -> 50% at sigma 8). The forward meaning is
    coarse-grained advection: trajectories follow the smoothed flow. Anneal sigma -> 0 during
    training to end up tracing the true field again.

    Unlike the drawing kernel this one is normalized and replicate-padded: it is a weighted average,
    so a constant field must come back unchanged, including at the domain border (zero padding would
    drag boundary velocities toward 0 and bend every line that runs near the edge).
    """
    if sigma <= 0:
        return v
    radius = max(int(np.ceil(2 * sigma)), 1)
    r = torch.arange(-radius, radius + 1, dtype=v.dtype, device=v.device)
    k = torch.exp(-r ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    shape = v.shape
    x = v.reshape(-1, 1, shape[-2], shape[-1])
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return x.reshape(shape)

def draw_streamlines_soft(streamlines, dims, faded:bool=False, sigma:float=0.7, window:int=None,
                          fade_mode:str="absolute", t_end:float=27.5, method:str="auto"):
    # Differentiable version of draw_streamlines: instead of setting single cells (gradient zero
    # almost everywhere), every sample spreads a normalized Gaussian of width sigma. Samples are
    # weighted by their arc length ds, so a cell's density is the faded line length crossing it
    # (independent of the time-sampling density) and does not saturate where samples are dense.
    # occupancy = 1 - exp(-density) keeps values in [0,1).
    # fade_mode "absolute" (default): fade = 1 - t/t_end - independent of where a line is cut,
    # so patch/window training and full-domain inference see identical channel semantics.
    # "per_line" (legacy, baseline comparability): fade 1 -> 0 over each line's visible segment,
    # the paper convention used by the offline hard-drawn datasets.
    #
    # Two ways to rasterize the same field, selected by `method`:
    #  "splat": evaluate the Gaussian at every cell of each sample's window x window neighborhood.
    #      Exact, but allocates n_samples x window^2 = n_samples x (4 sigma + 1)^2 entries per line
    #      and *keeps them in the autograd graph* -> memory grows quadratically in sigma and runs a
    #      24 GB card out of memory around sigma >~ 4 on full-domain data.
    #  "blur": a Gaussian splat is a point mass convolved with a Gaussian, so scatter each sample
    #      bilinearly (4 cells) and convolve the accumulated density once, separably. Memory per
    #      sample is constant in sigma and the convolution is one cheap pass over the grid; the
    #      backward is again a convolution. Gradients w.r.t. sample position flow through the
    #      bilinear weights and are then smoothed by the blur.
    #  "auto" (default): "blur" from sigma >= SIGMA_BLUR_MIN, "splat" below, where the splat path is
    #      cheap and its exact sub-cell kernel still matters relative to the blob size.
    # Bilinear deposition is itself a convolution with a triangular kernel of variance 1/6 per axis,
    # so the blur uses sqrt(sigma^2 - 1/6) to land on total width sigma - the two paths then agree
    # to well under a percent across the switch (the correction is 0.008% at sigma=10).
    if window is None:
        window = 2*int(np.ceil(2*sigma)) + 1  # cover +-2 sigma
    use_blur = method == "blur" or (method == "auto" and sigma >= SIGMA_BLUR_MIN)
    if method not in ("auto", "blur", "splat"):
        raise ValueError(f"unknown drawing method {method!r} (expected 'auto', 'blur' or 'splat')")
    device = streamlines[0][0].device if streamlines else "cpu"
    dtype = streamlines[0][0].dtype if streamlines else torch.float32
    density = torch.zeros(tuple(dims), dtype=dtype, device=device)
    half = window // 2
    if not use_blur:
        offs = torch.arange(-half, half+1, dtype=dtype, device=device)
        off_i, off_j = torch.meshgrid(offs, offs, indexing='ij')
        off_i = off_i.reshape(1,-1)
        off_j = off_j.reshape(1,-1)
    for sol_x, sol_y, t in streamlines:
        if len(t) < 2:
            continue
        seg = ((sol_x[1:]-sol_x[:-1])**2 + (sol_y[1:]-sol_y[:-1])**2 + 1e-12).sqrt()
        zero = seg.new_zeros(1)
        ds = (torch.cat([seg, zero]) + torch.cat([zero, seg])) / 2  # arc length per sample
        if not faded:
            fade = torch.ones_like(t)
        elif fade_mode == "absolute":
            fade = (1 - t/t_end).clamp(min=0)
        else:  # per_line
            fade = 1 - t/t[-1]
        if use_blur:
            # unit point masses now, Gaussian applied to the summed density below
            _scatter_bilinear(density, sol_x, sol_y, fade*ds, dims)
            continue
        # cells around each sample; which cell a blob lands in is discrete -> detached,
        # the smooth kernel below carries the gradient
        cells_i = (sol_x + 0.5).floor().detach().unsqueeze(1) + off_i
        cells_j = (sol_y + 0.5).floor().detach().unsqueeze(1) + off_j
        d2 = (sol_x.unsqueeze(1) - cells_i)**2 + (sol_y.unsqueeze(1) - cells_j)**2
        w = torch.exp(-d2/(2*sigma**2)) / (2*np.pi*sigma**2) * (fade*ds).unsqueeze(1)
        mask = (cells_i >= 0) & (cells_i < dims[0]) & (cells_j >= 0) & (cells_j < dims[1])
        density.index_put_((cells_i[mask].long(), cells_j[mask].long()), w[mask], accumulate=True)
    if use_blur:
        density = _gaussian_blur_separable(density, np.sqrt(max(sigma**2 - 1/6, 1e-6)), half)
    return 1 - torch.exp(-density)

def trace_and_draw_soft(hp_positions, vx, vy, dims, offsets:list=(0,), randomK_data:bool=False,
                        faded:bool=True, t_steps:int=10_000, sigma:float=1.0, use_compile:bool=False,
                        fade_mode:str="absolute", method:str="auto"):
    # Differentiable forward pass of step 2: trace streamlines for all offsets in one batch and
    # rasterize each offset group to a soft-occupancy image (one image per offset).
    # No detach/no_grad: gradients flow from the images back to vx, vy (physical velocities in
    # m/y, sampled on the grid; tensors may live on any device, computation follows them).
    resolution = 5
    hp_positions = torch.as_tensor(hp_positions, dtype=torch.float32)
    n_hps = hp_positions.shape[0]
    if n_hps == 0:
        return [torch.zeros(tuple(dims)) for _ in offsets]
    device = vx.device if torch.is_tensor(vx) else "cpu"
    hp_positions = hp_positions.to(device)
    starts = torch.cat([hp_positions + torch.tensor([0., float(o)], device=device) for o in offsets])

    velocity = build_velocity_grid(vx/resolution, vy/resolution, dims, randomK_data=randomK_data)
    streamlines = calc_streamlines(starts, velocity, (dims[0]-1, dims[1]-1), t_end=27.5, t_steps=t_steps, use_compile=use_compile)
    return [draw_streamlines_soft(streamlines[i*n_hps:(i+1)*n_hps], dims, faded=faded, sigma=sigma,
                                  fade_mode=fade_mode, method=method)
            for i in range(len(offsets))]

def make_streamlines_gradients(mat_ids, vx, vy, dims, offset:float=None, randomK_data:bool=False, faded:bool=True,
                               t_steps:int=2000, sigma:float=0.7, loss_fn=None, device=None,
                               fade_mode:str="absolute", **kwargs):
    # Gradient of the drawn streamlines (occupancy) w.r.t. the velocity fields:
    # back-propagates loss_fn(occupancy) (default: total occupancy) through the soft drawing,
    # the RK4 integration and the bilinear velocity interpolation.
    # Runs on `device` (default: cuda if available); results are returned on the CPU.
    # Returns (occupancy image, dloss/dvx, dloss/dvy), each with shape `dims`.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_hps = torch.nonzero(torch.as_tensor(mat_ids) == 2).float()
    if pos_hps.shape[0] == 0:
        return torch.zeros(tuple(dims)), torch.zeros(tuple(dims)), torch.zeros(tuple(dims))
    pos_hps += torch.tensor([0.5,0.5]) # cell-center offset

    # requires_grad_ must come after .to(device), so vx/vy stay leaf tensors with a .grad
    vx = torch.as_tensor(vx, dtype=torch.float32).detach().clone().to(device).requires_grad_(True)
    vy = torch.as_tensor(vy, dtype=torch.float32).detach().clone().to(device).requires_grad_(True)
    occupancy = trace_and_draw_soft(pos_hps, vx, vy, dims, offsets=(0 if offset is None else offset,),
                                    randomK_data=randomK_data, faded=faded, t_steps=t_steps, sigma=sigma,
                                    fade_mode=fade_mode)[0]

    loss = occupancy.sum() if loss_fn is None else loss_fn(occupancy)
    loss.backward()
    return occupancy.detach().cpu(), vx.grad.cpu(), vy.grad.cpu()

def make_streamlines(mat_ids, vx, vy, dims, offset:float=None, offsets:list=None, randomK_data:bool=False, faded:bool=True, **kwargs):
    # `offsets`: several start-offsets traced together in one batch (much faster than separate
    # calls), returns one drawn image per offset. `offset`: single offset, returns one image.
    single = offsets is None
    if single:
        offsets = [0 if offset is None else offset]
    pos_hps = torch.nonzero(torch.as_tensor(mat_ids) == 2).float()
    print("Number of heat pumps: ", pos_hps.shape[0])
    if pos_hps.shape[0] == 0:
        images = [torch.zeros(tuple(dims)) for _ in offsets]
        return images[0] if single else images
    pos_hps += torch.tensor([0.5,0.5]) # cell-center offset
    resolution = 5
    starts = torch.cat([pos_hps + torch.tensor([0.,float(o)]) for o in offsets])

    time = datetime.now()
    with torch.no_grad():
        velocity = build_velocity_grid(torch.as_tensor(vx)/resolution, torch.as_tensor(vy)/resolution, dims, randomK_data=randomK_data)
        streamlines = calc_streamlines(starts, velocity, (dims[0]-1, dims[1]-1), t_end=27.5, t_steps=10_000)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    n_hps = pos_hps.shape[0]
    images = [draw_streamlines(torch.zeros(tuple(dims)), streamlines[i*n_hps:(i+1)*n_hps], faded=faded) for i in range(len(offsets))]
    return images[0] if single else images

def save_new_datapoint(destination, runid:str, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs" / runid)
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / runid)

def correct_args_info(destination:Path, v_info_path:Path=None, based_on_predicted_v:bool=False):
    info = load_yaml(destination / "info.yaml")
    info_v = load_yaml(v_info_path / "info.yaml")
    info_vx = info_v["Labels"]["Liquid X-Velocity [m_per_y]"]
    info_vy = info_v["Labels"]["Liquid Y-Velocity [m_per_y]"]
    build_new_info(info, info_vx, info_vy)
    save_yaml(info, destination / "info.yaml")

    args = load_yaml(destination / "args.yaml")
    if based_on_predicted_v:
        build_new_args(args, v_info_path)
    else:
        build_new_args(args)
    save_yaml(args, destination / "args.yaml")

def extend_inputs_dims(inputs_normed):
    dummy_field = torch.zeros_like(inputs_normed[0])
    inputs_new = torch.cat([inputs_normed[:3], dummy_field.unsqueeze(0), inputs_normed[3].unsqueeze(0), dummy_field.unsqueeze(0)], dim=0)
    inputs_new = inputs_new.float()

    return inputs_new
