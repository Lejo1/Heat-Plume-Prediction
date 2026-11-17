import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
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
    # Compiles the RK4 step once per process (fuses its many small ops); falls back to eager on failure.
    global _rk4_step_compiled, _compile_failed
    if not use_compile or _compile_failed:
        return rk4_step
    if _rk4_step_compiled is None:
        try:
            # not mode="reduce-overhead": its CUDA graphs are slower than eager for this step
            _rk4_step_compiled = torch.compile(rk4_step)
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}), running RK4 without compilation")
            _compile_failed = True
            return rk4_step
    return _rk4_step_compiled

def calc_streamlines(start_points, velocity, maxs_xy, t_end=27.5, t_steps=10_000, max_step_cells=0.5,
                     use_compile:bool=False, detach_trajectory:bool=False):
    # Fixed-step RK4 for all start points at once, with at most max_step_cells movement per step;
    # the result is linearly upsampled to t_steps samples. Differentiable w.r.t. the velocity grid.
    # detach_trajectory: cut the gradient between steps (see the loop below).
    global _compile_failed
    x = torch.as_tensor(start_points, dtype=velocity.dtype).clone()  # dtype follows the velocity grid
    v_max = float(velocity.detach().norm(dim=0).max())  # only picks the step count, no gradient needed
    n_int = max(min(int(np.ceil(t_end * v_max / max_step_cells)), t_steps - 1), 16)
    dt = t_end / n_int

    step_fn = get_rk4_step(use_compile)
    # 0-d tensor for the compiled path, so a new dt does not trigger a recompilation
    dt_arg = torch.tensor(dt, dtype=velocity.dtype, device=x.device) if step_fn is not rk4_step else dt

    trajectory = torch.empty((x.shape[0], n_int+1, 2), dtype=x.dtype, device=x.device)
    trajectory[:,0] = x
    for i in range(n_int):
        if detach_trajectory:
            # Truncated gradient: no backprop through dx_{n+1}/dx_n, so each step's dL/dv only
            # reaches the velocity sampled in that step. The forward values are unchanged.
            x = x.detach()
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

# From this sigma on, method "auto" draws with the blur path instead of the splat path (the splat
# path has a much smoother gradient w.r.t. line position, see check_drawing_gradient.py).
SIGMA_BLUR_MIN = 1.5
# Half-width of the drawing kernel in units of sigma (the cut tail is < 0.03 % of the peak).
KERNEL_HALF_WIDTHS = 4.0
# Kernel entries per splat chunk, bounds the memory of one chunk to ~0.4 GB.
SPLAT_CHUNK_ENTRIES = 2**24


class _GaussianSplat(torch.autograd.Function):
    """density[c] = sum_s amp_s * exp(-|p_s - c|^2 / 2 sigma^2) / (2 pi sigma^2), cut to a
    window x window neighborhood of each sample.

    Only positions and amplitudes are saved; the backward recomputes the kernel chunk by chunk,
    so memory is O(n_samples). Which cells a sample covers gets no gradient.
    """

    @staticmethod
    def _offsets(window, dtype, device):
        half = window // 2
        o = torch.arange(-half, half + 1, dtype=dtype, device=device)
        oi, oj = torch.meshgrid(o, o, indexing="ij")
        return oi.reshape(1, -1), oj.reshape(1, -1)

    @staticmethod
    def _kernel(x, y, offs, sigma, dims):
        """Flat cell index, kernel value and (dx, dy) for one chunk of samples, each [n, window^2]."""
        ci = (x + 0.5).floor().unsqueeze(1) + offs[0]
        cj = (y + 0.5).floor().unsqueeze(1) + offs[1]
        dx, dy = x.unsqueeze(1) - ci, y.unsqueeze(1) - cj
        inside = (ci >= 0) & (ci < dims[0]) & (cj >= 0) & (cj < dims[1])
        k = torch.exp(-(dx * dx + dy * dy) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) * inside
        idx = torch.where(inside, ci.long() * dims[1] + cj.long(), 0)
        return idx, k, dx, dy

    @staticmethod
    def forward(ctx, x, y, amp, dims, sigma, window):
        offs = _GaussianSplat._offsets(window, x.dtype, x.device)
        step = max(SPLAT_CHUNK_ENTRIES // (window * window), 1)
        density = torch.zeros(dims[0] * dims[1], dtype=x.dtype, device=x.device)
        for s in range(0, x.shape[0], step):
            idx, k, _, _ = _GaussianSplat._kernel(x[s:s+step], y[s:s+step], offs, sigma, dims)
            density.index_add_(0, idx.reshape(-1), (k * amp[s:s+step].unsqueeze(1)).reshape(-1))
        ctx.save_for_backward(x, y, amp)
        ctx.dims, ctx.sigma, ctx.window, ctx.step = dims, sigma, window, step
        return density.view(dims[0], dims[1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_density):
        x, y, amp = ctx.saved_tensors
        g = grad_density.reshape(-1)
        need_x, need_y, need_amp = ctx.needs_input_grad[:3]
        gx = torch.zeros_like(x) if need_x else None
        gy = torch.zeros_like(y) if need_y else None
        gamp = torch.zeros_like(amp) if need_amp else None
        offs = _GaussianSplat._offsets(ctx.window, x.dtype, x.device)
        step = ctx.step
        for s in range(0, x.shape[0], step):
            sl = slice(s, s + step)
            idx, k, dx, dy = _GaussianSplat._kernel(x[sl], y[sl], offs, ctx.sigma, ctx.dims)
            gk = g[idx] * k  # dL/ddensity at every covered cell, times the kernel value there
            if need_amp:
                gamp[sl] = gk.sum(1)
            if need_x or need_y:
                gk = gk * (amp[sl] / -ctx.sigma ** 2).unsqueeze(1)
                if need_x:
                    gx[sl] = (gk * dx).sum(1)
                if need_y:
                    gy[sl] = (gk * dy).sum(1)
        return gx, gy, gamp, None, None, None

def _scatter_bilinear(density, x, y, amp, dims):
    # Deposit each sample's mass on the 4 surrounding cell centers; the bilinear weights carry the
    # gradient w.r.t. position.
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
    # The splat path's Gaussian as two 1D passes (same truncation, zero padding outside the domain).
    r = torch.arange(-radius, radius+1, dtype=img.dtype, device=img.device)
    k = torch.exp(-r**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    out = img.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(out, k.view(1, 1, -1, 1), padding=(radius, 0))
    out = F.conv2d(out, k.view(1, 1, 1, -1), padding=(0, radius))
    return out[0, 0]

def smooth_velocity_field(v, sigma: float):
    """Gaussian smoothing of a velocity field [..., H, W] before tracing.

    The tracer's gradient dL/dv only reaches the cells next to each line; smoothing spreads it to
    a band of ~2 sigma around the line. Normalized kernel with replicate padding, so a constant
    field stays unchanged.
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
    # Differentiable version of draw_streamlines: every sample adds a normalized Gaussian of width
    # sigma, weighted by its arc length ds; occupancy = 1 - exp(-density) lies in [0, 1).
    # fade_mode "absolute": fade = 1 - t/t_end, independent of where a line is cut (patches);
    #           "per_line": fade 1 -> 0 over each line's visible part (as in the hard-drawn datasets).
    # method "splat": evaluate the Gaussian around each sample (smooth gradient w.r.t. position);
    #        "blur": scatter bilinearly and convolve once (cheaper for large sigma, rougher gradient);
    #        "auto": blur from sigma >= SIGMA_BLUR_MIN, splat below.
    if window is None:
        window = 2*int(np.ceil(KERNEL_HALF_WIDTHS*sigma)) + 1  # cover +-KERNEL_HALF_WIDTHS sigma
    use_blur = method == "blur" or (method == "auto" and sigma >= SIGMA_BLUR_MIN)
    if method not in ("auto", "blur", "splat"):
        raise ValueError(f"unknown drawing method {method!r} (expected 'auto', 'blur' or 'splat')")
    device = streamlines[0][0].device if streamlines else "cpu"
    dtype = streamlines[0][0].dtype if streamlines else torch.float32
    density = torch.zeros(tuple(dims), dtype=dtype, device=device)
    half = window // 2
    splat_x, splat_y, splat_amp = [], [], []  # splat path: all lines' samples, drawn in one call
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
        splat_x.append(sol_x); splat_y.append(sol_y); splat_amp.append(fade*ds)
    if splat_x:
        # the gradient w.r.t. amp flows on through ds into the positions via ordinary autograd
        density = density + _GaussianSplat.apply(torch.cat(splat_x), torch.cat(splat_y), torch.cat(splat_amp),
                                                 (int(dims[0]), int(dims[1])), float(sigma), window)
    if use_blur:
        # the bilinear deposition already adds variance 1/6 per axis
        density = _gaussian_blur_separable(density, np.sqrt(max(sigma**2 - 1/6, 1e-6)), half)
    return 1 - torch.exp(-density)

STREAMLINE_CELL_SIZE = 5  # [m]: velocities in m/y become cells/y
STREAMLINE_T_END = 27.5   # [y]: integration time of every streamline

def trace_streamlines(starts, vx, vy, dims, randomK_data:bool=False, t_steps:int=10_000,
                      use_compile:bool=False, detach_trajectory:bool=False):
    # Differentiable trace of one streamline per start point (cell coordinates) through the physical
    # velocities vx, vy [m/y] on a dims-sized grid; returns calc_streamlines' (x, y, t) per line.
    velocity = build_velocity_grid(vx/STREAMLINE_CELL_SIZE, vy/STREAMLINE_CELL_SIZE, dims, randomK_data=randomK_data)
    return calc_streamlines(starts, velocity, (dims[0]-1, dims[1]-1), t_end=STREAMLINE_T_END, t_steps=t_steps,
                            use_compile=use_compile, detach_trajectory=detach_trajectory)

def trace_and_draw_soft(hp_positions, vx, vy, dims, offsets:list=(0,), randomK_data:bool=False,
                        faded:bool=True, t_steps:int=10_000, sigma:float=1.0, use_compile:bool=False,
                        fade_mode:str="absolute", method:str="auto", detach_trajectory:bool=False):
    # Differentiable step 2: trace all offsets in one batch and draw one soft image per offset.
    # Gradients flow from the images back to vx, vy [m/y].
    hp_positions = torch.as_tensor(hp_positions, dtype=torch.float32)
    n_hps = hp_positions.shape[0]
    if n_hps == 0:
        return [torch.zeros(tuple(dims)) for _ in offsets]
    device = vx.device if torch.is_tensor(vx) else "cpu"
    hp_positions = hp_positions.to(device)
    starts = torch.cat([hp_positions + torch.tensor([0., float(o)], device=device) for o in offsets])

    streamlines = trace_streamlines(starts, vx, vy, dims, randomK_data=randomK_data, t_steps=t_steps,
                                    use_compile=use_compile, detach_trajectory=detach_trajectory)
    return [draw_streamlines_soft(streamlines[i*n_hps:(i+1)*n_hps], dims, faded=faded, sigma=sigma,
                                  fade_mode=fade_mode, method=method)
            for i in range(len(offsets))]

def make_streamlines_gradients(mat_ids, vx, vy, dims, offset:float=None, randomK_data:bool=False, faded:bool=True,
                               t_steps:int=2000, sigma:float=0.7, loss_fn=None, device=None,
                               fade_mode:str="absolute", **kwargs):
    # Gradient of loss_fn(occupancy) (default: total occupancy) w.r.t. vx, vy.
    # Returns (occupancy, dloss/dvx, dloss/dvy) on the CPU, each with shape `dims`.
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
