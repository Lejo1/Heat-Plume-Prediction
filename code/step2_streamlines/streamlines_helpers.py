import numpy as np
import torch
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
def build_velocity_grid(vx, vy, dims, randomK_data:bool=False):
    vx = torch.as_tensor(vx, dtype=torch.float32)
    vy = torch.as_tensor(vy, dtype=torch.float32)
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

@torch.no_grad()
def calc_streamlines(start_points, velocity, maxs_xy, t_end=27.5, t_steps=1000, max_step_cells=0.5):
    # Solve for all start points at once with fixed-step RK4. The step count is chosen so that
    # no point moves more than max_step_cells per step; the coarse solution is then linearly
    # upsampled to t_steps samples for drawing.
    x = torch.as_tensor(start_points, dtype=torch.float32).clone()
    v_max = float(velocity.norm(dim=0).max())
    n_int = max(min(int(np.ceil(t_end * v_max / max_step_cells)), t_steps - 1), 16)
    dt = t_end / n_int

    trajectory = torch.empty((x.shape[0], n_int+1, 2))
    trajectory[:,0] = x
    for i in range(n_int):
        k1 = sample_velocity(velocity, x)
        k2 = sample_velocity(velocity, x + dt/2*k1)
        k3 = sample_velocity(velocity, x + dt/2*k2)
        k4 = sample_velocity(velocity, x + dt*k3)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        trajectory[:,i+1] = x

    # linear upsampling to t_steps samples, vectorized over all lines
    t = torch.linspace(0, t_end, t_steps)
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
